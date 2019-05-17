from keras.layers import Input, BatchNormalization, Activation, Add, Reshape, Multiply,\
    Conv2D, GlobalAveragePooling2D, Dense, Layer, Lambda
import keras.backend as K

def _shortcut(inputs, residual):
  # _keras_shape[3] チャンネル数
  n_filters = residual._keras_shape[3]
  # residual 側のフィルタ数に合わせる
  shortcut = Conv2D(n_filters, (1,1), strides=(1,1), padding='valid')(inputs)
  return Add()([shortcut, residual])

# -----------------------------------------
# single ReLU https://qiita.com/takedarts/items/fc6f6e96f2d0b7b55630
# https://arxiv.org/pdf/1610.02915.pdf
# -----------------------------------------
def _resblock(n_filters, strides=(1,1), SE=False):
  def f(input):
    x = BatchNormalization()(input)
    # x = Activation('relu')(x)
    x = Conv2D(n_filters, (3,3), strides=strides, kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, (3,3), strides=strides, kernel_initializer='he_normal', padding='same')(x)
    if SE:
      x = _SEblock(x, n_filters)
    return _shortcut(input, x)
  return f

# -----------------------------------------
# bottleneck https://qiita.com/Phoeboooo/items/a1ce1dae73623f3adacc
# https://arxiv.org/pdf/1610.02915.pdf
# -----------------------------------------
def _resblock_bottleneck(n_filters1, n_filters2, strides=(1,1), SE=False):
  def f(input):    
    x = BatchNormalization()(input)
    x = Conv2D(n_filters1, (1,1), strides=strides, kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters1, (3,3), strides=strides, kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters2, (1,1), strides=strides, kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    if SE:
      x = _SEblock(x, n_filters2)
    return _shortcut(input, x)
  return f

# -----------------------------------------
# SEblock https://github.com/yoheikikuta/senet-keras
# https://arxiv.org/pdf/1709.01507.pdf
# -----------------------------------------
def _SEblock(input, n_filters, ratio=8):
  # squeeze
  x = GlobalAveragePooling2D()(input)
  # excitation
  x = Dense(n_filters // ratio, activation='relu')(x)
  x = Dense(n_filters, activation='sigmoid')(x)
  x = Reshape((1,1,n_filters))(x)
  
  return Multiply()([input,x])


# -----------------------------------------
# Octave conv https://github.com/koshian2/OctConv-TFKeras/blob/master/oct_conv2d.py
# Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution
# https://arxiv.org/abs/1904.05049
# -----------------------------------------
class OctConv2D(Layer):
  def __init__(self, filters, alpha, kernel_size=(3,3), strides=(1,1), 
              padding="same", kernel_initializer='glorot_uniform',
              kernel_regularizer=None, kernel_constraint=None,
              **kwargs):
    """
    OctConv2D : Octave Convolution for image( rank 4 tensors)
    filters: # output channels for low + high
    alpha: Low channel ratio (alpha=0 -> High only, alpha=1 -> Low only)
    kernel_size : 3x3 by default, padding : same by default
    """
    assert alpha >= 0 and alpha <= 1
    assert filters > 0 and isinstance(filters, int)
    super().__init__(**kwargs)

    self.alpha = alpha
    self.filters = filters
    # optional values
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.kernel_initializer = kernel_initializer
    self.kernel_regularizer = kernel_regularizer
    self.kernel_constraint = kernel_constraint
    # -> Low Channels 
    self.low_channels = int(self.filters * self.alpha)
    # -> High Channles
    self.high_channels = self.filters - self.low_channels
    
  def build(self, input_shape):
    assert len(input_shape) == 2
    assert len(input_shape[0]) == 4 and len(input_shape[1]) == 4
    # Assertion for high inputs
    assert input_shape[0][1] // 2 >= self.kernel_size[0]
    assert input_shape[0][2] // 2 >= self.kernel_size[1]
    # Assertion for low inputs
    assert input_shape[0][1] // input_shape[1][1] == 2
    assert input_shape[0][2] // input_shape[1][2] == 2
    # channels last for TensorFlow
    assert K.image_data_format() == "channels_last"
    # input channels
    high_in = int(input_shape[0][3])
    low_in = int(input_shape[1][3])

    # High -> High
    self.high_to_high_kernel = self.add_weight(name="high_to_high_kernel", 
                                shape=(*self.kernel_size, high_in, self.high_channels),
                                initializer=self.kernel_initializer,
                                regularizer=self.kernel_regularizer,
                                constraint=self.kernel_constraint)
    # High -> Low
    self.high_to_low_kernel  = self.add_weight(name="high_to_low_kernel", 
                                shape=(*self.kernel_size, high_in, self.low_channels),
                                initializer=self.kernel_initializer,
                                regularizer=self.kernel_regularizer,
                                constraint=self.kernel_constraint)
    # Low -> High
    self.low_to_high_kernel  = self.add_weight(name="low_to_high_kernel", 
                                shape=(*self.kernel_size, low_in, self.high_channels),
                                initializer=self.kernel_initializer,
                                regularizer=self.kernel_regularizer,
                                constraint=self.kernel_constraint)
    # Low -> Low
    self.low_to_low_kernel   = self.add_weight(name="low_to_low_kernel", 
                                shape=(*self.kernel_size, low_in, self.low_channels),
                                initializer=self.kernel_initializer,
                                regularizer=self.kernel_regularizer,
                                constraint=self.kernel_constraint)
    super().build(input_shape)

  def call(self, inputs):
    # Input = [X^H, X^L]
    assert len(inputs) == 2
    high_input, low_input = inputs
    # High -> High conv
    high_to_high = K.conv2d(high_input, self.high_to_high_kernel,
                            strides=self.strides, padding=self.padding,
                            data_format="channels_last")
    # High -> Low conv
    high_to_low  = K.pool2d(high_input, (2,2), strides=(2,2), pool_mode="avg")
    high_to_low  = K.conv2d(high_to_low, self.high_to_low_kernel,
                            strides=self.strides, padding=self.padding,
                            data_format="channels_last")
    # Low -> High conv
    low_to_high  = K.conv2d(low_input, self.low_to_high_kernel,
                            strides=self.strides, padding=self.padding,
                            data_format="channels_last")
    low_to_high = K.repeat_elements(low_to_high, 2, axis=1) # Nearest Neighbor Upsampling
    low_to_high = K.repeat_elements(low_to_high, 2, axis=2)
    # Low -> Low conv
    low_to_low   = K.conv2d(low_input, self.low_to_low_kernel,
                            strides=self.strides, padding=self.padding,
                            data_format="channels_last")
    # Cross Add
    high_add = high_to_high + low_to_high
    low_add = high_to_low + low_to_low
    return [high_add, low_add]

  def compute_output_shape(self, input_shapes):
    high_in_shape, low_in_shape = input_shapes
    high_out_shape = (*high_in_shape[:3], self.high_channels)
    low_out_shape = (*low_in_shape[:3], self.low_channels)
    return [high_out_shape, low_out_shape]

  def get_config(self):
    base_config = super().get_config()
    out_config = {
        **base_config,
        "filters": self.filters,
        "alpha": self.alpha,
        "filters": self.filters,
        "kernel_size": self.kernel_size,
        "strides": self.strides,
        "padding": self.padding,
        "kernel_initializer": self.kernel_initializer,
        "kernel_regularizer": self.kernel_regularizer,
        "kernel_constraint": self.kernel_constraint,            
    }
    return out_config

def _resblock_octconv(n_filters, strides=(1,1), alpha=0.5, last=False):
  if not last:
    def f(inputs):
      high, low = inputs
      # define high&low percentage
      input_high = Conv2D(int(n_filters*(1-alpha)), (1,1), 
          strides=strides, kernel_initializer='he_normal', padding='same')(high)
      input_high = BatchNormalization()(input_high)
      input_high = Activation("relu")(input_high)
      input_low = Conv2D(int(n_filters*alpha), (1,1), 
          strides=strides, kernel_initializer='he_normal', padding='same')(low)
      input_low = BatchNormalization()(input_low)
      input_low = Activation("relu")(input_low)

      # 1st octconv -> each BN & relu
      high, low = OctConv2D(filters=n_filters, alpha=alpha)([high, low])
      high = BatchNormalization()(high)
      high = Activation("relu")(high)
      low = BatchNormalization()(low)
      low = Activation("relu")(low)
      
      # 2nd octconv -> each Add input
      high, low = OctConv2D(filters=n_filters, alpha=alpha)([high, low])
      high = _shortcut(input_high, high)
      low = _shortcut(input_low, low)

      return [high, low]
  else:
    def f(inputs):
      high, low = inputs
      high, low = OctConv2D(filters=n_filters, alpha=alpha)([high, low])
      high = BatchNormalization()(high)
      high = Activation("relu")(high)
      low = BatchNormalization()(low)
      low = Activation("relu")(low)

      # Last conv layers = alpha_out = 0 : vanila Conv2D
      # high -> high      
      high_to_high = Conv2D(n_filters, (3,3), strides=strides, kernel_initializer='he_normal', padding='same')(high)
      # low -> high
      low_to_high = Conv2D(n_filters, (3,3), strides=strides, kernel_initializer='he_normal', padding='same')(low)
      low_to_high = Lambda(lambda x: K.repeat_elements(K.repeat_elements(x, 2, axis=1), 2, axis=2))(low_to_high)
      x = _shortcut(high_to_high, low_to_high)

      return x
  return f