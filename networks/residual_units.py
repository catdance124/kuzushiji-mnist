from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add

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
def _resblock(n_filters, strides=(1,1)):
  def f(input):
    x = BatchNormalization()(input)
    # x = Activation('relu')(x)
    x = Conv2D(n_filters, (3,3), strides=strides, kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters, (3,3), strides=strides, kernel_initializer='he_normal', padding='same')(x)
    return _shortcut(input, x)
  return f

# -----------------------------------------
# bottleneck https://qiita.com/Phoeboooo/items/a1ce1dae73623f3adacc
# https://arxiv.org/pdf/1610.02915.pdf
# -----------------------------------------
def _bottleneck(n_filters1, n_filters2, strides=(1,1)):
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
    return _shortcut(input, x)
  return f