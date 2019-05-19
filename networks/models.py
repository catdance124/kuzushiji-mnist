from keras.layers import Input, Dense, BatchNormalization, Activation,\
                          Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from .modules import _resblock, _resblock_bottleneck, _resblock_octconv, OctConv2D

# GPU memory settings----------------------
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
# -----------------------------------------

# -----------------------------------------
# wideresnet https://qiita.com/Phoeboooo/items/a1ce1dae73623f3adacc
# -----------------------------------------
def wideresnet(N=3, k=2, SE=False):
  inputs = Input(shape=(28, 28, 1))
  x = Conv2D(16, (7,7), strides=(1,1), kernel_initializer='he_normal', padding='same')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3, 3), strides=(2,2), padding='same')(x)
  for i in range(N):
    x = _resblock(n_filters=16*k, SE=SE)(x)
  x = MaxPooling2D(strides=(2,2))(x)
  for i in range(N):
    x = _resblock(n_filters=32*k, SE=SE)(x)
  x = MaxPooling2D(strides=(2,2))(x)
  for i in range(N):
    x = _resblock(n_filters=64*k, SE=SE)(x)

  x = GlobalAveragePooling2D()(x)
  x = Dense(10, kernel_initializer='he_normal', activation='softmax')(x)

  model = Model(inputs=inputs, outputs=x)
  return model

def wideresnet_octconv(alpha=0.5, N=3, k=2):
  inputs = Input(shape=(28, 28, 1))
  # downsampling for lower
  low = MaxPooling2D(2, padding='same')(inputs)
  # octconv
  high, low = OctConv2D(filters=16, alpha=alpha)([inputs, low])
  # (high & low) BN & relu
  high = BatchNormalization()(high)
  high = Activation("relu")(high)  # (28,28)
  low = BatchNormalization()(low)
  low = Activation("relu")(low)  # (14,14)

  for i in range(N):
    high, low = _resblock_octconv(n_filters=16*k, alpha=alpha)([high, low])
  high = MaxPooling2D(2, padding='same')(high)  # (14,14)
  low = MaxPooling2D(2, padding='same')(low)  # (7,7)

  for i in range(N):
    high, low = _resblock_octconv(n_filters=32*k, alpha=alpha)([high, low])
  high = MaxPooling2D(3, 2, padding='valid')(high)  # (6,6)  KEEP high//low == 2
  low = MaxPooling2D(3, 2, padding='valid')(low)  # (3,3)

  for i in range(N-1):
    high, low = _resblock_octconv(n_filters=64*k, alpha=alpha)([high, low])
  x = _resblock_octconv(n_filters=64*k, alpha=alpha, last=True)([high, low])

  x = GlobalAveragePooling2D()(x)
  x = Dense(10, kernel_initializer='he_normal', activation='softmax')(x)

  model = Model(inputs=inputs, outputs=x)
  return model

# -----------------------------------------
# bottleneck https://qiita.com/Phoeboooo/items/a1ce1dae73623f3adacc
# -----------------------------------------
def pyramidnet_bottleneck(SE=False):
  inputs = Input(shape=(28, 28, 1))
  x = Conv2D(16, (7,7), strides=(1,1), kernel_initializer='he_normal', padding='same')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3, 3), strides=(2,2), padding='same')(x)

  for i in range(9):
    x = _resblock_bottleneck(n_filters1=8+4*i, n_filters2=32+16*i, SE=SE)(x)  # -> (n_filters1=40, n_filters2=160)
  x = MaxPooling2D(strides=(2,2))(x) 
  for i in range(9,15):
    x = _resblock_bottleneck(n_filters1=8+4*i, n_filters2=32+16*i, SE=SE)(x)  # -> (n_filters1=64, n_filters2=256)
  
  x = GlobalAveragePooling2D()(x)
  x = Dense(10, kernel_initializer='he_normal', activation='softmax')(x)

  model = Model(inputs=inputs, outputs=x)
  return model