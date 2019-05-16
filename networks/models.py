from keras.layers import Input, Dense, BatchNormalization, Activation,\
                          Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from .modules import _resblock, _resblock_bottleneck

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
def wideresnet():
  inputs = Input(shape=(28, 28, 1))
  x = Conv2D(16, (7,7), strides=(1,1), kernel_initializer='he_normal', padding='same')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3, 3), strides=(2,2), padding='same')(x)
  N=3
  k=2
  for i in range(N):
    x = _resblock(n_filters=16*k)(x)
  x = MaxPooling2D(strides=(2,2))(x)
  for i in range(N):
    x = _resblock(n_filters=32*k)(x)
  x = MaxPooling2D(strides=(2,2))(x)
  for i in range(N):
    x = _resblock(n_filters=64*k)(x)

  x = GlobalAveragePooling2D()(x)
  x = Dense(10, kernel_initializer='he_normal', activation='softmax')(x)

  model = Model(inputs=inputs, outputs=x)
  return model

def SEwideresnet():
  inputs = Input(shape=(28, 28, 1))
  x = Conv2D(16, (7,7), strides=(1,1), kernel_initializer='he_normal', padding='same')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3, 3), strides=(2,2), padding='same')(x)
  N=3
  k=2
  for i in range(N):
    x = _resblock(n_filters=16*k, SE=True)(x)
  x = MaxPooling2D(strides=(2,2))(x)
  for i in range(N):
    x = _resblock(n_filters=32*k, SE=True)(x)
  x = MaxPooling2D(strides=(2,2))(x)
  for i in range(N):
    x = _resblock(n_filters=64*k, SE=True)(x)

  x = GlobalAveragePooling2D()(x)
  x = Dense(10, kernel_initializer='he_normal', activation='softmax')(x)

  model = Model(inputs=inputs, outputs=x)
  return model

# -----------------------------------------
# bottleneck https://qiita.com/Phoeboooo/items/a1ce1dae73623f3adacc
# -----------------------------------------
def pyramidnet_bottleneck():
  inputs = Input(shape=(28, 28, 1))
  x = Conv2D(16, (7,7), strides=(1,1), kernel_initializer='he_normal', padding='same')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3, 3), strides=(2,2), padding='same')(x)

  for i in range(9):
    x = _resblock_bottleneck(n_filters1=8+4*i, n_filters2=32+16*i)(x)  # -> (n_filters1=40, n_filters2=160)
  x = MaxPooling2D(strides=(2,2))(x) 
  for i in range(9,15):
    x = _resblock_bottleneck(n_filters1=8+4*i, n_filters2=32+16*i)(x)  # -> (n_filters1=64, n_filters2=256)
  
  x = GlobalAveragePooling2D()(x)
  x = Dense(10, kernel_initializer='he_normal', activation='softmax')(x)

  model = Model(inputs=inputs, outputs=x)
  return model

def SEpyramidnet_bottleneck():
  inputs = Input(shape=(28, 28, 1))
  x = Conv2D(16, (7,7), strides=(1,1), kernel_initializer='he_normal', padding='same')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3, 3), strides=(2,2), padding='same')(x)

  for i in range(9):
    x = _resblock_bottleneck(n_filters1=8+4*i, n_filters2=32+16*i, SE=True)(x)  # -> (n_filters1=40, n_filters2=160)
  x = MaxPooling2D(strides=(2,2))(x) 
  for i in range(9,15):
    x = _resblock_bottleneck(n_filters1=8+4*i, n_filters2=32+16*i, SE=True)(x)  # -> (n_filters1=64, n_filters2=256)
  
  x = GlobalAveragePooling2D()(x)
  x = Dense(10, kernel_initializer='he_normal', activation='softmax')(x)

  model = Model(inputs=inputs, outputs=x)
  return model