from keras.layers import Input, Dense, Flatten, Dropout, \
Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, Add
from keras.models import Model
import keras

import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# resnet https://qiita.com/Phoeboooo/items/a1ce1dae73623f3adacc
def _shortcut(inputs, residual):
  # _keras_shape[3] チャンネル数
  n_filters = residual._keras_shape[3]
  # inputs と residual とでチャネル数が違うかもしれない。
  # そのままだと足せないので、1x1 conv を使って residual 側のフィルタ数に合わせている
  shortcut = Conv2D(n_filters, (1,1), strides=(1,1), padding='valid')(inputs)
  # 2つを足す
  return Add()([shortcut, residual])

def _resblock(n_filters1, n_filters2, strides=(1,1)):
  def f(input):    
    x = Conv2D(n_filters1, (1,1), strides=strides,
            kernel_initializer='he_normal', padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters1, (3,3), strides=strides,
            kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(n_filters2, (1,1), strides=strides,
            kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    return _shortcut(input, x)
  return f

def resnet():
  inputs = Input(shape=(28, 28, 1))
  x = Conv2D(16, (7,7), strides=(1,1),
                    kernel_initializer='he_normal', padding='same')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3, 3), strides=(2,2), padding='same')(x)

  x = _resblock(n_filters1=8, n_filters2=32)(x)
  x = _resblock(n_filters1=12, n_filters2=48)(x)
  x = _resblock(n_filters1=16, n_filters2=64)(x)
  x = _resblock(n_filters1=20, n_filters2=80)(x)
  x = _resblock(n_filters1=24, n_filters2=96)(x)
  x = _resblock(n_filters1=28, n_filters2=112)(x)
  x = _resblock(n_filters1=32, n_filters2=128)(x)
  x = _resblock(n_filters1=36, n_filters2=144)(x)
  x = _resblock(n_filters1=40, n_filters2=160)(x)
  x = MaxPooling2D(strides=(2,2))(x) 
  x = _resblock(n_filters1=44, n_filters2=176)(x)
  x = _resblock(n_filters1=48, n_filters2=192)(x)
  x = _resblock(n_filters1=52, n_filters2=208)(x) 
  x = _resblock(n_filters1=56, n_filters2=224)(x)
  x = _resblock(n_filters1=60, n_filters2=240)(x)
  x = _resblock(n_filters1=64, n_filters2=256)(x) 

  x = GlobalAveragePooling2D()(x)
  x = Dense(10, kernel_initializer='he_normal', activation='softmax')(x)

  model = Model(inputs=inputs, outputs=x)
  return model