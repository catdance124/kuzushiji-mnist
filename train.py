from loader import KMNISTDataLoader, LoadTestData
# from models.wideresnet_deep import resnet
from models.pyramidNet_bottleNeck_NoRelu import resnet
import keras
from DataAugmentation import MyImageDataGenerator, tta
from datetime import datetime
nowtime = datetime.now().strftime("%y%m%d_%H%M")
import glob, os
import pandas as pd
import numpy as np


if __name__ == '__main__':
  # load data
  datapath = "./data"
  validation_size = 0.15
  train_imgs, train_lbls, validation_imgs, validation_lbls = KMNISTDataLoader(validation_size).load(datapath)
  test_imgs = LoadTestData(datapath)

  # define model
  model = resnet()
  loss = keras.losses.categorical_crossentropy
  optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
  model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
  # model.summary()

  # data generator
  datagen = MyImageDataGenerator(
      rotation_range=20,
      width_shift_range=0.1,
      height_shift_range=0.1,
      #mix_up_alpha=1.2,
      #random_crop=(28, 28)
      random_erasing=True
  )

  # train
  os.makedirs(f'./{nowtime}', exist_ok=True)
  cp_cb = keras.callbacks.ModelCheckpoint(
      filepath = f'./{nowtime}'+'/weights.{epoch:04d}-{loss:.6f}-{acc:.6f}-{val_loss:.6f}-{val_acc:.6f}.hdf5' , 
      monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

  batch_size=128
  epochs=5000
  verbose=1
  steps_per_epoch = train_imgs.shape[0] // batch_size

  history = model.fit_generator(
      datagen.flow(train_imgs, train_lbls, batch_size=batch_size),
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      validation_data=(validation_imgs, validation_lbls),
      callbacks=[cp_cb],
      verbose=verbose)
  
  # test
  best_weight_path = sorted(glob.glob(f'./{nowtime}/*.hdf5'))[-1]
  model.load_weights(best_weight_path)
  
  predicts = tta(model, test_imgs)
  predict_labels = np.argmax(predicts, axis=1)

  # create submit file
  submit = pd.DataFrame(data={"ImageId": [], "Label": []})
  submit.ImageId = list(range(1, predict_labels.shape[0]+1))
  submit.Label = predict_labels

  submit.to_csv(f"./{nowtime}/submit{nowtime}.csv", index=False)