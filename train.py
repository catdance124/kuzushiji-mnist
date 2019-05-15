import keras
import glob, os
import pandas as pd
import numpy as np
import re
# import configparser
from datetime import datetime
dir_name = datetime.now().strftime("%y%m%d_%H%M")

# my modules
from loader import KMNISTDataLoader, LoadTestData
from dataAugmentation import MyImageDataGenerator, TTA
from networks.model import pyramidnet_bottleneck, wideresnet


if __name__ == '__main__':
  # load data
  datapath = "./data"
  validation_size = 0.15
  train_imgs, train_lbls, validation_imgs, validation_lbls = KMNISTDataLoader(validation_size).load(datapath)
  test_imgs = LoadTestData(datapath)

  # define model
  model = pyramidnet_bottleneck()
  loss = keras.losses.categorical_crossentropy
  optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
  model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
  # model.summary()

  # data generator
  datagen = MyImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.08,
    #mix_up_alpha=1.2,
    #random_crop=(28, 28),
    random_erasing=True,
  )

  # train setting
  batch_size = 128
  initial_epoch = 0
  epochs = 100
  steps_per_epoch = train_imgs.shape[0] // batch_size

  if os.path.exists(f'./{dir_name}'):
    best_weight_path = sorted(glob.glob(f'./{dir_name}/*.hdf5'))[-1]
    model.load_weights(best_weight_path)
    initial_epoch = re.search(r'weights.[0-9]{4}', best_weight_path)
    initial_epoch = int(initial_epoch.group().replace('weights.', ''))
  else:
    os.makedirs(f'./{dir_name}', exist_ok=True)

  if epochs > initial_epoch:
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1, cooldown=1, min_lr=0)
    cp = keras.callbacks.ModelCheckpoint(
        filepath = f'./{dir_name}'+'/weights.{epoch:04d}-{loss:.6f}-{acc:.6f}-{val_loss:.6f}-{val_acc:.6f}.hdf5',
        monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    history = model.fit_generator(
        datagen.flow(train_imgs, train_lbls, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        initial_epoch=initial_epoch,
        epochs=epochs,
        validation_data=(validation_imgs, validation_lbls),
        callbacks=[cp, reduce_lr],
        verbose=1,
    )
  
  # test
  best_weight_path = sorted(glob.glob(f'./{dir_name}/*.hdf5'))[-1]
  model.load_weights(best_weight_path)
  
  predicts = TTA(model, test_imgs, tta_steps=30)
  predict_labels = np.argmax(predicts, axis=1)

  # create submit file
  submit = pd.DataFrame(data={"ImageId": [], "Label": []})
  submit.ImageId = list(range(1, predict_labels.shape[0]+1))
  submit.Label = predict_labels

  submit.to_csv(f"./{dir_name}/submit{dir_name}.csv", index=False)