import keras
import glob, os, argparse
import pandas as pd
import numpy as np
import re
from datetime import datetime
from adabound import AdaBound

# my modules
from loader import KMNISTDataLoader, LoadTestData
from generator import MyImageDataGenerator, TTA
from networks.models import *
from plot_history import plot_history

def main(args):
  # load data
  datapath = "./data"
  validation_size = 0.15
  train_imgs, train_lbls, validation_imgs, validation_lbls = KMNISTDataLoader(validation_size).load(datapath)
  test_imgs = LoadTestData(datapath)

  # dir setting
  setting = f'{args.model}_o{args.optimizer}_b{args.batchsize}_e{args.epochs}_f{args.factor}_p{args.patience}'
  dir_name = f'./out/{setting}'
  nowtime = datetime.now().strftime("%y%m%d_%H%M")
  if args.force:
    dir_name = f'{dir_name}_{nowtime}'
  if args.ensemble > 1:
    dir_name_base = f'{dir_name}_ensemble{args.ensemble}'
    models = []
    results = np.zeros((test_imgs.shape[0],10))

  # define model
  for i in range(args.ensemble):
    model = eval(f'{args.model}')
    loss = keras.losses.categorical_crossentropy
    if args.optimizer == 'adam':
      optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    if args.optimizer == 'adabound':
      optimizer = AdaBound(lr=1e-03, final_lr=0.1, gamma=1e-03, weight_decay=5e-4, amsbound=False)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    if args.ensemble > 1:
      models.append(model)

  # data generator
  datagen = MyImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.08,
    #mix_up_alpha=1.2,
    #random_crop=(28, 28),
    random_erasing=True,
  )

  # each model train
  for i in range(args.ensemble):
    # train setting
    batch_size = args.batchsize
    initial_epoch = args.initialepoch
    epochs = args.epochs
    steps_per_epoch = train_imgs.shape[0] // batch_size

    if epochs > initial_epoch:
      if args.ensemble > 1:
        dir_name = f'{dir_name_base}/{i}'
        model = models[i]
      if os.path.exists(f'./{dir_name}'):
        best_weight_path = sorted(glob.glob(f'./{dir_name}/*.hdf5'))[-1]
        model.load_weights(best_weight_path)
        initial_epoch = re.search(r'weights.[0-9]{4}', best_weight_path)
        initial_epoch = int(initial_epoch.group().replace('weights.', ''))
      else:
        os.makedirs(f'./{dir_name}', exist_ok=True)
      print(f'train:{dir_name}')

      reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
          factor=args.factor, patience=args.patience, verbose=1, cooldown=1, min_lr=0)
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
      plot_history(history, dir_name=dir_name)
  
  # test
  for i in range(args.ensemble):
    if args.ensemble > 1:
      dir_name = f'{dir_name_base}/{i}'
      model = models[i]
    print(f'test:{dir_name}')

    best_weight_path = sorted(glob.glob(f'./{dir_name}/*.hdf5'))[-1]
    model.load_weights(best_weight_path)
    
    predicts = TTA(model, test_imgs, tta_steps=144)
    np.save(f'./{dir_name}/predicts_vec.npy', predicts)
    if args.ensemble > 1:
      results += predicts

  if args.ensemble > 1:
    predict_labels = np.argmax(results, axis=1)
    dir_name = dir_name_base
  else:
    predict_labels = np.argmax(predicts, axis=1)

  # create submit file
  submit = pd.DataFrame(data={"ImageId": [], "Label": []})
  submit.ImageId = list(range(1, predict_labels.shape[0]+1))
  submit.Label = predict_labels
  submit.to_csv(f"./{dir_name}/submit{nowtime}_{setting}.csv", index=False)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default='pyramidnet_bottleneck(SE=False)')
  parser.add_argument('--optimizer', '-o', default='adam')
  parser.add_argument('--initialepoch', '-ie', type=int, default=0)
  parser.add_argument('--epochs', '-e', type=int, default=300)
  parser.add_argument('--batchsize', '-b', type=int, default=128)
  parser.add_argument('--factor', '-f', type=float, default=0.2)
  parser.add_argument('--patience', '-p', type=int, default=30)
  parser.add_argument('--ensemble', type=int, default=1)
  parser.add_argument('--force', action='store_true')
  args = parser.parse_args()

  main(args)