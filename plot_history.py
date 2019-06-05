from matplotlib import pyplot as plt
import numpy as np
import csv

def plot_history(history, begin=2, dir_name=None, csv_output=True):
  # csv output
  if csv_output and (dir_name is not None):
    values = []
    for key in history.history.keys():
      values.append(history.history[key])
    values = np.array(values)
    with open(f'./{dir_name}/history.csv', 'w') as f_handle:
      writer = csv.writer(f_handle, lineterminator="\n")
      writer.writerows([history.history.keys()])  # header
      np.savetxt(f_handle, values.T, fmt="%.6f", delimiter=',')
  

  val_exist = 'val_acc' in history.history.keys()
  # plot accuracy
  plt.title('model accuracy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.plot(history.history['acc'][begin:])
  if val_exist:
    plt.plot(history.history['val_acc'][begin:])
    plt.legend(['acc', 'val_acc'], loc='lower right')
  else:
    plt.legend(['acc'], loc='lower right')
  if dir_name is None:
    plt.show()
  else:
    plt.savefig(f'./{dir_name}/accuracy.png')

  plt.clf()
  
  # plot loss
  plt.title('model loss')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.plot(history.history['loss'][begin:])
  if val_exist:
    plt.plot(history.history['val_loss'][begin:])
    plt.legend(['loss', 'val_loss'], loc='upper right')
  else:
    plt.legend(['loss'], loc='upper right')
  if dir_name is None:
    plt.show()
  else:
    plt.savefig(f'{dir_name}/loss.png')

  plt.clf()