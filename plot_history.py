from matplotlib import pyplot as plt
import numpy as np

def plot_history(history, csv=True, dir_name=None):
  if csv and (dir_name is not None):
    values = []
    for key in history.keys():
      values.append(history[key][0])
    values = np.array([values])
    with open(f'{dir_name}/history.txt', 'a') as f_handle:
      # output header
      np.savetxt(f_handle, values, delimiter=',')
  
  # 精度の履歴をプロット
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend(['acc', 'val_acc'], loc='lower right')
  if dir_name is None:
    plt.show()
  else:
    plt.savefig(f'{dir_name}/accuracy.png')

  plt.clf()
  
  # 損失の履歴をプロット
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(['loss', 'val_loss'], loc='lower right')
  if dir_name is None:
    plt.show()
  else:
    plt.savefig(f'{dir_name}/loss.png')