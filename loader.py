import numpy as np
import os
from tensorflow.keras.utils import to_categorical

class KMNISTDataLoader(object):
    """
    Example
    -------
    >>> kmnist_dl = KMNISTDataLoader()
    >>> datapath = "./data"
    >>> train_imgs, train_lbls, validation_imgs, validation_lbls = kmnist_dl.load(datapath)
    """
    def __init__(self, validation_size: float):
        """
        validation_size : float
        [0., 1.]
        ratio of validation data
        """
        self._basename_list = [
        'kmnist-train-imgs.npz',\
        'kmnist-train-labels.npz'
        ]
        self.validation_size = validation_size

    def load(self, datapath: str, random_seed: int=13) -> np.ndarray:
        filenames_list = self._make_filenames(datapath)
        data_list = [np.load(filename)['arr_0'] for filename in filenames_list]

        all_imgs, all_lbls = data_list

        # shuffle data
        np.random.seed(random_seed)
        perm_idx = np.random.permutation(len(all_imgs))
        all_imgs = all_imgs[perm_idx]
        all_lbls = all_lbls[perm_idx]

        # split train and validation
        validation_num = int(len(all_lbls)*self.validation_size)

        validation_imgs = all_imgs[:validation_num]
        validation_lbls = all_lbls[:validation_num]

        train_imgs = all_imgs[validation_num:]
        train_lbls = all_lbls[validation_num:]

        train_imgs, train_lbls, validation_imgs, validation_lbls = Preprocessor().transform(train_imgs, train_lbls, validation_imgs, validation_lbls)

        return train_imgs, train_lbls, validation_imgs, validation_lbls

    def _make_filenames(self, datapath: str) -> list:
        filenames_list = [os.path.join(datapath, basename) for basename in self._basename_list]
        return filenames_list

class Preprocessor(object):
    def transform(self, train_imgs, train_lbls, validation_imgs, validation_lbls):
        train_imgs, validation_imgs = self._convert_imgs_dtypes(train_imgs, validation_imgs)
        train_imgs, validation_imgs = self._convert_imgs_shape(train_imgs, validation_imgs)
        train_imgs, validation_imgs = self._normalize(train_imgs, validation_imgs)

        train_lbls, validation_lbls = self._to_categorical_labels(train_lbls, validation_lbls)
        return train_imgs, train_lbls, validation_imgs, validation_lbls

    def _convert_imgs_dtypes(self, train_imgs, validation_imgs):
        _train_imgs = train_imgs.astype('float32')
        _validation_imgs = validation_imgs.astype('float32')
        return _train_imgs, _validation_imgs

    def _convert_imgs_shape(self, train_imgs, validation_imgs):
        _train_imgs = train_imgs[:,:,:,np.newaxis]
        _validation_imgs = validation_imgs[:,:,:,np.newaxis]
        return _train_imgs, _validation_imgs

    def _normalize(self, train_imgs, validation_imgs):
        _train_imgs = train_imgs / 255.0
        _validation_imgs = validation_imgs / 255.0
        return _train_imgs, _validation_imgs

    def _to_categorical_labels(self, train_lbls, validation_lbls):
        label_num = len(np.unique(train_lbls))
        _train_lbls = to_categorical(train_lbls, label_num)
        _validation_lbls = to_categorical(validation_lbls, label_num)
        return _train_lbls, _validation_lbls

def LoadTestData(datapath):
  test_imgs = np.load(datapath+'/kmnist-test-imgs.npz')['arr_0']
  test_imgs = test_imgs.astype('float32')
  test_imgs = test_imgs[:,:,:,np.newaxis]
  test_imgs = test_imgs / 255.0
  return test_imgs