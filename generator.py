from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tqdm import tqdm

class MyImageDataGenerator(ImageDataGenerator):
  # 参考 https://qiita.com/koshian2/items/909360f50e3dd5922f32
  def __init__(self, featurewise_center = False, 
              samplewise_center = False, 
              featurewise_std_normalization = False, 
              samplewise_std_normalization = False, 
              zca_whitening = False, 
              zca_epsilon = 1e-06, 
              rotation_range = 0.0, 
              width_shift_range = 0.0, 
              height_shift_range = 0.0, 
              brightness_range = None, 
              shear_range = 0.0, 
              zoom_range = 0.0, 
              channel_shift_range = 0.0, 
              fill_mode = 'nearest', 
              cval = 0.0, 
              horizontal_flip = False, 
              vertical_flip = False, 
              rescale = None, 
              preprocessing_function = None, 
              data_format = None, 
              validation_split = 0.0, 
              random_crop = None, 
              mix_up_alpha = 0.0,
              random_erasing = False):
    # 親クラスのコンストラクタ
    super().__init__(featurewise_center, samplewise_center, featurewise_std_normalization, samplewise_std_normalization, zca_whitening, zca_epsilon, rotation_range, width_shift_range, height_shift_range, brightness_range, shear_range, zoom_range, channel_shift_range, fill_mode, cval, horizontal_flip, vertical_flip, rescale, preprocessing_function, data_format, validation_split)
    # 拡張処理のパラメーター
    # Mix-up
    assert mix_up_alpha >= 0.0
    self.mix_up_alpha = mix_up_alpha
    # Random Crop
    assert random_crop == None or len(random_crop) == 2
    self.random_crop_size = random_crop
    self.random_erasing_flag = random_erasing

  # Mix-up
  # 参考 https://qiita.com/yu4u/items/70aa007346ec73b7ff05
  def mix_up(self, X1, y1, X2, y2):
    assert X1.shape[0] == y1.shape[0] == X2.shape[0] == y2.shape[0]
    batch_size = X1.shape[0]
    l = np.random.beta(self.mix_up_alpha, self.mix_up_alpha, batch_size)
    X_l = l.reshape(batch_size, 1, 1, 1)
    y_l = l.reshape(batch_size, 1)
    X = X1 * X_l + X2 * (1-X_l)
    y = y1 * y_l + y2 * (1-y_l)
    return X, y

  # ランダムクロップ
  # 参考 https://jkjung-avt.github.io/keras-image-cropping/
  def random_crop(self, original_img):
    # Note: image_data_format is 'channel_last'
    assert original_img.shape[2] == 1  # 3
    if original_img.shape[0] < self.random_crop_size[0] or original_img.shape[1] < self.random_crop_size[1]:
        raise ValueError(f"Invalid random_crop_size : original = {original_img.shape}, crop_size = {self.random_crop_size}")

    height, width = original_img.shape[0], original_img.shape[1]
    dy, dx = self.random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return original_img[y:(y+dy), x:(x+dx), :]
  
  # ランダムイレーシング
  # 参考 https://www.kumilog.net/entry/numpy-data-augmentation#Random-Erasing
  def random_erasing(self,image_origin, p=0.5, s=(0.02, 0.3), r=(0.3, 3)):
    # マスクするかしないか
    if np.random.rand() > p:
      return image_origin
    image = np.copy(image_origin)
    # マスクする画素値をランダムで決める
    mask_value = np.random.rand()
    h, w, _ = image.shape
    # マスクのサイズを元画像のs(0.02~0.4)倍の範囲からランダムに決める
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])
    # マスクのアスペクト比をr(0.3~3)の範囲からランダムに決める
    mask_aspect_ratio = np.random.rand() * r[1] + r[0]
    # マスクのサイズとアスペクト比からマスクの高さと幅を決める
    # 算出した高さと幅(のどちらか)が元画像より大きくなることがあるので修正する
    mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
    if mask_height > h - 1:
      mask_height = h - 1
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
      mask_width = w - 1

    top = np.random.randint(0, h - mask_height)
    left = np.random.randint(0, w - mask_width)
    bottom = top + mask_height
    right = left + mask_width
    image[top:bottom, left:right] = mask_value
    return image

  def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None):
    # 親クラスのflow_from_directory
    batches = super().flow(x, y, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, subset)
    # 拡張処理
    while True:
      batch_x, batch_y = next(batches)  # <-- 追加が必要ですね
      if self.mix_up_alpha > 0:
        while True:
          batch_x_2, batch_y_2 = next(batches)
          m1, m2 = batch_x.shape[0], batch_x_2.shape[0]
          if m1 < m2:
            batch_x_2 = batch_x_2[:m1]
            batch_y_2 = batch_y_2[:m1]
            break
          elif m1 == m2:
            break
        batch_x, batch_y = self.mix_up(batch_x, batch_y, batch_x_2, batch_y_2)
      # Random crop
      if self.random_crop_size != None:
        x = np.zeros((batch_x.shape[0], self.random_crop_size[0], self.random_crop_size[1], 1))
        for i in range(batch_x.shape[0]):
          x[i] = self.random_crop(batch_x[i])
        batch_x = x
      # Random erasing
      if self.random_erasing_flag:
        x = np.zeros_like(batch_x)
        for i in range(batch_x.shape[0]):
          x[i] = self.random_erasing(batch_x[i])
        batch_x = x
          
      # 返り値
      yield (batch_x, batch_y)

# tta https://towardsdatascience.com/test-time-augmentation-tta-and-how-to-perform-it-with-keras-4ac19b67fb4d
def TTA(model, X, batch_size=128, tta_steps=30):
  test_datagen = ImageDataGenerator(
      rotation_range=20,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.08,
      )
  predictions = []
  for i in tqdm(list(range(tta_steps))):
    preds = model.predict_generator(test_datagen.flow(X, batch_size=batch_size, shuffle=False), steps = X.shape[0] / batch_size)
    predictions.append(preds)
  
  return np.mean(predictions, axis=0)