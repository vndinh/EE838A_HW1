import tensorflow as tf
import os
import numpy as np

from PIL import Image
from config import config
from imresize import imresize

# Parameters
scale = config.IMG.scale
patch_size = config.TRAIN.patch_size

def write_logs(filename, log, start=False):
  print(log)
  if start == True:
    f = open(filename, 'w')
    f.write(log + '\n')
  else:
    f = open(filename, 'a')
    f.write(log + '\n')
    f.close()

def get_filepath(path, suffix):
  file_path = []
  for f in os.listdir(path):
    if f.endswith(suffix):
      file_path.append(os.path.join(path, f))
  file_path = sorted(file_path)
  return file_path

def resize_img(img):
  return imresize(img, output_shape=(patch_size,patch_size))

def train_parse(hr_dir):
  hr_string = tf.read_file(hr_dir)
  hr_decoded = tf.image.decode_png(hr_string, channels=3)
  hr_patch = tf.random_crop(hr_decoded, [scale*patch_size, scale*patch_size, 3])
  
  lr_patch = tf.py_func(resize_img, [hr_patch], tf.uint8)
  lr_patch = tf.reshape(lr_patch, [patch_size,patch_size,3])

  lr_patch = tf.image.convert_image_dtype(lr_patch, tf.float32)
  hr_patch = tf.image.convert_image_dtype(hr_patch, tf.float32)

  return lr_patch, hr_patch

def valid_parse(lr, hr):
  lr_string = tf.read_file(lr)
  lr_decoded = tf.image.decode_png(lr_string, channels=3)
  lr_img = tf.image.convert_image_dtype(lr_decoded, tf.float32)

  hr_string = tf.read_file(hr)
  hr_decoded = tf.image.decode_png(hr_string, channels=3)
  hr_img = tf.image.convert_image_dtype(hr_decoded, tf.float32)
  
  return lr_img, hr_img

def save_image(hr, hr_dir, id):
  _, h, w, c = hr.shape
  hr = np.reshape(hr, [h,w,c])
  hr = hr * 255.0
  np.clip(hr, 0, 255, out=hr)
  hr = hr.astype('uint8')
  hr_save = Image.fromarray(hr)
  hr_save.save(os.path.join(hr_dir, '{:04d}.png'.format(id)))