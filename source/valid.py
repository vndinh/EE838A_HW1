import tensorflow as tf
import numpy as np
import time
import os
from PIL import Image

from utils import get_filepath, valid_parse, write_logs, save_image
from sisr_model import sisr_cnn
from config import config

model_dir = config.TRAIN.model_dir
scale = config.IMG.scale

def validate(lr_dir, hr_dir, sr_gen, logs_dir, is_valid=True):
  X = tf.placeholder(tf.float32, [1,None,None,3])
  Y = tf.placeholder(tf.float32, [1,None,None,3])

  valid_lr_path = get_filepath(lr_dir, '.png')
  valid_hr_path = get_filepath(hr_dir, '.png')
  n_valid = len(valid_lr_path)

  valid_lr_path = tf.constant(valid_lr_path)
  valid_hr_path = tf.constant(valid_hr_path)

  dataset = tf.data.Dataset.from_tensor_slices((valid_lr_path, valid_hr_path))
  dataset = dataset.map(valid_parse, num_parallel_calls=16)
  dataset = dataset.batch(1)
  iter = dataset.make_one_shot_iterator()
  lr_img, hr_img = iter.get_next()

  pred = sisr_cnn(X, scale, is_train=False, reuse=False)
  loss = tf.losses.absolute_difference(pred, Y)
  saver = tf.train.Saver(tf.global_variables())

  with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Restore model weights from previous saved model
    saver.restore(sess, model_dir)

    # Validation
    if is_valid:
      log = "\n========== Validation Begin ==========\n"
    else:
      log = "\n========== Test Begin ==========\n"
    write_logs(logs_dir, log, True)
    valid_start = time.time()
    avg_loss = 0
    for i in range(n_valid):
      valid_img_start = time.time()
      lr_image, hr_image = sess.run([lr_img, hr_img])
      hr_output, loss_val = sess.run([pred, loss], feed_dict={X:lr_image, Y:hr_image})
      avg_loss += loss_val
      save_image(hr_output, sr_gen, i+1)
      log = "Image {:04d}, Time {:2.5f}, Shape = {}, Loss = {:.4f}".format(i+1, time.time()-valid_img_start, np.shape(hr_output), loss_val)
      write_logs(logs_dir, log, False)
    
    log = "\nAverage Loss = {:.4f}".format(avg_loss/n_valid)
    write_logs(logs_dir, log, False)
    log = "\nValidation Time: {:2.5f}".format(time.time()-valid_start)
    write_logs(logs_dir, log, False)
    if is_valid:
      log = "\n========== Validation End ==========\n"
    else:
      log = "\n========== Test End ==========\n"
    write_logs(logs_dir, log, False)
    sess.close()

