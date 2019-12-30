import tensorflow as tf
import os
import time

from config import config
from sisr_model import sisr_cnn
from utils import get_filepath, train_parse, write_logs

model_dir = config.TRAIN.model_dir
logs_path = config.TRAIN.logs_path
logs_train = config.TRAIN.logs_train
train_lr_dir = config.TRAIN.lr_img_path
train_hr_dir = config.TRAIN.hr_img_path

# Image Parameters
scale = config.IMG.scale

# Hyper Parameters
n_epoch = config.TRAIN.n_epoch
patch_size = config.TRAIN.patch_size
batch_size = config.TRAIN.batch_size
learning_rate_init = config.TRAIN.learning_rate_init
learning_rate_decay = config.TRAIN.learning_rate_decay
decay_period = config.TRAIN.decay_period

# Adam Parameters
beta1 = config.TRAIN.beta1
beta2 = config.TRAIN.beta2

def training():
  x = tf.placeholder(tf.float32, [None,patch_size,patch_size,3], name='lr_input')
  y = tf.placeholder(tf.float32, [None,scale*patch_size,scale*patch_size,3], name='hr_target')

  train_hr_path = get_filepath(train_hr_dir, '.png')
  n_trains = len(train_hr_path)

  train_hr_path = tf.constant(train_hr_path)

  dataset = tf.data.Dataset.from_tensor_slices(train_hr_path)
  dataset = dataset.shuffle(n_trains)
  dataset = dataset.map(train_parse, num_parallel_calls=16)
  dataset = dataset.batch(batch_size)

  iter = dataset.make_initializable_iterator()
  lr_patch, hr_patch = iter.get_next()
  
  with tf.name_scope('Model'):
    pred = sisr_cnn(x, scale, is_train=True, reuse=False)
  
  with tf.name_scope('L1_Loss'):
    loss = tf.losses.absolute_difference(pred, y)
  tf.summary.scalar("loss", loss)

  with tf.variable_scope('learning_rate'):
    lr_v = tf.Variable(learning_rate_init, trainable=False)
  
  optimizer = tf.train.AdamOptimizer(lr_v, beta1, beta2)
  train_op = optimizer.minimize(loss)

  saver = tf.train.Saver()

  merged_summary_op = tf.summary.merge_all()

  n_batches = int(n_trains/batch_size) + 1
  
  with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Op to write logs to Tensorboard
    train_sum_writer = tf.summary.FileWriter(logs_path, tf.get_default_graph())

    # Training process
    log = "\n========== Training Begin ==========\n"
    write_logs(logs_train, log, True)
    train_start = time.time()
    for epoch in range(n_epoch):
      epoch_start = time.time()

      if epoch != 0 and (epoch % decay_period == 0):
        new_lr_decay = lr_v * learning_rate_decay
        sess.run(tf.assign(lr_v, new_lr_decay))
        log = "** New learning rate: %.9f **\n" % (lr_v.eval())
        write_logs(logs_train, log, False)
      elif epoch == 0:
        sess.run(tf.assign(lr_v, learning_rate_init))
        log = "** Initial learning rate: %.9f **\n" % (learning_rate_init)
        write_logs(logs_train, log, False)

      avg_loss = 0
      sess.run(iter.initializer)
      for batch in range(n_batches):
        batch_start = time.time()
        lr_patches, hr_patches = sess.run([lr_patch, hr_patch])
        _, loss_val, summary = sess.run([train_op, loss, merged_summary_op], feed_dict={x:lr_patches, y:hr_patches})
        avg_loss += loss_val
        train_sum_writer.add_summary(summary, epoch*n_batches+batch)

        log = "Epoch {}, Time {:2.5f}, Batch {}, Batch Loss = {:.4f}".format(epoch, time.time()-batch_start, batch, loss_val)
        write_logs(logs_train, log, False)

      log = "\nEpoch {}, Time {:2.5f}, Average Loss = {:.4f}\n".format(epoch, time.time()-epoch_start, avg_loss/n_batches)
      write_logs(logs_train, log, False)

    log = "\nTraining Time: {}".format(time.time()-train_start)
    write_logs(logs_train, log, False)
    log = "\n========== Training End ==========\n"
    write_logs(logs_train, log, False)

    # Save model
    save_path = saver.save(sess, model_dir)
    log = "Model is saved in file: %s" % save_path
    write_logs(logs_train, log, False)

    log = "Run the command line:\n" \
          "--> tensorboard --logdir=../logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser"
    write_logs(logs_train, log, False)
    sess.close()

  
