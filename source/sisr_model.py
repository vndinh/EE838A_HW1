import tensorflow as tf

def sisr_cnn(x, scale, is_train=True, reuse=False):
  x = tf.layers.conv2d(x, 64, 7, strides=(1,1), padding="same", name="k7n64s1")
  x = tf.nn.relu(x, name="relu_1")
  tmp = x
  for i in range(4):
    xx = tf.layers.conv2d(x, 64, 3, strides=(1,1), padding="same", name="res_%s/k3n64s1_in" % i)
    xx = tf.nn.relu(xx, name="res_%s/relu" % i)
    xx = tf.layers.conv2d(xx, 64, 3, strides=(1,1), padding="same", name="res_%s/k3n64s1_out" % i)
    xx = tf.add(x, xx)
    x = xx
  
  x = tf.layers.conv2d(x, 64, 3, strides=(1,1), padding="same", name="k3n64s1")
  x = tf.add(x, tmp)
  
  x = tf.layers.conv2d(x, 256, 3, strides=(1,1), padding="same", name="k3n256s1")
  x = tf.depth_to_space(x, scale) # Pixel shuffle layer (x2)
  
  x = tf.nn.relu(x, name="relu_2")
  x = tf.layers.conv2d(x, 3, 7, strides=(1,1), padding="same", name="k7n3s1")
  
  return x



