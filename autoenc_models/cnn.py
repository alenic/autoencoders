import tensorflow as tf


class CNNAutoencoder():
  def __init__(self, image, conv_depths, encoding_size, training_phase, variational=False):
    self.image = image
    self.conv_depths = conv_depths.copy()
    self.encoding_size = encoding_size
    self.training_phase = training_phase
    self.variational = variational
  
  def build(self):
    nodes = {}
    encoded = self.image
    i = 1
    with tf.variable_scope('Autoencoder'):
      # Encoder
      with tf.variable_scope('encoder'):
        for conv_depth in self.conv_depths:
          encoded = tf.layers.conv2d(encoded, conv_depth, [3,3], padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
          encoded = tf.layers.batch_normalization(encoded, training=self.training_phase)
          encoded = tf.nn.relu(encoded)
          encoded = tf.layers.max_pooling2d(encoded, pool_size=(2,2), strides=(2,2))
          nodes['conv'+str(i)] = encoded
          i += 1

        last_encoded_size = encoded.get_shape()
        flatten = tf.layers.flatten(encoded)
        flatten_size = flatten.get_shape()

        # Code
      with tf.variable_scope('code'):
        if self.variational:
          code_mean = tf.layers.dense(flatten, self.encoding_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
          code_sigma = tf.layers.dense(flatten, self.encoding_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
          noise = tf.random_normal(tf.shape(code_sigma), dtype=tf.float32)
          code = code_mean + code_sigma*noise
        else:
          code = tf.layers.dense(flatten, self.encoding_size)
        
      nodes['code'] = tf.identity(code, name='code')

      if self.variational:
        nodes['code_mean'] = tf.identity(code_mean, name='code_mean')
        nodes['code_sigma'] = tf.identity(code_sigma, name='code_sigma')

      # Decoder
      with tf.variable_scope('decoder'):
        fc = tf.layers.dense(nodes['code'], flatten_size[1])
        reshaped = tf.reshape(fc, [-1]+ last_encoded_size.as_list()[1:])

        rev = self.conv_depths
        rev.reverse()
        decoded = reshaped
        i = 1
        for conv_depth in rev[1:]:
          decoded = tf.layers.conv2d_transpose(decoded, conv_depth, kernel_size=[3,3], strides=[2,2], padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
          decoded = tf.layers.batch_normalization(decoded, training=self.training_phase)
          decoded = tf.nn.relu(decoded)
          nodes['deconv'+str(i)] = decoded
          i += 1
        
        # Last layer
        decoded = tf.layers.conv2d_transpose(decoded, 3, kernel_size=[3,3], strides=[2,2], padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer())
        decoded = tf.nn.relu(decoded)

    nodes['decoded'] = tf.identity(decoded, name='decoded')
    nodes['code'] = tf.identity(code, name='code')

    if self.variational:
      nodes['code_mean'] = tf.identity(code_mean, name='code_mean')
      nodes['code_sigma'] = tf.identity(code_sigma, name='code_sigma')
    
    return nodes
