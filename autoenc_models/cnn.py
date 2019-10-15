import tensorflow as tf


class CNNAutoencoder():
  def __init__(self, image, conv_depths, encoding_size, training_phase, dropout_rate=0.4):
    self.image = image
    self.conv_depths = conv_depths.copy()
    self.encoding_size = encoding_size
    self.training_phase = training_phase
    self.dropout_rate = dropout_rate
  
  def build(self):
    nodes = {}
    encoded = self.image
    i = 1
    with tf.variable_scope('simple_cnn'):
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
        drop1 = tf.layers.dropout(flatten, self.dropout_rate, training=self.training_phase)
        code = tf.layers.dense(drop1, self.encoding_size)
        nodes['code'] = tf.identity(code, name='code')
        # Decoder
        drop2 = tf.layers.dropout(code, self.dropout_rate, training=self.training_phase)
        fc = tf.layers.dense(drop2, flatten_size[1])
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
    return nodes
