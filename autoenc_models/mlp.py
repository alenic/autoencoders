import tensorflow as tf 


class MLPAutoencoder():
  def __init__(self, image, layers, encoding_size, training_phase):
    self.image = image
    self.layers = layers.copy()
    self.encoding_size = encoding_size
    self.training_phase = training_phase
  
  def build(self):
    nodes = {}

    with tf.variable_scope('nn'):
      input_shape = self.image.get_shape()
      # Encoder
      encoded = tf.layers.flatten(self.image)
      flatten_shape = encoded.get_shape()
      i = 1
      with tf.variable_scope('encoder'):
        for layer in self.layers:
          encoded = tf.layers.dense(encoded, layer, activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer())
          nodes['enc_dense'+str(i)] = encoded
          i += 1
      # Code
      code = tf.layers.dense(encoded, self.encoding_size)
      nodes['code'] = tf.identity(code, name='code')
      # Decoder
      rev = self.layers
      rev.reverse()
      decoded = code
      i = 1
      with tf.variable_scope('decoder'):
        for layer in rev[1:]:
          decoded = tf.layers.dense(decoded, layer, kernel_initializer=tf.contrib.layers.xavier_initializer())
          nodes['denc_dense'+str(i)] = encoded
          i += 1
        # Last layer
        decoded = tf.layers.dense(decoded, flatten_shape.as_list()[1], activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        decoded = tf.reshape(decoded, [-1] + input_shape.as_list()[1:])
    
    nodes['decoded'] = tf.identity(decoded, name='decoded')
    return nodes
