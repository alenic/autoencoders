import tensorflow as tf 


class MLPAutoencoder():
  def __init__(self, image, layer_units, encoding_size, training_phase, variational=False):
    self.image = image
    self.layer_units = layer_units.copy()
    self.encoding_size = encoding_size
    self.training_phase = training_phase
    self.variational = variational
  
  def build(self):
    nodes = {}
    
    with tf.variable_scope('Autoencoder'):
      input_shape = self.image.get_shape()
      # Encoder
      encoded = tf.layers.flatten(self.image)
      flatten_shape = encoded.get_shape()
      i = 1
      with tf.variable_scope('encoder'):
        for layer in self.layer_units:
          encoded = tf.layers.dense(encoded, layer, activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer())
          nodes['enc_dense'+str(i)] = encoded
          i += 1
      # Code
      with tf.variable_scope('code'):
        if self.variational:
          code_mean = tf.layers.dense(encoded, self.encoding_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
          code_sigma = tf.layers.dense(encoded, self.encoding_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
          noise = tf.random_normal(tf.shape(code_sigma), dtype=tf.float32)
          code = code_mean + code_sigma*noise
        else:
          code = tf.layers.dense(encoded, self.encoding_size)

      nodes['code'] = tf.identity(code, name='code')

      if self.variational:
        nodes['code_mean'] = tf.identity(code_mean, name='code_mean')
        nodes['code_sigma'] = tf.identity(code_sigma, name='code_sigma')

      # Decoder
      with tf.variable_scope('decoder'):
        rev = self.layer_units
        rev.reverse()
        decoded = nodes['code']
        i = 1
        for layer in rev[1:]:
          decoded = tf.layers.dense(decoded, layer, kernel_initializer=tf.contrib.layers.xavier_initializer())
          nodes['denc_dense'+str(i)] = encoded
          i += 1
        # Last layer
        decoded = tf.layers.dense(decoded, flatten_shape.as_list()[1], activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        decoded = tf.reshape(decoded, [-1] + input_shape.as_list()[1:])
    
    nodes['decoded'] = tf.identity(decoded, name='decoded')
    
    return nodes
