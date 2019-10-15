import tensorflow as tf
import math

def augment_tf_image(image):
  # Brightness
  image = tf.image.random_brightness(image, 0.05)
  # Constrast
  image = tf.image.random_contrast(image, 0.9, 1.1)
  # Translation
  dp_x = tf.random_uniform([], minval=-10.0, maxval=10.0)
  dp_y = tf.random_uniform([], minval=-10.0, maxval=10.0)
  image = tf.contrib.image.translate(image, [dp_x, dp_y])
  # Rotation
  max_angle = 15.0 * math.pi / 180.0
  rot_angle = 2.0*tf.random_uniform([])*max_angle-max_angle
  image = tf.contrib.image.rotate(image, rot_angle)

  return image


def parse_function_train(example, image_size, augment=False):
  features = {
              'image_encoded': (tf.FixedLenFeature([], tf.string)),
             }
  parsed = tf.parse_single_example(example, features)
  decoded_img = tf.image.decode_jpeg(parsed['image_encoded'], channels=3)

  if augment:
    decoded_img = augment_tf_image(decoded_img)

  decoded_img = tf.image.resize_images(decoded_img, image_size)
  decoded_img = tf.cast(decoded_img, tf.float32) / 255.0
  return decoded_img


def parse_function_val(example, image_size, augment=False):
  return parse_function_train(example, image_size, augment)


def get_train_iterator(tfrecord_list, image_size, batch_size, augment=False):
  train_dataset = tf.data.TFRecordDataset(tfrecord_list)
  train_dataset = train_dataset.shuffle(buffer_size=batch_size*2000)
  train_dataset = train_dataset.apply(tf.contrib.data.map_and_batch(lambda x: parse_function_train(x, image_size, augment), batch_size=batch_size))
  train_dataset = train_dataset.prefetch(buffer_size=batch_size)
  train_iterator = train_dataset.make_initializable_iterator()
  train_next_batch = train_iterator.get_next()
  return train_iterator, train_next_batch


def get_val_iterator(tfrecord_list, image_size, batch_size, augment=False):
  test_dataset = tf.data.TFRecordDataset(tfrecord_list)
  test_dataset = test_dataset.apply(tf.contrib.data.map_and_batch(lambda x: parse_function_val(x, image_size, augment=False), batch_size=batch_size))
  test_dataset = test_dataset.prefetch(buffer_size=batch_size)
  test_iterator = test_dataset.make_initializable_iterator()
  test_next_batch = test_iterator.get_next()
  return test_iterator, test_next_batch


def get_count_iterator(tfrecord_list, image_size, batch_size):
  count_dataset = tf.data.TFRecordDataset(tfrecord_list)
  count_dataset = count_dataset.batch(batch_size)
  count_iterator = count_dataset.make_initializable_iterator()
  count_next_batch = count_iterator.get_next()
  return count_iterator, count_next_batch
