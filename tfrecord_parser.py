import tensorflow as tf
import numpy as np
import cv2

def augment_tf_image(image, image_size):
  # Brightness
  image = tf.image.random_brightness(image, 0.15)

  # Constrast
  image = tf.image.random_contrast(image, 0.7, 1.4)
  
  # Crop and resize
  image = tf.random_crop(image, [tf.shape(image)[0], int(image_size[1]*0.85), int(image_size[0]*0.85), 3])

  # Flip
  image = tf.image.random_flip_left_right(image)

  # Rotation
  max_angle = 15.0 * np.pi / 180.0
  rot_angle = 2.0*tf.random_uniform([])*max_angle-max_angle
  image = tf.contrib.image.rotate(image, rot_angle)
  image = tf.image.resize_images(image, image_size)
  return image


def parse_function_train(example, image_size, augment=False):
  features = {
              'image_encoded': (tf.FixedLenFeature([], tf.string)),
             }
  parsed = tf.parse_single_example(example, features)
  decoded_img = tf.image.decode_jpeg(parsed['image_encoded'], channels=3)

  if augment:
    decoded_img = augment_tf_image(decoded_img, image_size)
  else:
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

if __name__ == '__main__':

  img_width = 128
  img_height = 128
  image = tf.placeholder(tf.uint8, [None, img_height, img_width, 3])
  distort_node = augment_tf_image(image, [img_width, img_height])

  frame = cv2.imread('test.jpg')
  frame = cv2.resize(frame, (img_width,img_height))

  with tf.Session() as sess:
    while True:
      batch = [frame, frame]
      image_out = sess.run(distort_node, feed_dict={image: batch})

      distorted_frame = image_out[0].astype(np.uint8)

      cv2.imshow('test', distorted_frame)
      if cv2.waitKey(0) == ord('q'):
        break


