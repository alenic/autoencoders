import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import tfrecord_parser
import os
import cv2

model_name = 'cnn.ckpt-968'
test_tfrecord = 'tfrecords/lfw_val.tfrecords'
width = 64
height = 64
channels = 3

with tf.Session() as sess:
  saver = tf.train.import_meta_graph('checkpoints/{}.meta'.format(model_name))
  saver.restore(sess, 'checkpoints/{}'.format(model_name))
  graph = tf.get_default_graph()

  input_image = graph.get_tensor_by_name('input_image:0')
  training_phase = graph.get_tensor_by_name('training_phase:0')
  #encoded = graph.get_tensor_by_name('code:0')
  decoded = graph.get_tensor_by_name('decoded:0')
  
  test_images_folder = 'test_images'
  file_list = os.listdir(test_images_folder)

  X_images_test = []
  for f in file_list:
    image_path = os.path.join(test_images_folder, f)
    img = cv2.imread(image_path)
    if channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X_images_test.append(cv2.resize(img, (width,height))/255.0)

  X_images_test = np.array(X_images_test)

  feed_val = {input_image: X_images_test, training_phase: False}
  decoded_np = sess.run(decoded, feed_dict=feed_val)
  print(decoded_np.shape)

  n_images_test = len(X_images_test)
  for i in range(n_images_test):
    img1 = (X_images_test[i]*255).astype(np.uint8)
    img_gen = (decoded_np[i]*255).clip(0,255).astype(np.uint8)
    cv2.imshow('Error %.3f' % (np.linalg.norm(img1-img_gen)), cv2.hconcat([img1, img_gen]))
    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyAllWindows()