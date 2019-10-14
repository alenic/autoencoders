'''
filename: create_tfrecords.py
author:   Alessandro Nicolosi
website:  https://github.com/alenic
license:  MIT
'''

import tensorflow as tf
import argparse
import random
import glob
import os


def create_tf_record(image_path_list, path):
  with tf.python_io.TFRecordWriter(path) as writer:
    print("Writing " + path + "...")

    n = len(image_path_list)
    n10 = n//10
    count = 0
    # Loop through all image's paths
    for i in range(n):
      file_extension = image_path_list[i].split('.')[-1]
      if file_extension == 'jpeg' or file_extension == 'jpg':
        with tf.gfile.FastGFile(image_path_list[i], 'rb') as fid:
          jpeg_data = fid.read()
        
        if len(jpeg_data) > 0:
          # Create the tensorflow example
          features = tf.train.Features(
            feature = {
              'image_encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[jpeg_data]))
            }
          )
    
          example = tf.train.Example(features=features)
          serialized = example.SerializeToString()
          writer.write(serialized)

          # Print progress
          count += 1
          if count % n10 == 0:
            print('Progress %d/100' % (int(100.0*count/n)))
      else:
        print('WARNING: The file ', image_path_list[i], ' is not jpeg! Skip it')

    
    print('Tfrecord created!')


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input_path', type=str, required=True, help='Root folder path of dataset')
  parser.add_argument('-o', '--output_name', type=str, required=True, help='Dataset name')
  parser.add_argument('-p', '--train_perc', type=float, default=0.85, help='Training example percentage')
  args = parser.parse_args()

  # Get image path list
  image_path_list = glob.glob(os.path.join(args.input_path, '**/*.jpg'))
  image_path_list.extend(glob.glob(os.path.join(args.input_path, '**/*.jpeg')))

  # Shuffle list
  random.shuffle(image_path_list)
  
  # get number of total data and the number of training examples
  n_data = len(image_path_list)
  n_train = int(n_data*args.train_perc)

  # create train tfrecord
  train_filename = '%s_train.tfrecords' % (args.output_name)
  create_tf_record(image_path_list[:n_train], train_filename)

  # create validation tfrecord
  if args.train_perc < 1.0:
    val_filename = '%s_val.tfrecords' % (args.output_name)
    create_tf_record(image_path_list[n_train:], val_filename)

