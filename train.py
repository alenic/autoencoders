'''
filename: train.py
author:   Alessandro Nicolosi
website:  https://github.com/alenic/autoencoders
license:  MIT
'''

import tensorflow as tf 
import argparse 
import autoenc_models
import tfrecord_parser
import os

if __name__ == '__main__':

  parser = argparse.ArgumentParser(usage='python3 train.py -t tfrecords/lfw_train.tfrecords -v tfrecords/lfw_val.tfrecords -rw 64 -rh 64 --model cnn')
  parser.add_argument('-t', '--tfrecord_train', type=str, required=True, help='Tfrecord for training file paths (e.g. -t file1.tfrecord,file2.tfrecord) ')
  parser.add_argument('-v', '--tfrecord_val', type=str, default='', help='Tfrecord for validation file paths (e.g. -t file1.tfrecord,file2.tfrecord) ')
  parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size')
  parser.add_argument('-e', '--epochs', type=int, default=40, help='Number of epochs')
  parser.add_argument('-rw', '--res_width', type=int, default=28, help='Input image width')
  parser.add_argument('-rh', '--res_height', type=int, default=28, help='Input image height')
  parser.add_argument('-m', '--model', type=str, default='cnn', help='Model type: cnn | mlp')
  parser.add_argument('--augment', action='store_true', help='Augment images during training')
  
  args = parser.parse_args()
  image_size = (args.res_width, args.res_height)
  image_channels = 3

  # ========================= Parse Data ========================================
  train_iterator, train_next_batch = tfrecord_parser.get_train_iterator(args.tfrecord_train.split(','), image_size, args.batch_size, augment=args.augment)
  val_iterator, val_next_batch = tfrecord_parser.get_val_iterator(args.tfrecord_val.split(','), image_size, args.batch_size, augment=args.augment)
  count_iterator, count_next_batch = tfrecord_parser.get_count_iterator(args.tfrecord_train.split(','), image_size, args.batch_size)

  # ============================== Count elements of tfrecord =============================
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(count_iterator.initializer)
    n_sample = 0
    while True:
      try:
        X = sess.run(count_next_batch)
        n_sample += X.shape[0]
      except tf.errors.OutOfRangeError:
        break
  
  print('Tfrecords:', args.tfrecord_train, 'has', n_sample, 'samples')
  
  train_batches_per_epoch = (n_sample//args.batch_size) + 1
  total_iterations = train_batches_per_epoch*args.epochs
  summary_iteration = total_iterations//100    # summary 100 times per batch

  # ============================== Input nodes ==============================================
  input_image = tf.placeholder(tf.float32, [None, args.res_height, args.res_width, image_channels], name='input_image')
  training_phase = tf.placeholder(tf.bool, name='training_phase')

  # ============================== Model ===================================================
  if args.model == 'cnn':
    model = autoenc_models.CNNAutoencoder(input_image, [32, 64, 128], 32, training_phase, dropout_rate=0.4).build()
  elif args.model == 'mlp':
    model = autoenc_models.MLPAutoencoder(input_image, [512, 256, 128], 32, training_phase).build()
  # ============================== Evaluation nodes ========================================
  cost = tf.reduce_mean(tf.square(input_image - model['decoded']))
  # ============================== Optimization nodes ======================================
  global_step = tf.Variable(0, trainable=False)
  boundaries = [int(total_iterations*0.65), int(total_iterations*0.8)]
  learning_rate_values = [1e-3, 2e-5, 5e-6]
  learning_rate = tf.train.piecewise_constant(global_step, boundaries, learning_rate_values)

  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    optimize_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

  # ========================= Init ========================================
  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  saver = tf.train.Saver()

  with tf.Session() as sess:

    # Tensorboard configuration
    tb_writer = tf.summary.FileWriter('./tensorboard', sess.graph)
    tf.summary.scalar('Training Cost', cost)
    tf.summary.scalar('Learning rate', learning_rate)
    summary_merged = tf.summary.merge_all()

    sess.run(init_op)

    for epoch in range(args.epochs):
      train_cost = 0.0
      train_cost_count = 0
      val_cost = 0.0
      val_cost_count = 0

      sess.run(train_iterator.initializer)

      while True:
        iteration = sess.run(global_step)
        try:
          train_batch_X = sess.run(train_next_batch)
        except tf.errors.OutOfRangeError:
          break
        
        feed_train = {input_image: train_batch_X, training_phase: True}
        sess.run(optimize_step, feed_dict=feed_train)

        if iteration % summary_iteration == 0:
          feed_val = {input_image: train_batch_X, training_phase: False}
          cost_np, sum_merged = sess.run([cost, summary_merged], feed_dict=feed_val)
          train_cost += cost_np
          train_cost_count += 1
          tb_writer.add_summary(sum_merged, iteration)


      print('================= Epoch %d/%d =============' % (epoch+1, args.epochs))
      train_cost /= train_cost_count
      print('Training cost: ', train_cost)

      # Validation
      sess.run(val_iterator.initializer)
      while True:
        try:
          val_batch_X = sess.run(val_next_batch)
        except tf.errors.OutOfRangeError:
          break
        
        feed_val= {input_image: val_batch_X, training_phase: False}
        val_cost_np = sess.run(cost, feed_dict=feed_val)
        val_cost += val_cost_np
        val_cost_count += 1
      
      print('Validation cost: ', val_cost)
      chkpt_file = './checkpoints/{}.ckpt'.format(args.model)
      save_path = saver.save(sess, chkpt_file, global_step=iteration)
      print(chkpt_file, ' saved.')