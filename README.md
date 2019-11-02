# autoencoders
A Simple implementation of autoencoders in tensorflow. The following graph models are implemented

### MLP (MultiLayer Perceptron Autoencoder)

It is the simplest form of autoencoders:

![MLP Autoencoder](/docs/mlp.jpg "Autoencoder of type MLP")


### CNN (Convolutional Neural Network Autoencoder)

The encoder is a sequence of convolutional + maxPooling layers and the decoder a sequence of deconvolutional layers

![CNN Autoencoder](/docs/cnn.jpg "Autoencoder of type MLP")


## Usage

* First, download a dataset, for example

[Labeled Face Database Dataset](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz)


* After extract it, create a tfrecord dataset (90% training and 10% validation) with the command:

<code>
  python3 create_dataset.py --input_path lfw-deepfunneled --output_name lfw --train_perc 0.9
</code>

* Finally, train the autoencoder

<code>
  python3 train.py -t lfw_train.tfrecords -v lfw_val.tfrecords -rw 64 -rh 64 --model cnn
</code>
