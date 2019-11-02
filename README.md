# autoencoders
A Simple implementation of autoencoders in tensorflow. The implemented models, are of the following types:

### MLP (MultiLayer Perceptron Autoencoder)

It is the simplest form of autoencoder:

![MLP Autoencoder](/docs/mlp.jpg "Autoencoder of type MLP")


### CNN (Convolutional Neural Network Autoencoder)

The encoder is a sequence of convolutional + maxPooling layers and the decoder a sequence of deconvolutional layers

![CNN Autoencoder](/docs/cnn.jpg "Autoencoder of type CNN")


## Usage

* First of all, download a dataset, for example

[Labeled Face Database Dataset](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz)


* After it was extracted, you must create a tfrecord dataset (e.g. 90% training and 10% validation) typing the command:

<code>
  python3 create_dataset.py --input_path lfw-deepfunneled --output_name lfw --train_perc 0.9
</code>

The above script simply adds all images (of type JPEG) inside the root folder, to the tfrecord files {output_name}_train.tfrecords and {output_name}_val.tfrecords



* Finally, train the autoencoder typing the command

<code>
  python3 train.py -t lfw_train.tfrecords -v lfw_val.tfrecords -rw 64 -rh 64 --model cnn
</code>
