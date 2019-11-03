# autoencoders
A Simple implementation of autoencoders in tensorflow. The implemented models, are of the following types:

### MLP (MultiLayer Perceptron Autoencoder)

It is the simplest form of autoencoder:

![MLP Autoencoder](/docs/mlp.jpg "Autoencoder of type MLP")


### CNN (Convolutional Neural Network Autoencoder)

The encoder is a sequence of convolutional + maxPooling layers and the decoder a sequence of deconvolutional layers

![CNN Autoencoder](/docs/cnn.jpg "Autoencoder of type CNN")


## Usage

* First step: download an image dataset, like:

[Labeled Face Database Dataset](http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz)

<br/>

* Second step: create a tfrecord dataset (e.g. 90% training and 10% validation) typing the command:

```
  python3 create_dataset.py --input_path lfw-deepfunneled --output_name lfw --train_perc 0.9
```

<br/>

The above script simply adds all images (of type JPEG) inside the root folder, to the tfrecord files {output_name}_train.tfrecords and {output_name}_val.tfrecords

<br/>

* Third step: train the autoencoder typing the command

```
  python3 train.py -t lfw_train.tfrecords -v lfw_val.tfrecords -rw 64 -rh 64 --model cnn
```
