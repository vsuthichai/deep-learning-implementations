# TensorFlow implementation of DenseNet

This is my implementation of DenseNet, a neural network architecture that was published in [Densely Connected Convolutional Networks](http://arxiv.org/abs/1608.06993) by Gao Huang and Zhuang Liu.  DenseNet architecture is composed of dense blocks where every layer within a block is connected to every other layer within the same block.  The input of every layer is a concatenation of all previous layer inputs before it.  A layer is typically composed of the following ops BatchNorm->Relu->Conv2d->dropout.  Transition layers separate the dense blocks and also perform a pooling operation to reduce the size of the feature maps after every dense block operation.  Finally, a fully connected layer output layer performs a standard softmax loss with L2 regularization.

## Prerequisites

Python 3.5

* TensorFlow 1.4+
* six
* numpy

## Results

![tensorboard summary](densenet_tensorboard_cifar10.png)

