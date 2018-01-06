# TensorFlow implementation of DenseNet

This is my reimplementation of DenseNet, a neural network architecture that was published in [Densely Connected Convolutional Networks](http://arxiv.org/abs/1608.06993) by Gao Huang and Zhuang Liu.  DenseNet architecture is composed of dense blocks where every layer within a block is connected to every other layer within the same block.  The input of every layer is a concatenation of all previous layer inputs before it.  A layer is typically composed of the following ops BatchNorm->Relu->Conv2d->dropout.  Transition layers separate the dense blocks and also perform a pooling operation to reduce the size of the feature maps after every dense block operation.  Finally, a fully connected layer output layer performs a standard softmax loss with L2 regularization.

## Prerequisites

Python 3.5

* TensorFlow 1.4+
* six
* numpy

## Implementation Details

Strong effort was made to keep the training details as close to the paper as possible.

* MomentumOptimizer with learning rate 1e-1.  Divide learning rate by 10 at epoch 150 and 225.
* Batch size 64
* Dropout probability of 0.2, ie. keep_prob = 0.8
* Growth rate k = 12
* Weight decay for L2 regularization 1e-4
* He initialization of convolutional filters

## Results

The figure below is a screenshot of the summary plots on TensorBoard after training and testing have completed on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).  Accuracy on the test set reaches nearly 93%.

![tensorboard summary](densenet_tensorboard_cifar10.png)

