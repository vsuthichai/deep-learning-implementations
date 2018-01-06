# TensorFlow implementation of DenseNet

This is my TensorFlow reimplementation of DenseNet, a neural network architecture that was published in [Densely Connected Convolutional Networks](http://arxiv.org/abs/1608.06993) by Gao Huang and Zhuang Liu.  DenseNet architecture is composed of dense blocks where every layer within a block is connected to every other layer within the same block.  The input of every layer is a concatenation of all previous layer inputs before it.  A layer is typically composed of the following ops BatchNorm->Relu->Conv2d->dropout.  Transition layers separate the dense blocks and also perform a pooling operation to reduce the size of the feature maps after every dense block operation.  Finally, a fully connected layer output layer performs a standard softmax loss with L2 regularization.

The original authors' Torch implementation can be found here : [https://github.com/liuzhuang13/DenseNet](https://github.com/liuzhuang13/DenseNet).

<img src="https://cloud.githubusercontent.com/assets/8370623/17981494/f838717a-6ad1-11e6-9391-f0906c80bc1d.jpg" width="480">

Figure 1: A dense block with 5 layers and growth rate 4. 

![densenet](https://cloud.githubusercontent.com/assets/8370623/17981496/fa648b32-6ad1-11e6-9625-02fdd72fdcd3.jpg)
Figure 2: A deep DenseNet with three dense blocks. 

## Prerequisites

Python 3.5

* TensorFlow 1.4+
* six
* numpy

## Usage

Assuming your environment is setup properly, make sure to point the script to your `cifar-10-batches-py` directory and simply execute the following:

```
python train_cifar10.py
```

A summary of the loss and accuracy is written to disk every 50 training steps, and a model is saved every epoch.

## Implementation Details

Strong effort was made to keep the training details as close to the paper as possible.  This was trained on a single Nvidia GTX 1080 gpu and took 13+ hours.

* MomentumOptimizer with learning rate 1e-1 and momentum 0.9.  Divide learning rate by 10 at epoch 150 and 225.
* Batch size 64
* Dropout probability of 0.2, ie. keep_prob = 0.8
* Growth rate k = 12
* Weight decay for L2 regularization 1e-4
* He initialization of convolutional filters and no convolutional bias

## Results

The figure below is a screenshot of the summary plots on TensorBoard after training and testing have completed on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).  Accuracy on the test set reaches nearly 93%.

![tensorboard summary](densenet_tensorboard_cifar10.png)

## Citation

Citation of original publication found below.  However, The TensorFlow re-implementation is my own work and can be used freely for learning purposes.  

```
@article{Huang2016Densely,
  author = {Huang, Gao and Liu, Zhuang and Weinberger, Kilian Q.},
  title = {Densely Connected Convolutional Networks},
  journal = {arXiv preprint arXiv:1608.06993},
  year = {2016}
}
```

