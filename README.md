# Pytorch Experiments

This repository contains my code for various models that I built in PyTorch. All models have been trained using Colab GPUs unless explicitly stated otherwise.

# Vision

## VGG-like architecture

I built a deep ConvNet inspired by the [VGG architecture](https://arxiv.org/abs/1409.1556) for the purpose of image classification. This has 16 hidden layers. The first 13 layers are convolutional layers interspersed with max pooling layers, followed by fully connected layers. The convolutional layers all have size 3 x 3, while the max pooling layers have size 2 x 2. Dropout with rate 0.5 and batch normalization have also been used liberally while building this network.

I trained the model for 50 epochs with a batch size of 256. The optimizer used was Adam with learning rate of 0.001 and weight decay of 0.0001. The learning rate was reduced by a factor of 10 twice, once after 35 epochs and once after 45 epochs. I achieved a decent train accuracy of 99.64% and test accuracy of 87.66% on CIFAR-10.

## ResNet-like architecture

I built a deep ConvNet inspired by the [ResNet architecture](https://arxiv.org/abs/1512.03385) for the purpose of image classification. The architecture is hybrid, with the first three layers based on a vanilla residual block structure and the subsequent based on bottleneck residual blocks as described in the original paper. A 5 x 5 convolutional block is used to feed the image into the main network, and global average pooling is used at the end instead of a fully connected network and fed into a softmax. Dropout of probability 0.05 is used intermittently. 

Data augmentation is used as in the original paper - random crops of size 32 with a padding of 4, normalizing pixels with per channel mean values and random horizontal flips.

The network was trained for 200 epochs with Adam, with a batch size of 128 and a learning rate of 0.003. Weight decay of 0.0001 was implemented. The learning rate was also reduced by a factor of 10 thrice, at 100, 150 and 175 epochs respectively. I achieved a good train accuracy of 96.824% and a train accuracy of 88.9%, despite having just 356,234 parameters as opposed to the VGG model's 9,985,034.

# Vision

## Character-level RNN

I built a character-level RNN based on the [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) to classify last names by nationality. 