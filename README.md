# Pytorch Experiments

This repository contains my code for various models that I built in PyTorch. All models have been trained using Colab GPUs unless explicitly stated otherwise. 

# Vision

## CIFAR-10

The data augmentation procedure followed was standard - random crops of size 32 with a padding of 4, normalizing pixels with per channel mean values and random horizontal flips. The models have been trained for 100 epochs with a batch size of 128. The optimizer used was Adam with learning rate of 3e-3 and weight decay of 0.1. The learning rate was reduced by a factor of 10 twice, once after 50 epochs and once after 75 epochs. 

### VGG-like architecture

I built a deep ConvNet inspired by the [VGG architecture](https://arxiv.org/abs/1409.1556) for the purpose of image classification. This network has 2,178,314 parameters spread over 15 layers. The first 13 layers are convolutional layers interspersed with max pooling layers, followed by two fully connected layers. The convolutional layers all have size 3 x 3, while the max pooling layers have size 2 x 2. Dropout and batch normalization have been used in alternate layers while building this network. I achieved a decent validation accuracy of 87.3% and test accuracy of **87.19%**.

### ResNet-like architecture

I built a deep ConvNet inspired by the [ResNet architecture](https://arxiv.org/abs/1512.03385) for the purpose of image classification. The architecture is hybrid, with the first three layers based on a vanilla residual block structure and the subsequent based on bottleneck residual blocks as described in the original paper. A 5 x 5 convolutional block is used to feed the image into the main network, and global average pooling is used at the end instead of a fully connected network and fed into a softmax.

I achieved a good validation accuracy of 91.04% and a test accuracy of **90.51%**, despite having just 266,122 parameters.

# NLP

## Character-level RNN

I built a character-level RNN based on the [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) to classify last names by nationality. 