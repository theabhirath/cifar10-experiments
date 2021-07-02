# Pytorch Experiments

This repository contains my code for various models that I built in PyTorch. All models have been trained using Colab GPUs unless explicitly stated otherwise.

# CIFAR-10

## VGG-like architecture

I built a deep ConvNet inspired by the VGG architecture for the purpose of image classification. This has 16 hidden layers. The first 13 layers are convolutional layers interspersed with max pooling layers, followed by fully connected layers. The convolutional layers all have size 3 x 3, while the max pooling layers have size 2 x 2. Dropout with rate 0.5 and batch normalization have also been used liberally while building this network.

I trained the model for 50 epochs with a batch size of 256. The optimizer used was Adam with learning rate of 0.001 and weight decay of 0.0001. The learning rate was reduced by a factor of 10 twice, once after 35 epochs and once after 45 epochs. I achieved a decent train accuracy of 99.64% and test accuracy of 87.66%.