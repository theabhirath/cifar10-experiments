# Pytorch Experiments

This repository contains my code for various models that I built in PyTorch. All models have been trained using Colab GPUs unless explicitly stated otherwise.

> **NOTE**:
    GMACs refer to (giga) **multiply-accumulate operations**, and are an indicator of the amount of compute a certain model requires.

# Vision

## Image Classification on CIFAR-10 

I have trained models using a 45k-5k training-val split from the original CIFAR-10 training data.The data augmentation procedure followed was standard - random horizontal flips, random crops of size 32 with a padding of 4 and normalizing pixels with per channel mean values. The code for the models and a model notebook designed for training them is attached -- the file paths can be modified appropriately as per local requirements to run the notebook.

### VGG-like architecture

I built a deep ConvNet inspired by the [VGG architecture](https://arxiv.org/abs/1409.1556.pdf). This network has 1,110,826 parameters spread over 8 layers, and requires 11.28 GMACs. The first 6 layers are convolutional layers interspersed with max pooling layers, followed by two fully connected layers. The convolutional layers all have size 3 x 3, while the max pooling layers have size 2 x 2. 2D-Dropout has been used after every convolutional layer i.e. entire channel values are zeroed, and regular dropout was used after the first fully connnected layer. 

The model manages to achieve a modest training accuracy of 87.59, and faces difficulty in converging. While the validation and test accuracy were higher (90.8% and 90.16% respectively), this cannot be attributed to the model fitting the data well. 

The optimizer used was AdamW with learning rate of 3e-3 and weight decay of 0.1. The learning rate was reduced by a factor of 10 twice, once after 50 epochs and once after 75 epochs.

### ResNet-like architecture

I built a deep ConvNet inspired by the [ResNet architecture](https://arxiv.org/abs/1512.03385.pdf). The architecture is hybrid, with the first few layers based on a vanilla residual block structure and subsequent ones based on bottleneck residual blocks as described in the original paper. A 5 x 5 convolutional block is used to feed the image into the main network, and global average pooling is used at the end instead of a fully connected network and fed into a single fully connected layer followed by softmax.

I evaluated two networks, a lighter one with three vanilla residual blocks and six bottleneck residual blocks, and achieved a good validation accuracy of 91.86% and a test accuracy of **91.20%**, despite having just 310,474 parameters and requiring 8.95 GMACs. A heavier network with six vanilla residual blocks and nine bottleneck residual blocks (3,921,994 parameters, 16.73 GMACs) further improved the accuracy to 92.54% on the validation set and **91.79%** on the test set. 

Both the models were trained for 100 epochs with a batch size of 128. The optimizer used was AdamW with learning rate of 3e-3 and weight decay of 0.1. The learning rate was reduced by a factor of 10 twice, once after 50 epochs and once after 75 epochs. 

### ResNeXt-like architecture

I built a deep ConvNet inspired by the [ResNeXt architecture](https://arxiv.org/pdf/1611.05431.pdf). The cardinality of the network was 64, and the bottleneck width was 4 i.e the network can be described as being 64 x 4d. A 5 x 5 convolutional block is used to feed the image into the main network similar to the ResNet. The subsequent architecture consists of 3 ResNeXt chunks with 2 blocks each. Each block implements aggregated transforms by using grouped convolutions with a bottleneck to reduce compute. Downsampling is done at the end of every chunk. Further, residual connections are incoporated in each chunk as well to strengthen the network. The output from the stack of ResNeXt blocks is fed into a single fully connected layer post global average pooling, followed by softmax.

I achieved a good validation accuracy of 92.96% and a test accuracy of **92.30%**, with a smaller network of 3 constituent blocks (1,087,114 parameters, 19.19 GMACs). A larger model with one extra block (3,090,058 parameters, 21.23 GMACs) doesn't increase the accuracy by much (93.14 and 92.00%). 

Both the models were trained for 100 epochs with a batch size of 128. The optimizer used was AdamW with learning rate of 3e-3 and weight decay of 0.1. The learning rate was reduced by a factor of 10 twice, once after 50 epochs and once after 75 epochs.

### DenseNet-like architecture

I built a deep ConvNet inspired by the [DenseNet architecture](https://arxiv.org/pdf/1608.06993.pdf). The model is inspired by the DenseNet-BC architecture that is described in the paper, and thus implements both bottlenecks in each dense block to reduce compute as well as compression in the transition blocks so as to make the model even more compact. The overall design consists of 3 dense chunks consisting of one dense block and one transition block each. A final dense block is used before global average pooling, a fully-connected layer and a softmax as with the other ResNet variants.

Due to feature map reuse, this network takes much fewer parameters to return competitive results, although it requires a large amount of compute. For example, even with just a 4-chunk network with a growth rate of 12 (508,447 parameters, 16.30 GMACs), I achieve a validation accuracy of 93.62% and a test accuracy of **92.8%**. With a slightly larger network (5 chunks, growth rate of 16 with computation worth 1,098,724 params and 28.75 GMACs), the accuracy is boosted to 94.04% on the validation set and **93.09%** on the test set. 

Both the models use a compression factor of 0.5 in the transition blocks and were trained for 100 epochs with a batch size of 128. The optimizer used was AdamW with learning rate of 3e-3 and weight decay of 0.1. The learning rate was reduced by a factor of 10 twice, once after 50 epochs and once after 75 epochs.

# NLP

## Character-level RNN

I built a character-level RNN based on the [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html) to classify last names by nationality.

