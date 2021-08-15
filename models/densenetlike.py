import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoConv2d_BN(nn.Module):
    """
    custom block that implements 'same' padding for a convolution followed by 
    batch normalisation and ReLU non-linearity.
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride = 1):
        super(AutoConv2d_BN, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 
                              kernel_size = kernel_size, 
                              padding = kernel_size // 2, stride = stride)
        self.bn = nn.BatchNorm2d(output_channels)
    
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x

class DenseBlock(nn.Module):
    """dense block of DenseNet."""
    def __init__(self, input_channels, kernel_size, growth_rate, 
                    num_layers, bottleneck_width):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        bottleneck_layers = bottleneck_width * growth_rate
        self.bottles = nn.ModuleList([AutoConv2d_BN(input_channels + i * growth_rate, 
                                                bottleneck_layers,
                                                kernel_size = 1)
                                                for i in range(num_layers)])
        self.convs = nn.ModuleList([AutoConv2d_BN(bottleneck_layers, 
                                                growth_rate,
                                                kernel_size = kernel_size)
                                                for _ in range(num_layers)])

    def forward(self, x):
        inputs = [x]
        for i in range(self.num_layers):
            x = self.bottles[i](x)
            x = self.convs[i](x)
            inputs.append(x)
            x = torch.cat(inputs, dim = 1)
        return x

class TransitionBlock(nn.Module):
    """transition block of DenseNet."""
    def __init__(self, input_channels, kernel_size = 1, pool_size = 2,
                    stride = 1, compression = 0.5):
        super(TransitionBlock, self).__init__()
        output_channels = int(input_channels * compression)
        self.conv = AutoConv2d_BN(input_channels, output_channels, 
                              kernel_size = kernel_size, stride = stride)
        self.pool = nn.AvgPool2d((pool_size, pool_size))

    def forward(self, x):
        x = self.pool(self.conv(x))
        return x

class densechunk(nn.Module):
    """one dense block and one transition block chained together."""
    def __init__(self, input_channels, growth_rate = 4, kernel_size = 3, pool_size = 2,
                    num_layers = 4, bottleneck_width = 4, compression = 0.5):
        super(densechunk, self).__init__()
        self.dense = DenseBlock(input_channels, kernel_size = kernel_size,
                                growth_rate = growth_rate, num_layers = num_layers, 
                                bottleneck_width = bottleneck_width)
        self.transition = TransitionBlock(input_channels + (growth_rate * num_layers), 
                                            kernel_size = 1,
                                            pool_size = pool_size,
                                            compression = compression)

    def forward(self, x):
        x = self.dense(x)
        x = self.transition(x)
        return x
    
class densenetlike(nn.Module):
    """
    neural network architecture based on DenseNet-BC.

    inputs:

    1. `image_size`: size of input image (assumed to be three-dimensional of type (C, H, W) 
    where C is channels, H is height, and W is width). Default value is (3, 32, 32).
    2. `in_channels`: number of channels the image is upsampled to before feeding into the main
    network. (assumed to be an integer equal to or larger than `image_size[0]`). Default value 
    is 32.
    3. `kernel_size`: size of convolutional kernels throughout the network. Default value is 3.
    4. `pool_size`: size of pooling windows across the image in each dense block. Default value 
    is 2.
    5. `num_chunks`: number of dense chunks to be used in the network. Default value is 4.
    6. `growth_rate`: number of new channels created by each layer of a single dense block. 
    Default value is 12.
    7. `num_layers`: number of layers in each dense block in the network. Default value is 8.
    8. `bottleneck_width`: width of bottleneck layers in each dense block (as a factor 
    multiplied by the growth rate). Default value is 4.
    9. `compression`: compression factor of transition blocks in the network. Default value is 0.5.
    10. `num_classes`: number of classes to predict. Default value is 10.
    """
    def __init__(self, image_size = (3, 32, 32), in_channels = None, kernel_size = 3,
                    pool_size = 2,  num_chunks = 4, growth_rate = 12, num_layers = 8, 
                    bottleneck_width = 4, compression = 0.50, num_classes = 10):
        super(densenetlike, self).__init__()
        in_channels = 4 * growth_rate if in_channels is None else in_channels
        self.out_channels = [int(in_channels * (compression ** i) + 
                                growth_rate * num_layers * sum(compression ** j 
                                for j in range(1, i + 1))) for i in range(num_chunks + 1)]
        self.conv = AutoConv2d_BN(image_size[0], in_channels, kernel_size = 5)
        self.chunks = nn.ModuleList([densechunk(input_channels = self.out_channels[i],
                                                growth_rate = growth_rate,
                                                kernel_size = kernel_size,
                                                pool_size = pool_size,
                                                num_layers = num_layers,
                                                bottleneck_width = bottleneck_width,
                                                compression = compression)
                                                for i in range(num_chunks)])
        self.fblock = DenseBlock(input_channels = self.out_channels[-1],
                                kernel_size = kernel_size,
                                growth_rate = growth_rate,
                                num_layers = num_layers,
                                bottleneck_width = bottleneck_width)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.out_channels[-1] + growth_rate * num_layers, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        for chunk in self.chunks:
            x = chunk(x)
        x = self.fblock(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

        
    