import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoConv2d_BN(nn.Module):
    """
    custom block that implements 'same' padding for a convolution followed by 
    batch normalisation.
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride = 1):
        super(AutoConv2d_BN, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 
                              kernel_size = kernel_size, 
                              padding = kernel_size // 2, stride = stride)
        self.bn = nn.BatchNorm2d(output_channels)
    
    def forward(self, x):
        x = self.bn(F.relu(self.conv(x)))
        return x

class DenseBlock(nn.Module):
    """returns one dense block of DenseNet."""
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
                                                (i + 1) * growth_rate,
                                                kernel_size = kernel_size)
                                                for i in range(num_layers)])

    def forward(self, x):
        input = x
        for i in range(self.num_layers):
            x = self.bottles[i](x)
            x = self.convs[i](x)
            x = torch.cat((input, x), dim = 1)
        return x

class TransitionBlock(nn.Module):
    """returns one transition block of DenseNet."""
    def __init__(self, input_channels, kernel_size = 1, stride = 1, 
                    compression = 0.5, pool_size = 2):
        super(TransitionBlock, self).__init__()
        output_channels = int(input_channels * compression)
        self.conv = AutoConv2d_BN(input_channels, output_channels, 
                              kernel_size = kernel_size, stride = stride)
        self.pool = nn.AvgPool2d((pool_size, pool_size))

    def forward(self, x):
        x = self.pool(self.conv(x))
        return x

class densechunk(nn.Module):
    """returns one dense block and one transition block together."""
    def __init__(self, input_channels, growth_rate = 4, kernel_size = 3,
                    num_layers = 4, bottleneck_width = 4, compression = 0.5):
        super(densechunk, self).__init__()
        self.dense = DenseBlock(input_channels, kernel_size = kernel_size,
                                growth_rate = growth_rate, num_layers = num_layers, 
                                bottleneck_width = bottleneck_width)
        self.transition = TransitionBlock(input_channels + (growth_rate * num_layers), 
                                            compression = compression,
                                            kernel_size = 1)

    def forward(self, x):
        x = self.dense(x)
        x = self.transition(x)
        return x
    
class densenetlike(nn.Module):
    """returns DenseNet-like neural network architecture."""
    def __init__(self, image_size = (3, 32, 32), in_channels = None, num_chunks = 4, 
                    drop_rate = 0.10, growth_rate = 12, kernel_size = 3, num_layers = 8, 
                    bottleneck_width = 4, compression = 0.50, num_classes = 10):
        super(densenetlike, self).__init__()
        in_channels = 2 * growth_rate if in_channels is None else in_channels
        self.out_channels = [int(in_channels * (compression ** i) + 
                                growth_rate * num_layers * sum(compression ** j 
                                for j in range(1, i + 1))) for i in range(num_chunks + 1)]
        self.conv = AutoConv2d_BN(image_size[0], in_channels, kernel_size = 5)
        self.chunks = nn.ModuleList([densechunk
                                (input_channels = self.out_channels[i],
                                growth_rate = growth_rate,
                                kernel_size = kernel_size,
                                num_layers = num_layers,
                                bottleneck_width = bottleneck_width,
                                compression = compression)
                                for i in range(num_chunks)])
        self.fblock = DenseBlock(input_channels = self.out_channels[-1],
                                kernel_size = kernel_size,
                                growth_rate = growth_rate,
                                num_layers = num_layers,
                                bottleneck_width = bottleneck_width)
        self.drop = nn.Dropout2d(drop_rate)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.out_channels[-1] + growth_rate * num_layers, num_classes)
    
    def forward(self, x):
        x = self.conv(x)
        for i, chunk in enumerate(self.chunks):
            x = chunk(x)
            if i % 2 == 0:
                x = self.drop(x)
        x = self.fblock(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

        
    