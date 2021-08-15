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
        x = self.bn(self.conv(x))
        return x

class VanillaResBlock(nn.Module):
    """simple residual block with downsampling of input."""
    def __init__(self, input_channels, output_channels, kernel_size = 3, down_stride = 1):
        super(VanillaResBlock, self).__init__()
        self.conv1 = AutoConv2d_BN(input_channels, output_channels, 
                                   kernel_size = kernel_size, stride = down_stride)
        self.conv2 = AutoConv2d_BN(output_channels, output_channels, 
                                   kernel_size)
        self.conv3 = AutoConv2d_BN(input_channels, output_channels, 
                                   kernel_size = 1, stride = down_stride)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(x + self.conv3(residual))
        return x

class BottleneckResBlock(nn.Module):
    """bottleneck block that downsamples input."""
    def __init__(self, input_channels, output_channels, kernel_size = 3, 
                    bottleneck_channels = None, down_stride = 1):
        super(BottleneckResBlock, self).__init__()
        if bottleneck_channels is None:
            bottleneck_channels = output_channels // 4
        self.conv1 = AutoConv2d_BN(input_channels, bottleneck_channels, 
                                   kernel_size = 1)
        self.conv2 = AutoConv2d_BN(bottleneck_channels, bottleneck_channels, 
                                   kernel_size = kernel_size, stride = down_stride)
        self.conv3 = AutoConv2d_BN(bottleneck_channels, output_channels, 
                                   kernel_size = 1)
        self.conv4 = AutoConv2d_BN(input_channels, output_channels, 
                                   kernel_size = 1, stride = down_stride)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.relu(x + self.conv4(residual))
        return x

class vanillaresnetchunk(nn.Module):
    """chunk with vanilla ResNet blocks chained together."""
    def __init__(self, input_channels, expansion = 2, num_blocks = 3, kernel_size = 3, 
                    down_stride = None):
        super(vanillaresnetchunk, self).__init__()
        down_stride = expansion if down_stride is None else down_stride
        output_channels = input_channels * expansion
        self.blocks = nn.ModuleList([VanillaResBlock(input_channels, input_channels,
                                                    kernel_size = kernel_size)
                                                    for _ in range(num_blocks - 1)])
        self.downblock = VanillaResBlock(input_channels, output_channels, 
                                        kernel_size = kernel_size, down_stride = down_stride)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.downblock(x)
        return x

class bottleresnetchunk(nn.Module):
    """chunk with bottleneck ResNet blocks chained together."""
    def __init__(self, input_channels, expansion = 2, num_blocks = 3, kernel_size = 3, 
                    down_stride = None):
        super(bottleresnetchunk, self).__init__()
        down_stride = expansion if down_stride is None else down_stride
        output_channels = input_channels * expansion
        self.blocks = nn.ModuleList([BottleneckResBlock(input_channels,
                                                    input_channels,
                                                    kernel_size = kernel_size)
                                                    for _ in range(num_blocks - 1)])
        self.downblock = BottleneckResBlock(input_channels, output_channels, 
                                            kernel_size = kernel_size, 
                                            down_stride = down_stride)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.downblock(x)
        return x

class resnetlike(nn.Module):
    """
    neural network architecture based on ResNet.

    inputs:

    1. `image_size`: size of input image (assumed to be three-dimensional of type (C, H, W) 
    where C is channels, H is height, and W is width). Default value is (3, 32, 32).
    2. `in_channels`: number of channels the image is upsampled to before feeding into the main
    network. (assumed to be an integer equal to or larger than `image_size[0]`). Default value 
    is 32.
    3. `kernel_size`: size of convolutional kernels throughout the network. Default value is 3.
    4. `expansion`: factor by which the channel size is upsampled at each chunk of the network. 
    Default value is 2.
    5. `down_stride`: stride of kernel when downsampling. Usually equal to `expansion`.
    Default value is 2.
    6. `vanilla_blocks`: number of residual blocks without bottleneck in each vanilla 
    chunk of the network. Default value is 3.
    7. `bottleneck_blocks`: number of residual blocks with bottleneck in each bottleneck
    chunk of the network. Default value is 3.
    8. `vanilla_chunks`: number of chunks with residual blocks without bottlenecks in the 
    network. Default value is 1.
    9. `bottleneck_chunks`: number of chunks with residual blocks with bottlenecks in the 
    network. Default value is 2. 
    10. `num_classes`: number of classes to predict. Default value is 10.
    """
    def __init__(self, image_size = (3, 32, 32), in_channels = 32, kernel_size = 3,
                    expansion = 2, down_stride = None, vanilla_blocks = 3, bottleneck_blocks = 3,
                    vanilla_chunks = 1, bottleneck_chunks = 2, num_classes = 10):
        super(resnetlike, self).__init__()
        down_stride = expansion if down_stride is None else down_stride
        num_chunks = vanilla_chunks + bottleneck_chunks
        self.conv1 = nn.Conv2d(image_size[0], in_channels, 5, padding = 2)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.vanillachunks = nn.ModuleList([vanillaresnetchunk(in_channels * (expansion ** i), 
                                                                num_blocks = vanilla_blocks, 
                                                                kernel_size = kernel_size,
                                                                down_stride = down_stride,
                                                                expansion = expansion)
                                                                for i in range(vanilla_chunks)]) 
                                                                
        self.bottlechunks = nn.ModuleList([bottleresnetchunk(in_channels * (expansion ** i),
                                                                num_blocks = bottleneck_blocks, 
                                                                kernel_size = kernel_size, 
                                                                down_stride = down_stride,
                                                                expansion = expansion)
                                                                for i in range(vanilla_chunks, 
                                                                                num_chunks)])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels * expansion ** (num_chunks), num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.vanillachunks:
            x = block(x)
        for block in self.bottlechunks:
            x = block(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.fc(x)
        return x