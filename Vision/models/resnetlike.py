import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoConv2d_BN(nn.Module):
    """
    custom block that implements 'same' padding for a convolution followed by 
    batch normalisation
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
    """
    simple residual block with 2 `AutoConv2d_BN` layers and a shortcut mapping
    """
    def __init__(self, input_channels, output_channels, kernel_size):
        super(VanillaResBlock, self).__init__()
        self.conv1 = AutoConv2d_BN(input_channels, output_channels, 
                                  kernel_size = kernel_size)
        self.conv2 = AutoConv2d_BN(input_channels, output_channels, 
                                  kernel_size = kernel_size)

    def forward(self, x):
        input = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(x + input)
        return x

class BottleneckResBlock(nn.Module):
    """
    bottleneck residual block to reduce compute in the deeper layers
    """
    def __init__(self, input_channels, bottleneck_channels, output_channels, kernel_size):
        super(BottleneckResBlock, self).__init__()
        self.conv1 = AutoConv2d_BN(input_channels, bottleneck_channels, 
                                   kernel_size = 1)
        self.conv2 = AutoConv2d_BN(bottleneck_channels, bottleneck_channels, 
                                   kernel_size = kernel_size)
        self.conv3 = AutoConv2d_BN(bottleneck_channels, output_channels, 
                                   kernel_size = 1)

    def forward(self, x):
        input = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.relu(x + input)
        return x

class DownsampleResBlock(nn.Module):
    """
    simple residual block with downsampling of images
    """
    def __init__(self, input_channels, output_channels, kernel_size, down_stride):
        super(DownsampleResBlock, self).__init__()
        self.conv1 = AutoConv2d_BN(input_channels, output_channels, 
                                   kernel_size = kernel_size, stride = down_stride)
        self.conv2 = AutoConv2d_BN(output_channels, output_channels, 
                                   kernel_size)
        self.conv3 = AutoConv2d_BN(input_channels, output_channels, 
                                   kernel_size = 1, stride = down_stride)

    def forward(self, x):
        input = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(x + self.conv3(input))
        return x

class DownbottleResBlock(nn.Module):
    """
    bottleneck block that downsamples images
    """
    def __init__(self, input_channels, bottleneck_channels , output_channels, kernel_size, down_stride):
        super(DownbottleResBlock, self).__init__()
        self.conv1 = AutoConv2d_BN(input_channels, bottleneck_channels, 
                                   kernel_size = 1)
        self.conv2 = AutoConv2d_BN(bottleneck_channels, bottleneck_channels, 
                                   kernel_size = kernel_size, stride = down_stride)
        self.conv3 = AutoConv2d_BN(bottleneck_channels, output_channels, 
                                   kernel_size = 1)
        self.conv4 = AutoConv2d_BN(input_channels, output_channels, 
                                   kernel_size = 1, stride = down_stride)

    def forward(self, x):
        input = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = F.relu(x + self.conv4(input))
        return x

class ResNetlike(nn.Module):
    """
    returns ResNet-like neural network architecture.
    """
    def __init__(self, nclasses):
        super().__init__()
        self.nclasses = nclasses
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.bn1 = nn.BatchNorm2d(32)
        self.block1 = VanillaResBlock(32, 32, 3)
        self.block2 = VanillaResBlock(32, 32, 3)
        self.block3 = DownsampleResBlock(32, 64, 3, 2)
        self.block4 = BottleneckResBlock(64, 16, 64, 3)
        self.block5 = BottleneckResBlock(64, 16, 64, 3)
        self.block6 = DownbottleResBlock(64, 32, 128, 3, 2)
        self.block7 = BottleneckResBlock(128, 32, 128, 3)
        self.block8 = BottleneckResBlock(128, 32, 128, 3)
        self.block9 = DownbottleResBlock(128, 64, 256, 3, 2)
        self.gap = nn.AvgPool2d((4, 4))
        self.fc = nn.Linear(256, self.nclasses)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x