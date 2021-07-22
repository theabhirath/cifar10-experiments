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
    def __init__(self, input_channels, output_channels, kernel_size, down_stride = 1):
        super(VanillaResBlock, self).__init__()
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

class BottleneckResBlock(nn.Module):
    """bottleneck block that downsamples input."""
    def __init__(self, input_channels, output_channels, kernel_size, 
                    bottleneck_channels = None, down_stride = 1):
        bottleneck_channels = output_channels // 4 if bottleneck_channels is None else bottleneck_channels
        super(BottleneckResBlock, self).__init__()
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

class resnetlike(nn.Module):
    """returns ResNet-like neural network architecture."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding = 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.block1 = VanillaResBlock(32, 32, 3)
        self.block2 = VanillaResBlock(32, 32, 3)
        self.block3 = VanillaResBlock(32, 64, 3, down_stride = 2)
        self.block4 = BottleneckResBlock(64, 64, 3)
        self.block5 = BottleneckResBlock(64, 64, 3)
        self.block6 = BottleneckResBlock(64, 128, 3, down_stride = 2)
        self.block7 = BottleneckResBlock(128, 128, 3)
        self.block8 = BottleneckResBlock(128, 128, 3)
        self.block9 = BottleneckResBlock(128, 256, 3, down_stride = 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

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
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x