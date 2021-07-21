import torch.nn as nn
import torch.nn.functional as F

class ResNeXtBlock(nn.Module):
    """bottleneck ResNeXt block with grouped convolutions."""
    def __init__(self, input_channels, output_channels, kernel_size, down_stride, cardinality, bottleneck_channels = None):
        super(ResNeXtBlock, self).__init__()
        bottleneck_channels = input_channels // 2 if bottleneck_channels is None else bottleneck_channels
        self.conv1 = nn.Conv2d(input_channels, bottleneck_channels, 
                                   kernel_size = 1)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 
                                   kernel_size = kernel_size, groups = cardinality, stride = down_stride, padding = kernel_size // 2)
        self.conv3 = nn.Conv2d(bottleneck_channels, output_channels, 
                                   kernel_size = 1)
        self.conv4 = nn.Conv2d(input_channels, output_channels, 
                                   kernel_size = 1, stride = down_stride)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.bn3 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        input = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.relu(x + self.conv4(input))
        return x

class resnextlike(nn.Module):
    """returns ResNeXt-like neural network architecture."""
    def __init__(self, cardinality = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding = 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.block1 = ResNeXtBlock(64, 128, 3, 2, cardinality)
        self.block3 = ResNeXtBlock(128, 256, 3, 2, cardinality)
        self.block3 = ResNeXtBlock(256, 512, 3, 2, cardinality)
        self.block4 = ResNeXtBlock(512, 1024, 3, 2, cardinality)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

