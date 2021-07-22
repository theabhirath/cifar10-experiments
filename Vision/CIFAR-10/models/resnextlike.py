import torch.nn as nn
import torch.nn.functional as F

class ResNeXtBlock(nn.Module):
    """bottleneck ResNeXt block with grouped convolutions."""
    def __init__(self, input_channels, output_channels, kernel_size, 
                    cardinality, bottleneck_width, down_stride = 1):
        super(ResNeXtBlock, self).__init__()
        self.bottleneck_channels = bottleneck_width * cardinality
        self.conv1 = nn.Conv2d(input_channels, self.bottleneck_channels, 
                                kernel_size = 1)
        self.conv2 = nn.Conv2d(self.bottleneck_channels, self.bottleneck_channels, 
                                kernel_size = kernel_size, groups = cardinality, 
                                stride = down_stride, padding = kernel_size // 2)
        self.conv3 = nn.Conv2d(self.bottleneck_channels, output_channels, 
                                kernel_size = 1)
        self.conv4 = nn.Conv2d(input_channels, output_channels, 
                                kernel_size = 1, stride = down_stride)
        self.bn1 = nn.BatchNorm2d(self.bottleneck_channels)
        self.bn2 = nn.BatchNorm2d(self.bottleneck_channels)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.bn4 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        input = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = F.relu(x + self.bn4(self.conv4(input)))
        return x

class resnextlike(nn.Module):
    """returns ResNeXt-like neural network architecture."""
    def __init__(self, in_channels = 32, cardinality = 32, bottleneck_width = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, in_channels, 5, padding = 2)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.block1 = ResNeXtBlock(in_channels, 2 * in_channels, 3, down_stride = 2, 
                                    cardinality = cardinality, 
                                    bottleneck_width = bottleneck_width // 2)
        self.block2 = ResNeXtBlock(2 * in_channels, 2 * in_channels, 3, 
                                    cardinality = cardinality, 
                                    bottleneck_width = bottleneck_width // 2)
        self.block3 = ResNeXtBlock(2 * in_channels, 4 * in_channels, 3, down_stride = 2, 
                                    cardinality = cardinality, 
                                    bottleneck_width = bottleneck_width)
        self.block4 = ResNeXtBlock(4 * in_channels, 4 * in_channels, 3, 
                                    cardinality = cardinality, 
                                    bottleneck_width = bottleneck_width)
        self.block5 = ResNeXtBlock(4 * in_channels, 8 * in_channels, 3, down_stride = 2, 
                                    cardinality = cardinality, 
                                    bottleneck_width = 2 * bottleneck_width)
        self.block6 = ResNeXtBlock(8 * in_channels, 16 * in_channels, 3, down_stride = 2, 
                                    cardinality = cardinality, 
                                    bottleneck_width = 2 * bottleneck_width)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16 * in_channels, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

