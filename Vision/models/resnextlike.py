import torch.nn as nn
import torch.nn.functional as F

class ResNeXtBlock(nn.Module):
    """bottleneck ResNeXt block with grouped convolutions."""
    def __init__(self, input_channels, output_channels, 
                    cardinality, bottleneck_width, kernel_size = 3, down_stride = 1):
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

class ResNeXtchunk(nn.Module):
    """returns two ResNeXt blocks with downsampling and connected by residual learning."""
    def __init__(self, input_channels, expansion = 2, down_stride = 2,
                    cardinality = 32, bottleneck_width = 4):
        super().__init__()
        self.block1 = ResNeXtBlock(input_channels, input_channels, cardinality, 
                                    bottleneck_width)
        self.block2 = ResNeXtBlock(input_channels, expansion * input_channels,
                                    cardinality, bottleneck_width, down_stride = down_stride)
        self.conv = nn.Conv2d(input_channels, expansion * input_channels, 
                                kernel_size = 1, stride = down_stride)
    
    def forward(self, x):
        residual = x
        x = self.block1(x)
        x = self.block2(x)
        x += self.conv(residual)
        return x

class resnextlike(nn.Module):
    """returns ResNeXt-like neural network architecture."""
    def __init__(self, image_size = (3, 32, 32), in_channels = 64, num_chunks = 2, 
                    expansion = 2, cardinality = 64, bottleneck_width = 4, 
                    num_classes = 10):
        super(resnextlike, self).__init__()
        self.conv1 = nn.Conv2d(image_size[0], in_channels, 5, padding = 2)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.chunks = nn.ModuleList([ResNeXtchunk(in_channels * (expansion ** i), 
                                                    expansion = expansion, 
                                                    cardinality = cardinality, 
                                                    bottleneck_width = bottleneck_width) 
                                                    for i in range(num_chunks)])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels * (expansion ** num_chunks), num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for chunk in self.chunks:
            x = chunk(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.fc(x)
        return x
