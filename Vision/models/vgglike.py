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
        x = F.relu(self.bn(self.conv(x)))
        return x

class vggblock(nn.Module):
    """block of a VGG-like neural network."""
    def __init__(self, in_channels, kernel_size = 3, drop_rate = 0.2, num_layers = 2,
                    expansion = 2, pool_size = 2):
        super(vggblock, self).__init__()
        out_channels = in_channels * expansion
        self.convs = nn.ModuleList([AutoConv2d_BN(in_channels, in_channels, kernel_size) for 
                                            _ in range(num_layers - 1)])
        self.convf = AutoConv2d_BN(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(pool_size)
        self.drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        for conv in self.convs:
            x = self.drop(conv(x))
        x = self.convf(x)
        x = self.pool(x)
        return x

class vgglike(nn.Module):
    """
    neural network architecture based on VGG-Net.

    inputs:

    1. `image_size`: size of input image (assumed to be three-dimensional of type (C, H, W) 
    where C is channels, H is height, and W is width). Default value is (3, 32, 32).
    2. `in_channels`: number of channels the image is upsampled to before feeding into the main
    network. (assumed to be an integer equal to or larger than `image_size[0]`). Default value 
    is 32.
    3. `kernel_size`: size of convolutional kernels throughout the network. Default value is 3.
    4. `expansion`: factor by which the channel size is upsampled at each chunk of the network. 
    Default value is 2.
    5. `pool_size`: size of pooling windows across the image in each block. Usually equal to 
    `expansion`.
    6. `drop_rate`: dropout rate for all intermittent layers in the network - this is 
    2D dropout for the convolutional blocks i.e. entire channel values are zeroed and regular 
    dropout for the fully connected layer. Default value is 0.2.
    7. `num_blocks`: number of blocks, each of which has `num_layers` convolutional layers. 
    Default value is 2.
    8. `num_layers`: number of layers in each block in the network. Default value is 2.
    9. `num_classes`: number of classes to predict. Default value is 10.
    """
    def __init__(self, image_size = (3, 32, 32), in_channels = 32, kernel_size = 3, 
                    expansion = 2, pool_size = None, drop_rate = 0.2, num_blocks = 3, 
                    num_layers = 2, num_classes = 10):
        super(vgglike, self).__init__()
        pool_size = expansion if pool_size is None else pool_size
        self.conv1 = nn.Conv2d(image_size[0], in_channels, 5, padding = 2)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.blocks = nn.ModuleList([vggblock(in_channels * (expansion ** i), 
                                        kernel_size = kernel_size, 
                                        drop_rate = drop_rate, 
                                        num_layers = num_layers,
                                        expansion = expansion, 
                                        pool_size = pool_size) for i in range(num_blocks)])
        im_after_size = (in_channels * expansion ** (num_blocks), 
                            image_size[1] // (pool_size ** num_blocks), 
                            image_size[2] // (pool_size ** num_blocks))
        self.fc1 = nn.Linear(im_after_size[0] * im_after_size[1] * im_after_size[2], 
                                in_channels * (expansion ** num_blocks) // 2)
        self.fc2 = nn.Linear(in_channels * (expansion ** num_blocks) // 2, num_classes)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x