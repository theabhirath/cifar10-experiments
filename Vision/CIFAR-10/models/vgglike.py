import torch.nn as nn
import torch.nn.functional as F

class vgglike(nn.Module):
    """returns VGG-like neural network architecture."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size = 3, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(p = 0.10)
        self.dropout2 = nn.Dropout(p = 0.20)
        self.fc1 = nn.Linear(4 * 4 * 256, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = self.dropout1(F.relu(self.conv4(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv5(x)))
        x = self.dropout1(F.relu(self.conv6(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x