import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class SimpleCNN(nn.Module):
    def __init__(self, img_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten_features = 64 * (img_size[0] // 4) * (img_size[1] // 4)

        self.fc1 = nn.Linear(self.flatten_features, 128)
        self.fc2 = nn.Linear(128, 2) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x