# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # patch_size=(50, 50)
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#
#         # First convolutional block
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.dropout1 = nn.Dropout(0.25)
#
#         # Second convolutional block
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.dropout2 = nn.Dropout(0.3)
#
#         # Fully connected layers
#         self.fc1 = nn.Linear(2304, 256)  # Manually set flattened size
#         self.bn5 = nn.BatchNorm1d(256)
#         self.dropout3 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(256, 1)  # Binary output
#
#     def forward(self, x):
#         # First convolutional block
#         x = self.pool1(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool1(F.relu(self.bn2(self.conv2(x))))
#         x = self.dropout1(x)
#
#         # Second convolutional block
#         x = self.pool2(F.relu(self.bn3(self.conv3(x))))
#         x = self.pool2(F.relu(self.bn4(self.conv4(x))))
#         x = self.dropout2(x)
#
#         # Flatten
#         x = x.view(x.size(0), -1)
#
#         # Fully connected layers
#         x = F.relu(self.bn5(self.fc1(x)))
#         x = self.dropout3(x)
#         x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
#         return x
import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)  # Adjust dimensions based on patch size
        self.fc2 = nn.Linear(128, 1)  # Binary output

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x
