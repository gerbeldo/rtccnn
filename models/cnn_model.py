import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Example CNN architecture
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # More layers as needed...
        self.fc1 = nn.Linear(16 * 19 * 19, 64)  # Adjust the size
        self.fc2 = nn.Linear(64, 1)  # Binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # More operations as needed...
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
