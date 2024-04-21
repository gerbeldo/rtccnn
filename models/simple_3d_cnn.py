import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(
            in_channels=1, out_channels=16, kernel_size=(3, 3, 3), stride=1, padding=1
        )
        self.conv2 = nn.Conv3d(
            in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=1
        )
        self.conv3 = nn.Conv3d(
            in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=1
        )

        # Pooling layer
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=64 * 9 * 9 * 5, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        # Apply 3D convolutions followed by pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # print(f"Shape after convolutions: {x.shape}")

        # Dynamically calculate the correct number of features for fc1
        # num_features = x.size(1) * x.size(2) * x.size(3) * x.size(4)
        # print(f"num_features: {num_features}")
        # now that I know the correct number, set it as in_features
        num_features = 64 * 9 * 9 * 2

        # Adjust the input size for the first fully connected layer based on the actual size
        self.fc1 = nn.Linear(in_features=num_features, out_features=512).to(x.device)

        # Flatten the output for the fully connected layer
        x = x.view(-1, num_features)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Apply sigmoid activation function to the output layer for binary classification
        x = torch.sigmoid(x)

        return x
