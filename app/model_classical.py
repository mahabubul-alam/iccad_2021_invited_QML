import torch
import torch.nn as nn
import numpy as np
import pennylane as qml

torch.manual_seed(0)

n_class = 3
n_features = 196

class Net(nn.Module):
    # define nn
    def __init__(self, feature_extraction = False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 4, stride = 2)
        self.lr1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(16, 8, 4, stride = 2)
        self.lr2 = nn.LeakyReLU(0.1)
        self.fl1 = nn.Flatten()
        self.ln1 = nn.LayerNorm(32, elementwise_affine=True)

        self.fc1 = nn.Linear(32, n_class * 2)
        self.lr3 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(n_class * 2, n_class)

        self.extract = feature_extraction

    def forward(self, X):
        X = X.view(X.shape[0], 1, 14, 14)

        X = self.conv1(X)
        X = self.lr1(X)
        X = self.conv2(X)
        X = self.lr2(X)
        X = self.fl1(X)
        X = self.ln1(X)

        if self.extract:
            return X

        X = self.fc1(X)
        X = self.lr3(X)
        X = self.fc2(X)
        return X


if __name__ == '__main__':
    network = Net()
    random_input = torch.rand(1, n_features)
    print(network(random_input))