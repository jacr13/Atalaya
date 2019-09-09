"""
Simple example of Atalaya with code from :
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from atalaya import Logger


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def main():
    logger = Logger(
        name="exp",  # name of the logger
        path="logs",  # path to logs
        verbose=True,  # logger in verbose mode
        grapher="visdom",
        server="http://localhost",
        port=8097,,
    )
    # the model
    net = Net()
    logger.add("model", net)

    # the loss
    criterion = nn.MSELoss()

    # the optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    logger.add("optimizer", optimizer)

    # data
    inputs = torch.randn(100, 1, 32, 32)
    targets = torch.randn(100, 10)
    targets = targets.view(1, -1)
    
    for epoch in range(100):
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

    logger.close()

if __name__ == "__main__":
    main()
