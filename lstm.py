from email import generator
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import pylab as py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os

# Number of files
fileNum = 1467  # 1903

# Number of files to be use as test data
testNum = 100

# Location of data
stored = ".\DatasetDiameterNormalised.pt"

# Load Dataset
Dataset = torch.load(stored)

# hyperparameters
input_size = 2085 - 10  # Length of truncated signal
output_size = 30  # 5 Different Depths, 6 Different Diameters
hidden_size = 10  # Not sure what to pick this these parameters, just run training with all of them and see whats best?

epochs = 300
batch_size = 64
learning_rate = 0.01

# Split into training and testing data
# Might be better to split in a more fair way
Train, Test = torch.utils.data.random_split(
    Dataset, [fileNum - testNum, testNum], generator=torch.Generator().manual_seed(40)
)


class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_size):
        super(Network, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, target_size)

    def forward(self, input_):
        lstm_out, (h, c) = self.lstm(input_)
        logits = self.fc(lstm_out[:, :])
        scores = F.log_softmax(logits, dim=0)
        return scores

net = Network(input_size, hidden_size, output_size)
print(net)

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_func = nn.BCEWithLogitsLoss()

data_loader = torch.utils.data.DataLoader(Train, batch_size, shuffle=True)
data_test = torch.utils.data.DataLoader(Test, testNum, shuffle=True)


def train(dataset, network):
    for e in range(epochs):
        for data in dataset:
            x_var = data[0]
            y_var = data[1]

            # x_var = x_var.to(device, torch.float)
            # y_var = y_var.to(device, torch.long)

            optimizer.zero_grad()
            net_out = network(x_var)

            loss = loss_func(net_out, y_var)
            loss.backward()
            optimizer.step()

        print("Epoch: {} - Loss: {:.6f}".format(e, loss.item()))


def evaluate(validation, network):
    correct = 0
    incorrect = 0
    for data in validation:
        net_out = network(data[0])
        print(net_out)
        a = net_out.argmax(1)
        b = data[1].argmax(1)
        diff = a - b
        for element in diff:
            if element == 0:
                correct += 1
            else:
                incorrect += 1

    error = incorrect / (correct + incorrect)
    print(correct)
    print(error)


train(data_loader, net)
evaluate(data_loader, net)
evaluate(data_test, net)
