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
fileNum = 1903

# Number of files to be use as test data
testNum = 100

# Location of data
stored = ".\DatasetDepthDiameter.pt"

# Load Dataset
Dataset = torch.load(stored)

# hyperparameters
input_size = 2085 - 10 #Length of truncated signal
output_size = 30 # 5 Different Depths, 6 Different Diameters
hidden_size = 1300 # Not sure what to pick this these parameters, just run training with all of them and see whats best?

epochs = 100
batch_size = 64
learning_rate = 0.000001

# Split into training and testing data
# Might be better to split in a more fair way
Train, Test = torch.utils.data.random_split(Dataset, [fileNum - testNum, testNum], generator = torch.Generator().manual_seed(42))



class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.bnorm = nn.BatchNorm1d(hidden_size)
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.bnorm(x)
        x = self.relu(x)
        x = self.l4(x)
        return F.log_softmax(x, dim=0)

net = Network()
print(net)

optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

data_loader = torch.utils.data.DataLoader(Train, batch_size, shuffle = True)
data_test = torch.utils.data.DataLoader(Test, testNum, shuffle = True)

def train(dataset, network):
    for e in range(epochs):
        for data in dataset:
            x_var = data[0]
            y_var = data[1]

            #x_var = x_var.to(device, torch.float)
            #y_var = y_var.to(device, torch.long)
            
            optimizer.zero_grad()
            net_out = network(x_var)

            loss = loss_func(net_out, y_var)
            loss.backward()
            optimizer.step()
            
        print('Epoch: {} - Loss: {:.6f}'.format(e, loss.item()))

def evaluate(validation, network):
    correct = 0
    for data in validation:
        net_out = network(data[0])
        print(net_out)
        a = net_out.argmax(1)
        b = data[1].argmax(1)
        diff = a - b
        for element in diff:
            if element == 0:
                correct += 1
    
    error = (100 - correct) 
    print(correct)
    print(error)
    

train(data_loader, net)
evaluate(data_test, net)