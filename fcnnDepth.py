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

# Directory where data is located
path = ".\\Data"

# Setups the strings to look for when contructing the binary vectors
yy = ["10", "12", "16", "20", "24", "28"]
mm = ["10", "30", "50", "70", "90"]

# Number of files
fileNum = 1903

# Number of files to be use as test data
testNum = 100

# Constructs the binary vector for the given file/data
def getLabel(filename):
    # Grab the diameter and depth value
    filename = filename[11:]
    diameter = filename[0:2]
    depth = filename[8:10]

    depthOnlyPosition = mm.index(depth)
    label = np.zeros(len(mm))

    label[depthOnlyPosition] = 1
    #print(label)
    return label

# Can be used to specify how many files are to be opened
numOpen = fileNum

first = 1
i = 0
for filename in os.listdir(path):
    if (i >= numOpen):
        break
    # Get the file
    f = os.path.join(path, filename)
    print(f"Opening {f}")

    # Get the associated data and binary vector
    if os.path.isfile(f):
        data = (rf.Network(f).impulse_response())[1]
        label = getLabel(filename)
        if first == 1:
            Data = [data]
            Labels = [label]
            first = 0
        else:
            # Stack the data and labels ontop of each other
            Data = np.concatenate((Data, [data]), axis = 0)
            Labels = np.concatenate((Labels, [label]), axis = 0)
    i = i + 1

# Convert arrays to tensors
Data = torch.tensor(Data, dtype=torch.float)
Labels = torch.tensor(Labels)
print(Data)
print(Labels)

# Convert to torch dataset
Dataset = torch.utils.data.TensorDataset(Data, Labels)

# Split into training and testing data
# Might be better to split in a more fair way
Train, Test = torch.utils.data.random_split(Dataset, [numOpen - testNum, testNum], generator = torch.Generator().manual_seed(42))
print(len(Train))
print(len(Test))


# hyperparameters
input_size = 4000
output_size = 5
hidden_size = 800

epochs = 500
batch_size = 500
learning_rate = 0.0005

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l3(x)
        return F.log_softmax(input = x, dim = 1)

net = Network()
print(net)


optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

loss_log = []

for e in range(epochs):
    for i in range(0, Data.shape[0], batch_size):
        x_mini = Data[i:i + batch_size] 
        y_mini = Labels[i:i + batch_size] 
        
        x_var = Variable(x_mini)
        y_var = Variable(y_mini)
        
        optimizer.zero_grad()
        net_out = net(x_var)

        loss = loss_func(net_out, y_var)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            loss_log.append(loss.item())
        
    print('Epoch: {} - Loss: {:.6f}'.format(e, loss.item()))
       

plt.figure(figsize=(10,8))
plt.plot(loss_log)
plt.show()
