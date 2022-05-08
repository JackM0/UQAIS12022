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

# File to store data in
stored = ".\DatasetDepth.pt"

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
        data = (rf.Network(f).impulse_response())[1][10:2085]
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

torch.save(Dataset, stored)