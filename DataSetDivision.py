from email import generator
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import pylab as py
import torch 
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
    # Calculate from 1 to 30 which label we need to assign
    labelPosition = (yy.index(diameter) * (len(mm))) + mm.index(depth)
    # print(labelPosition)
    # Construct the binary vector with a 1 in the right position
    label = np.zeros(len(mm) * len(yy))
    label[labelPosition] = 1
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
Train, Test = torch.utils.data.random_split(Dataset, [fileNum - testNum, testNum], generator = torch.Generator().manual_seed(42))
print(len(Train))
print(len(Test))