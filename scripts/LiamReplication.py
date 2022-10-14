import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as pypt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import nn

import warnings
warnings.filterwarnings("ignore")

device = "cpu" #cpu is faster on my laptop...
#device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

AllData = pd.read_csv('../data/16Classes/allData.csv',sep = ',', header = 0)
class0 = pd.read_csv('../data/16Classes/class0Data.csv')
class1 = pd.read_csv('../data/16Classes/class1Data.csv')
class2 = pd.read_csv('../data/16Classes/class2Data.csv')
class3 = pd.read_csv('../data/16Classes/class3Data.csv')
class4 = pd.read_csv('../data/16Classes/class4Data.csv')
class5 = pd.read_csv('../data/16Classes/class5Data.csv')
class6 = pd.read_csv('../data/16Classes/class6Data.csv')
class7 = pd.read_csv('../data/16Classes/class7Data.csv')
class8 = pd.read_csv('../data/16Classes/class8Data.csv')
class9 = pd.read_csv('../data/16Classes/class9Data.csv')
class10 = pd.read_csv('../data/16Classes/class10Data.csv')
class11 = pd.read_csv('../data/16Classes/class11Data.csv')
class12 = pd.read_csv('../data/16Classes/class12Data.csv')
class13 = pd.read_csv('../data/16Classes/class13Data.csv')
class14 = pd.read_csv('../data/16Classes/class14Data.csv')
class15 = pd.read_csv('../data/16Classes/class15Data.csv')

print(AllData)

print(AllData["Classification"])

# Neural Network Class

class spatialStreamProcessorNN(nn.Module):
    def __init__(self):
        super(spatialStreamProcessorNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits

model = spatialStreamProcessorNN().to(device)
print(model)

input_image = torch.rand(3,28,28)
print(input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

layer1 = nn.Linear(in_features=28*28, out_features = 20)
hidden1 = layer1(flat_image)
print(hidden1.size())

print(f"Before ReLU: {hidden1}\n\n")
#hidden1 = nn.RelU()(hidden1)
print(f"After ReLU: {hidden1}")

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20,10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)

# Model Parameters
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")