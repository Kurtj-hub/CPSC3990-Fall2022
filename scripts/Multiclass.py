import pandas as pd
import torch.nn as nn
import torch
import numpy as np
import sklearn.preprocessing as skp
import sklearn.metrics as skm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

# See this for better dataloading
# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel


# dataset definition
class SpatialData16(Dataset):
    # Using class group to specify included groups
    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    # 2^16 max = 65536, min = 0 but more likely 1.
    # bit vector would be faster but this is more legible. 
    
    def __init__(self, INTclassGroup):
    	# this will depend on class group....
        # load the csv file as a dataframe loop 
        powerOf2 = 15
        finalDF = pd.DataFrame()
        while INTclassGroup > 0:
            # print("BORK")
            if(INTclassGroup - (2**powerOf2)) >= 0:
                #print(INTclassGroup)
                INTclassGroup = INTclassGroup - 2**powerOf2
                tempDF = pd.read_csv('../data/16Classes/class{}Data.csv'.format(powerOf2),sep = ',', header = 0)
                finalDF = finalDF.append(tempDF, ignore_index=True)
            #    print("Meow")
            powerOf2 = powerOf2 - 1
        #print(finalDF.values[:, :-1])
        #print(finalDF.values[:, -1])
        self.X = finalDF.values[:, :-1]
        self.y = finalDF.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = skp.LabelEncoder().fit_transform(self.y)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.1):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

# model definition
class Spatial16Class(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(Spatial16Class, self).__init__()
        # input to first hidden layer Maintains Ratio of input to hidden.
        self.hidden1 = nn.Linear(n_inputs, 40)
        self.act1 = nn.Tanh()
        # second hidden layer
        self.hidden2 = nn.Linear(40, 40)
        self.act2 = nn.Tanh()
        # third hidden layer
        self.hidden3 = nn.Linear(40, 40)
        self.act3 = nn.Tanh()
        # output ? 
        self.out = nn.LogSoftmax()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # output layer
        X = self.hidden3(X)
        X = self.act3(X)
        X = self.out(X) # warning that implicit dim choice is depreciated
        return X

# prepare the dataset
def prepare_data(INTclassGroup):
    # load the dataset
    dataset = SpatialData16(INTclassGroup)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=100, shuffle=True)
    test_dl = DataLoader(test, batch_size=100, shuffle=True)
    return train_dl, test_dl

# train_stream
def train_stream(train_dl, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(2):
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(train_dl):
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = model(inputs)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()

# train the model
def train_model(train_dl, model):
    # Just have to reset the data everytime here?
    # define the optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # enumerate epochs
    for epoch in range(2):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # convert to class labels
        yhat = np.argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # calculate accuracy
    acc = skm.accuracy_score(actuals, predictions)
    return acc

# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = torch.Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

# prepare the data
print("flag1")
train_dl, test_dl = prepare_data(65534)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
print("flag2")
model = Spatial16Class(4)
# train the model
train_model(train_dl, model)
print("flag3")
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
print("flag4")
# make a single prediction
row = [5.1,3.5,1.4,0.2]
yhat = predict(row, model)
print('Predicted: %s (class=%d)' % (yhat, np.argmax(yhat)))

# Change dataset

train_dl, test_dl = prepare_data(65534)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
print("flag2")
#model = Spatial16Class(4) # removed to test just changing dataset
# train the model
train_model(train_dl, model)
print("flag3")
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
print("flag4")
# make a single prediction
row = [5.1,3.5,1.4,0.2]
yhat = predict(row, model)
print('Predicted: %s (class=%d)' % (yhat, np.argmax(yhat)))

