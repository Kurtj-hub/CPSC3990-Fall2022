import pandas as pd
import torch.nn as nn
import torch
import array
import numpy as np
import sklearn.preprocessing as skp
import sklearn.metrics as skm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

# TO DO:
# 	1. Separate into Class and implementation file.  
#	2. Write the algorithms. 
#	3. Figure out better evaluation? 
#	4. Write Tests

# See this for alternative dataloading for stream processing.
# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel


# dataset definition
class SpatialData16(Dataset):
	# Using class group to specify included groups
	# _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
	# 2^16 max = 65536, min = 0 but more likely 1.
	# bit vector would be faster but this is more legible. 	
	def __init__(self, boolVector16): # use raw bool vector or convert using intToBool function. 
		# this will depend on class group....
		# load the csv file as a dataframe loop 
		powerOf2 = 15
		finalDF = pd.DataFrame()
		counter = 0 # there is a better way to do this but it works for now.
		self.boolVector = boolVector16 		
		for x in boolVector16:
			if x == True:
				tempDF = pd.read_csv('../data/16Classes/class{}Data.csv'.format(counter),sep = ',', header = 0)
				finalDF = pd.concat([finalDF, tempDF], ignore_index=True)				
			counter = counter + 1
			
		self.X = finalDF.values[:, :-1] 	# all but key
		self.y = finalDF.values[:, -1] 		# only key (e.g. Classigfication)
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

		
	def get_adjacency_torus(self):
		# Returns boolVector of adjacency x and y wrapped at edges. 
		# removes currently occupied sites from return value. 
		# if "0" boolVector all sites are adjacent to allow random start.
		# can set manually for randomization. 
		adjBoolVector16 = np.array([False] * 16)
		count = 0
		adjTest = False			# tests if the boolvector is empty or not.  
		
		# Finds adjacent sites
		for x in self.boolVector16: 	# inefficient but I can't be bothered to store the information. 
			if x == True:
				adjBoolVector16[(count - 3) % 15] = True			# top
				adjBoolVector16[(count + 4) % 15] = True			# bottom
				adjBoolVector16[(4*(count // 4)) + ((count+1) % 4)] = True	# right
				adjBoolVector16[(4*(count // 4)) + ((count-1) % 4)] = True	# left
				adjTest = True
			count = count + 1
		
		# Returns for case of adjBool Vector being entirely False. For potential Random Start.
		if adjTest == False:
			return np.array([True] * 16)
		
		# remove current sites to prevent non-expansion...
		count = 0
		for x in self.boolVector16: 
			if x == True:
				adjBoolVector16[x] = False
			count = count + 1
		return adjBoolVector16
						 
		
	def get_adjacency_cartesian(self):
		# Returns boolVector of adjacency x and y not wrapped. Corners and Edges have fewer adjacent. 
		# removes currently occupied sites from return value. 
		# if "0" boolVector all sites are adjacent to allow random start.
		# duplicate code can probably write this better with a parameter but the clarity may be useful. 
		adjBoolVector16 = np.array([False] * 16)
		count = 0
		adjTest = False			# tests if the boolvector is empty or not. 
		for x in self.boolVector16: # inefficient but I can't be bothered to store the information. 
			if x == True:
				if ((count + 1) % 4) != 0: 	# right
					adjBoolVector16[count + 1] = True
				if ((count - 1) % 4) != 3:	# left
					adjBoolVector16[count - 1] = True # worried about negative index but I don't think it is possible.
				if (count - 4) >= 0:		# Top
					adjBoolVector16[count - 1] = True 
				if (count + 4) <= 15:		# Bottom
					adjBoolVector16[count - 1] = True 
				adjTest = True 
			count = count + 1
		
		if adjTest == False:
			return np.array([True] * 16)
		
		count = 0
		for x in self.boolVector16: # remove current sites to prevent non-expansion...
			if x == True:
				adjBoolVector16[x] = False
			count = count + 1 
		return adjBoolVector16
		
	def get_bool_vector(self):
		return self.boolVector16


# model definition
class Spatial16Class(nn.Module):
	# define model elements
	def __init__(self, n_inputs):
		super(Spatial16Class, self).__init__()
		# input to first hidden layer Maintains Ratio of input to hidden as Liam (1 : 2.5).
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
	dataset = SpatialData16(intToBoolArray(INTclassGroup))
	# calculate split
	train, test = dataset.get_splits()
	# prepare data loaders
	train_dl = DataLoader(train, batch_size=100, shuffle=True)
	test_dl = DataLoader(test, batch_size=100, shuffle=True)
	return train_dl, test_dl


# train_stream
# Only trains a single for now to be updated for specific functions. 
def train_stream(train_dl, model):
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
		for epoch in range(1):
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
# Batch method example. 
def train_model(train_dl, model):
	# Just have to reset the data everytime here?
	# define the optimization
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	# enumerate epochs
	for epoch in range(10):
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
# will have to figure this out and modify it as req'd
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


# Int to bool wraps around with mod 	
def intToBoolArray(numInt):
	normal = numInt % 65536
	powerOf2 = 15
	boolArray = np.array([False] * 16)
	while powerOf2 >= 0:
		if(normal - (2**powerOf2)) >= 0:
				normal = (normal - (2**powerOf2))
				boolArray[powerOf2] = True
		powerOf2 = powerOf2 - 1
	return boolArray


# bool to int 	
def boolArrayToInt(boolArray16):
	numInt = 0
	powerOf2 = 0
	for x in boolArray16: # technically works for any array size may have to add in error handling. 
		if x == True:
			numInt = numInt + 2**powerOf2
		powerOf2 = powerOf2 + 1
	return numInt  
					

# prepare the data
print(35345)
print(intToBoolArray(35345))
meow = intToBoolArray(35345)
print(boolArrayToInt(meow))


train_dl, test_dl = prepare_data(20)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = Spatial16Class(4)
# train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc) # check calculation
# make a single prediction
# row = [5.1,3.5,1.4,0.2]
# yhat = predict(row, model)
# print('Predicted: %s (class=%d)' % (yhat, np.argmax(yhat)))


# Change dataset
train_dl, test_dl = prepare_data(13056) # bottom 4 quadrants
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
#model = Spatial16Class(4) # removed to test just changing dataset
# train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
# make a single prediction
# row = [5.1,3.5,1.4,0.2]
# yhat = predict(row, model)
# print('Predicted: %s (class=%d)' % (yhat, np.argmax(yhat)))	# May be useful for specifying / biasing expansion. 

