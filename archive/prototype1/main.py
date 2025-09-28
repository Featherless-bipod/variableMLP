import numpy as np 
import pandas as pd 
import nHnN_MLP as nMLP

#insert model paramters
hidden_layers = 2
nodes_per_layer = 10
learning_rate = 0.1
epochs = 1000

data = pd.read_csv('../data/TMNIST_Data.csv')
data = data.drop("names",axis=1) #first column of TMNIST_Data.csv is credits

data = np.array(data)
np.random.shuffle(data)
results = 10 #number of nodes output

test_data = data[0:1000].T
test_y  = test_data[0]
test_x  = test_data[1:]/255

train_data = data[1000:].T
train_y = train_data[0]
train_x = train_data[1:]/255

pixels, train_trials = train_x.shape

params = nMLP.gradient_descent(hidden_layers,nodes_per_layer,train_x,train_y,pixels,results,epochs,learning_rate)

