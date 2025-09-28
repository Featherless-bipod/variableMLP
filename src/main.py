import numpy as np 
import pandas as pd 
import nHnN_MLP as nMLP
import matplotlib.pyplot as plt
import time
import test as t

route = '../data/TMNIST_data.csv'
#route = 
testAmount = 1000
scale = 255
results = 10
rem_axis = "names"

#insert model paramters
hidden_layers = 2
nodes_per_layer = 10
learning_rate = 0.1
epochs = 1000

test_y,test_x,train_y,train_x, pixels,train_trials,results = nMLP.process_data(route, testAmount, scale, results,rem_axis=rem_axis)
#params = nMLP.gradient_descent(hidden_layers,nodes_per_layer,train_x,train_y,pixels,results,epochs,learning_rate)

neurons = [10,16,32,64,128,256]
accs = []
times = []

for N in neurons:
    print(f"neuron number:{N}")
    acc, dur = t.train_and_evaluate(hidden_layers, N=N, train_x=train_x, train_y=train_y,
                                  test_x=test_x, test_y=test_y,
                                  pixels=pixels, results=10, iters=500, alpha=0.01)
    accs.append(acc)
    times.append(dur)

plt.plot(neurons, accs, marker='o')
plt.xlabel("Neurons per hidden layer")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Hidden Layer Width")
plt.show()