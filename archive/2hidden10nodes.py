import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

data = pd.read_csv('TMNIST_Data.csv')

data.head()
data = data.drop("names",axis=1)

#dataset initialization
data = np.array(data)
np.random.shuffle(data)

test_data = data[0:1000].T
test_Y = test_data[0]
test_X = test_data[1:]/255

train_data = data[1000:].T
train_Y = train_data[0]
train_X = train_data[1:]/255
pixels, m = train_X.shape


#neuron amount
H1 = 10
H2 = 10
Output = 10

def init_parameters(H1,H2,pixels,results):
    W3 = np.random.rand(H1,pixels)-0.5
    B3 = np.random.rand(H1,1)-0.5
    W2 = np.random.rand(H2,H1)-0.5
    B2 = np.random.rand(H2,1)-0.5
    W1 = np.random.rand(results,H2)-0.5
    B1 = np.random.rand(results,1)-0.5
    return W3,B3,W2,B2,W1,B1

def ReLu(Z):
    return np.maximum(0,Z)

def sig(Z):
    return 1/(1+np.exp(-Z))

def softmax(Z):
    return np.exp(Z)/sum(np.exp(Z))

def forward_propogation(X,W3,B3,W2,B2,W1,B1):
    Z2 = W3.dot(X)+B3
    A2 = ReLu(Z2)
    Z1 = W2.dot(A2)+B2
    A1 = sig(Z1)
    Z = W1.dot(A1)+B1
    A = softmax(Z)
    return Z2,A2,Z1,A1,Z,A

def one_hot(Y):
    Y_one_hot = np.zeros((Y.size, Y.max()+1)) #LEARN NEED (())
    Y_one_hot[np.arange(Y.size),Y] = 1    #LEARN
    return Y_one_hot.T


def derv_sig(Z):
    return np.exp(-Z)/(1+np.exp(-Z))**2


def derv_ReLu(Z):
    return Z > 0 #LEARN HOW TRUTH STATEMETNS IN ARRAYS WORK

def back_propogation(X,Y,Z2,A2,Z1,A1,Z,A,W1,W2):
    Y = one_hot(Y)

    #why 1/m everytime???
    
    dz1 = A-Y
    dw1 = 1/m * dz1.dot(A1.T)
    db1 = 1/m * np.sum(dz1)

    dz2 = W1.T.dot(dz1)*derv_sig(Z1)
    dw2 = 1/m*dz2.dot(A2.T) 
    db2 = 1/m*np.sum(dz2) #1/m * dz1.dot(W1.T).dot(derv_sig(Z1))

    dz3 = W2.T.dot(dz2) * derv_ReLu(Z2)
    dw3 = 1/m*dz3.dot(X.T)
    db3 = 1/m*np.sum(dz3)

    return dw1, db1, dw2, db2, dw3, db3
    

def get_accuracy(A,Y):
    return sum(np.argmax(A,0) == Y)/Y.size #LEARN ARGMAX
       
  

def gradient_descent(X,Y,alpha,iterations):
    W3,B3,W2,B2,W1,B1 = init_parameters(H1, H2, pixels, train_Y.max()+1)
    for i in range(iterations):
        Z2,A2,Z1,A1,Z,A = forward_propogation(train_X,W3,B3,W2,B2,W1,B1)
        dw1,db1,dw2,db2,dw3,db3 = back_propogation(train_X,train_Y,Z2,A2,Z1,A1,Z,A,W1,W2)

        W3 = W3 - dw3*alpha
        B3 = B3 - db3*alpha
        W2 = W2 - dw2*alpha
        B2 = B2 - db2*alpha
        W1 = W1 - dw1*alpha
        B1 = B1 - db1*alpha

        if (i%100 ==0):
            print(f"Iteration:{i}")
            print(get_accuracy(A,Y))
    return W3,B3,W2,B2,W1,B1

W3,B3,W2,B2,W1,B1 = gradient_descent(train_X, train_Y, 0.1,1000)