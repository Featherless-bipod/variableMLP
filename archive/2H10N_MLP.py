import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import functions as fun

def init_parameters(H1,H2,pixels,results):
    W3 = np.random.rand(H1,pixels)-0.5
    B3 = np.random.rand(H1,1)-0.5
    W2 = np.random.rand(H2,H1)-0.5
    B2 = np.random.rand(H2,1)-0.5
    W1 = np.random.rand(results,H2)-0.5
    B1 = np.random.rand(results,1)-0.5
    return W3,B3,W2,B2,W1,B1

def forward_propogation(X,W3,B3,W2,B2,W1,B1):
    Z2 = W3.dot(X)+B3
    A2 = fun.ReLu(Z2)
    Z1 = W2.dot(A2)+B2
    A1 = fun.sig(Z1)
    Z = W1.dot(A1)+B1
    A = fun.softmax(Z)
    return Z2,A2,Z1,A1,Z,A

def back_propogation(X,Y,Z2,A2,Z1,A1,Z,A,W1,W2):
    Y = fun.one_hot(Y)
    
    dz1 = A-Y
    dw1 = 1/m * dz1.dot(A1.T)
    db1 = 1/m * np.sum(dz1)

    dz2 = W1.T.dot(dz1)*fun.derv_sig(Z1)
    dw2 = 1/m*dz2.dot(A2.T) 
    db2 = 1/m*np.sum(dz2)

    dz3 = W2.T.dot(dz2) * fun.derv_ReLu(Z2)
    dw3 = 1/m*dz3.dot(X.T)
    db3 = 1/m*np.sum(dz3)

    return dw1, db1, dw2, db2, dw3, db3

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
            print(fun.get_accuracy(A,Y))
    return W3,B3,W2,B2,W1,B1
