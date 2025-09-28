#HouseKeeping
import numpy as np 
import pandas as pd 

data = pd.read_csv('TMNIST_Data.csv')

data = data.drop("names",axis=1)
data.head()

#BASIC STUFF

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

train_y.size

def one_hot(Y):
    """
    Convert class labels to one-hot encoded vectors.

    Parameters
    ----------
    Y : numpy.ndarray
        Class labels (shape [batch_size,]).

    Returns
    -------
    numpy.ndarray
        One-hot encoded matrix (shape [num_classes, batch_size]).
    """
    Y_one_hot = np.zeros((Y.size, Y.max()+1)) #LEARN NEED (())
    Y_one_hot[np.arange(Y.size),Y] = 1    #LEARN
    return Y_one_hot.T

def derv_sigmoid(Z):
    return np.exp(-Z)/(1+np.exp(-Z))**2
def derv_ReLu(Z): 
    """
    Derivative of the ReLU function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation input matrix.

    Returns
    -------
    numpy.ndarray
        Binary mask where 1 indicates Z > 0, else 0.
    """
    return Z > 0

def back_propogation(OUT, train_y, Z, A, params):
    """
    Compute gradients for all parameters using backpropagation.

    Parameters
    ----------
    OUT : numpy.ndarray
        Output probabilities from forward pass.
    train_y : numpy.ndarray
        Ground truth labels (shape [batch_size,]).
    Z : list of numpy.ndarray
        Pre-activations from forward pass.
    A : list of numpy.ndarray
        Activations from forward pass (including input).
    params : list of lists
        Current network parameters.

    Returns
    -------
    list of lists
        Gradients for each layer [dW, db], ordered from output to input.
    """
    Y = one_hot(train_y)
    m = train_y.size

    gradient = []

    dz = OUT - Y
    dw = 1/m * np.dot(dz, A[-2].T) 
    db = 1/m * np.sum(dz, axis=1, keepdims=True)
    gradient.append([dw, db])
    
    for i in range(1, len(params)):
        dw = 1/m * np.dot(dz, A[-i-1].T) 
        db = 1/m * np.sum(dz, axis=1, keepdims=True)
        gradient.append([dw, db])
    
    return gradient[::-1]  # Reverse to match params order

def get_accuracy(A,Y):
    """
    Calculate classification accuracy.

    Parameters
    ----------
    A : numpy.ndarray
        Predicted probabilities (shape [num_classes, batch_size]).
    Y : numpy.ndarray
        True labels (shape [batch_size,]).

    Returns
    -------
    float
        Accuracy between 0 and 1.
    """
    return sum(np.argmax(A,0) == Y)/Y.size #LEARN ARGMAX

def gradient_descent(H, N, X, Y, pixels, results, iterations, alpha):
    """
    Train the neural network using gradient descent.

    Parameters
    ----------
    H : int
        Number of hidden layers.
    N : int
        Neurons per hidden layer.
    X : numpy.ndarray
        Training data (shape [input_dim, batch_size]).
    Y : numpy.ndarray
        Training labels (shape [batch_size,]).
    pixels : int
        Input dimension.
    results : int
        Output dimension (number of classes).
    iterations : int
        Number of training iterations.
    alpha : float
        Learning rate.

    Returns
    -------
    list of lists
        Trained parameters for each layer.
    """
    params = init_parameters(H, N, pixels, results)
    for i in range(iterations):
        OUT, Z, A = forward_propogation(X, params)
        gradient = back_propogation(OUT, Y, Z, A, params)
        
        # Update parameters
        for j in range(len(params)):
            params[j][0] -= alpha * gradient[-j][0]
            params[j][1] -= alpha * gradient[-j][1]
        
        if i % 100 == 0:
            alpha = alpha
            print(f"Iteration:{i}")
            print(get_accuracy(A[-1], Y))

    return params

params = gradient_descent(3,10,train_x,train_y,pixels,results,1000,0.1)


