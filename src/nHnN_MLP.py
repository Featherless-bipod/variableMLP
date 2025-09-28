import numpy as np 
import pandas as pd 
import functions as fun
import time


def process_data(route, testAmount, scale,results,rem_axis = None):
    data = pd.read_csv(route)
    if rem_axis is not None:
        data = data.drop(rem_axis,axis = 1)
    data = np.array(data)
    np.random.shuffle(data)

    test_data = data[0:testAmount].T
    test_y = test_data[0]
    test_x = test_data[1:]/scale

    train_data = data[testAmount:].T
    train_y = train_data[0]
    train_x = train_data[1:]/scale

    pixels, train_trials = train_x.shape
    
    return test_y,test_x,train_y,train_x, pixels,train_trials,results


def init_parameters(H, N, pixels, results):
    """
    Initialize weights and biases for a neural network with H hidden layers.

    Parameters
    ----------
    H : int
        Number of hidden layers.
    N : int
        Number of neurons in each hidden layer.
    pixels : int
        Input dimension (e.g., flattened image pixels).
    results : int
        Output dimension (number of classes).

    Returns
    -------
    list of lists
        A list where each element is a layer's parameters [W, b]:
        - W: Weight matrix (shape [neurons_current, neurons_previous]).
        - b: Bias vector (shape [neurons_current, 1]).
    """
    params = []
    params.append([
        np.random.rand(N, pixels) - 0.5,  
        np.random.rand(N, 1) - 0.5       
    ])
    for _ in range(H-1):
        params.append([
            np.random.rand(N, N) - 0.5,   
            np.random.rand(N, 1) - 0.5    
        ])
    params.append([
        np.random.rand(results, N) - 0.5,  
        np.random.rand(results, 1) - 0.5   
    ])
    return params

def forward_propogation(IN, params):
    """
    Perform forward propagation through the network.

    Parameters
    ----------
    IN : numpy.ndarray
        Input data (shape [input_dim, batch_size]).
    params : list of lists
        Network parameters (weights and biases per layer).

    Returns
    -------
    tuple
        - OUT: Output probabilities (shape [output_dim, batch_size]).
        - Zs: List of pre-activations for each layer.
        - As: List of activations (including input layer).
    """
    Zs, As = [], [IN]  # As starts with input X
    
    for W, b in params[:-1]:  # Process all but last layer
        Z = np.dot(W, As[-1]) + b
        A = fun.ReLu(Z)  # or sigmoid for first layer
        Zs.append(Z)
        As.append(A)
    
    # Output layer (softmax)
    W_out, b_out = params[-1]
    Z_out = np.dot(W_out, As[-1]) + b_out
    OUT = fun.softmax(Z_out)
    
    return OUT, Zs, As
        
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
    Y = fun.one_hot(train_y)
    m = train_y.size

    gradient = []

    dz = OUT - Y
    dw = (1/m) * np.dot(dz, A[-2].T)    # last hidden activation
    db = (1/m) * np.sum(dz, axis=1, keepdims=True)
    gradient.append([dw, db])

    for i in range(len(params)-2, -1, -1):   # from last hidden down to first hidden
        W_next = params[i+1][0]              # weights of the layer after current
        dz = np.dot(W_next.T, dz) * fun.derv_ReLu(Z[i])
        dw = (1/m) * np.dot(dz, A[i].T)      # A[i] is activation of previous layer
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        gradient.insert(0, [dw, db])         # insert at front to keep order

    return gradient 
     


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
    accuracy = []
    epoch_time = []
    for i in range(iterations):
        start_time = time.time()
        OUT, Z, A = forward_propogation(X, params)
        gradient = back_propogation(OUT, Y, Z, A, params)
        
        for j in range(len(params)):
            params[j][0] -= alpha * gradient[j][0]
            params[j][1] -= alpha * gradient[j][1]
        
        if i % 100 == 0:
            alpha = alpha
            print(f"Iteration:{i}")
            print(fun.get_accuracy(OUT, Y))
            accuracy.append(fun.get_accuracy(OUT, Y))
            epoch_time.append(time.time()-start_time)


    return params,accuracy,epoch_time