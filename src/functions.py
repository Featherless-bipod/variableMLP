import numpy as np 
import pandas as pd 


def ReLu(Z):
    """
    Rectified Linear Unit (ReLU) activation function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation input matrix.

    Returns
    -------
    numpy.ndarray
        Element-wise maximum of 0 and Z.
    """
    return np.maximum(0,Z)

def softmax(Z): 
    """
    Rectified Linear Unit (ReLU) activation function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation input matrix.

    Returns
    -------
    numpy.ndarray
        Element-wise maximum of 0 and Z.
    """
    return np.exp(Z)/sum(np.exp(Z))

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
    """
    Compute the derivative of the sigmoid activation function.

    Parameters
    ----------
    Z : numpy.ndarray
        Pre-activation values (linear combination of weights and inputs).
        Can be a scalar, vector, or matrix.

    Returns
    -------
    numpy.ndarray
        Derivative of the sigmoid function evaluated at Z.
        Shape matches the input Z.
    """
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