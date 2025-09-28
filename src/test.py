import nHnN_MLP as nMLP
import functions as fun
import time
import matplotlib
import numpy as np

def train_and_evaluate(H, N, train_x, train_y, test_x, test_y, pixels, results, iters, alpha):
    start = time.time()
    params,training_accuracy, epoch_time = nMLP.gradient_descent(H, N, train_x, train_y, pixels, results, iters, alpha)
    total_duration = time.time() - start
    OUT_test, _, _ = nMLP.forward_propogation(test_x, params)
    preds = np.argmax(OUT_test, axis=0)
    test_acc = np.mean(preds == test_y)
    
    return test_acc, total_duration, training_accuracy, epoch_time