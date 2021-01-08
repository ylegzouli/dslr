import numpy as np
import csv
import math
from describe import init_dataset

#---------------------------------------------------------------------

def     set_data():
    return (features, targets)


def     init_variables():
    return (weights, bias)


def     get_cost(predictions, targets):
    return np.mean((predictions - targets) ** 2)


def     pre_activation():
    return np.dot(features, weights) + bias


def     activation(z):
    return 1 / (1 + np.exp(-z))


def     train(features, target, weigths, bias):
    return


#---------------------------------------------------------------------

if __name__ == '__main__':
    features, targets = get_dataset()
    weights, bias = init_variables()
    train(features, targets, weights, bias)