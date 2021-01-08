import numpy as np
import csv
import math
from describe import init_dataset

#---------------------------------------------------------------------

def     set_data(data_path):
    data = init_dataset(data_path)
    
    features = data[0, :] 
    targets = data[1:, 1]
    
    print ("features ----> ", features)
    print ("targets ----> ", targets)
    print (len(targets))
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
    features, targets = set_data("dataset_train.csv")
    # weights, bias = init_variables()
    # train(features, targets, weights, bias)