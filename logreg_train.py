import numpy as np
import matplotlib.pyplot as plt
import math
from describe import init_dataset

#---------------------------------------------------------------------

def     set_data(data_path):
    data = init_dataset(data_path)
    features = data[1:, 6:]
    features = features.astype(np.float128)
    # np.where(np.isfinite(features), features, 0)
    # features = features[~np.isnan(features)]
    targets = data[1:, 1]
    targets[(targets != 'Gryffindor')] = 0. 
    targets[(targets == 'Gryffindor')] = 1.
    targets = targets.astype(np.float128) 
    
    # print ("features: ", features)
    # print ("features.len: ", len(features))
    # print ("features.shape: ", features.shape)
    # print ("targets: ", targets)
    # print ("targets.len: ", len(targets))
    # print ("targets.shape: ", targets.shape)
    
    return (features, targets)


def     init_variables():
    weights = np.random.normal(size=13)
    # weights = weights.astype(np.float128)
    bias = 0

    # print ("weights: ", weights)
    # print ("bias: ", bias)
    
    return (weights, bias)



def     pre_activation(features, weights, bias):
    z = np.dot(features, weights) + bias
    z = z.astype(np.float128)
    
    # print ("features: ", features)
    # print ("targets: ", targets)
    # print ("weights: ", weights)
    # print ("bias: ", bias)
    # print ("z: ", z)
    
    return (z)


def     activation(z):
    y = 1.0 / (1.0 + np.exp(-z))
    
    # print ("z: ", z)
    # print ("y: ", y)
    
    return (y)


def     cost(features, targets, weithts, bias):
    m = len(features)
    z = pre_activation(features, weights, bias)
    y = activation(z)
    y[(y == 1)] = 0.999 
    y[(y == 0)] = 0.001 
    cost = (-(1 / m) * np.nansum(targets * np.log(y) + (1 - targets) * np.log(1 - y)))
    
    # print ("features: ", features)
    # print ("___________________________________________________")
    # print ("targets: ", targets)
    # print ("___________________________________________________")
    # print ("weights: ", weights)
    # print ("___________________________________________________")
    # print ("bias: ", bias)
    # print ("___________________________________________________")
    # print ("z: ", z)
    # print ("___________________________________________________")
    # print ("y: ", y)
    # print ("___________________________________________________")
    # print ("m: ", m)
    # print ("___________________________________________________")
    # print ("cost: ", cost)
    # print ("___________________________________________________")

    return (cost)


def     derivative_cost(features, targets, weithts, bias):
    m = len(features)
    z = pre_activation(features, weights, bias)
    y = activation(z)
    grad = 1 / m * np.dot(features.transpose(), (y - targets))
    
    # print ("features: ", features)
    # print ("features.len: ", len(features))
    # print ("targets: ", targets)
    # print ("weights: ", weights)
    # print ("bias: ", bias)
    # print ("z: ", z)
    # print ("y: ", y)
    # print ("grad: ", grad)

    return (grad)


def     train(features, target, weights, bias):
    
    
    # print ("features: ", features)
    # print ("targets: ", targets)
    # print ("weights: ", weights)
    # print ("bias: ", bias)
    # print ("z: ", z)
    print ("y: ", activation(z))
    # print ("m: ", m)
    print ("cost :", cost(features, targets, weights, bias))
    
    return


#---------------------------------------------------------------------

if __name__ == '__main__':
    features, targets = set_data("dataset_train.csv")
    weights, bias = init_variables()
    train(features, targets, weights, bias)
    
    # print (derivative_cost(features, targets, weights, bias))