import numpy as np
import matplotlib.pyplot as plt
import math
from describe import init_dataset

#---------------------------------------------------------------------

def     set_data(data_path):
 
    data = init_dataset(data_path)
    features = data[1:, 6:]
    features = features.astype(np.float128)
    targets = data[1:, 1]
    targets[(targets != 'Gryffindor')] = 0. 
    targets[(targets == 'Gryffindor')] = 1.
    targets = targets.astype(np.float128) 
     # np.where(np.isfinite(features), features, 0)
    # features = features[~np.isnan(features)]
    
    # print ("features: ", features)
    # print ("features.len: ", len(features))
    # print ("features.shape: ", features.shape)
    # print ("targets: ", targets)
    # print ("targets.len: ", len(targets))
    # print ("targets.shape: ", targets.shape)
    
    return (features, targets)


def     init_variables():

    weights = np.random.normal(size=13)
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
    # print ("targets: ", targets)
    # print ("weights: ", weights)
    # print ("bias: ", bias)
    # print ("z: ", z)
    # print ("y: ", y)
    # print ("m: ", m)
    # print ("cost: ", cost)

    return (cost)


def     gradient(features, targets, weithts, bias):

    m = len(features)
    z = pre_activation(features, weights, bias)
    y = activation(z)
    tr_features = features.transpose()
    diff = y - targets
    grad = 1 / m * np.nansum(tr_features * diff)

    # print ("features: ", features)
    # print ("m: ", m)
    # print ("targets: ", targets)
    # print ("weights: ", weights)
    # print ("bias: ", bias)
    # print ("z: ", z)
    # print ("y: ", y)
    # print ("gradient: ", grad)

    return (grad)


def     train(features, target, weights, bias):
    
    print ("features: ", features)
    print ("targets: ", targets)
    print ("weights: ", weights)
    print ("bias: ", bias)
    print ("z: ", pre_activation(features, weights, bias))
    print ("y: ", activation(pre_activation(features, weights, bias)))
    print ("cost :", cost(features, targets, weights, bias))
    print ("gradient :", gradient(features, targets, weights, bias))
    print("Accuracy: ", np.mean(np.round(activation(pre_activation(features, weights, bias)) == targets))) 

    return


#---------------------------------------------------------------------

if __name__ == '__main__':
    features, targets = set_data("dataset_train.csv")
    weights, bias = init_variables()
    train(features, targets, weights, bias)