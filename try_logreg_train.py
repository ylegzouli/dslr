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
    return (features, targets)


def     init_variables():
    weights = np.random.normal(size=13)
    weights = weights.astype(np.float128)
    bias = 0
    return (weights, bias)


def     cost(predictions, targets):
    ret = np.nanmean((predictions - targets) ** 2)
    return ret


def     pre_activation(features, weights, bias):
    ret = np.dot(features, weights) + bias
    ret = ret.astype(np.float128)
    return ret


def     activation(z):
    return 1.0 / (1.0 + np.exp(-z))


def     derivative_activation(z):
    return activation(z) * (1.0 - activation(z))


def     predict(features, weights, bias):
    z = pre_activation(features, weights, bias)

    y = activation(z)
    return np.round(y)


def     train(features, target, weights, bias):
    predictions = predict(features, weights, bias)
    predictions = predictions.astype(np.float128) 
    print("Accuracy = %s" % np.mean(predictions == targets))
    
    epochs = 500
    learning_rate = 1
    
    for epoch in range(epochs):
        if epoch % 10 == 0:
            predictions = activation(pre_activation(features, weights, bias))
            print("weights: ", weights)
            print("predictions: ", predictions)
        weights_gradients = np.zeros(weights.shape)
        bias_gradient = 0.
        for feature, target in zip(features, targets):
            if not np.isnan(feature).any():
                z = pre_activation(feature, weights, bias)
                y = activation(z)
                weights_gradients += (y - target) * derivative_activation(z) * feature
                bias_gradient += (y - target) * derivative_activation(z)
                # print("test:  !!!!!!!!!!!!!!", y)
                # print("test:  !!!!!!!!!!!!!!", derivative_activation(z))
        weights = weights - (learning_rate * weights_gradients)
        bias = bias - (learning_rate * bias_gradient)
    predictions = predict(features, weights, bias)
    print("weights: ", weights)
    print("predictions: ", predictions)
    print("Accuracy = %s" % np.mean(predictions == targets))
    return


#---------------------------------------------------------------------

if __name__ == '__main__':
    features, targets = set_data("dataset_train.csv")
    weights, bias = init_variables()
    train(features, targets, weights, bias)