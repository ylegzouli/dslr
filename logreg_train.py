import numpy as np
import pandas as pd
from sys import argv

#---------------------------------------------------------------------

def     scale(features):

    for i in range(len(features)):
        features[i] = ( features[i] - features.mean())  / features.std()

    return features


def     set_data(data_path):

    data = pd.read_csv(data_path, index_col = "Index")
    data = data.dropna()
    features = np.array((data.iloc[:,5:]))
    targets = np.array(data.loc[:,"Hogwarts House"])
    np.apply_along_axis(scale, 0, features)

    return (features, targets)


def     init_variables(features):

    weights = np.ones(features.shape[1])
    
    return (weights)



def     pre_activation(features, weights):

    z = np.dot(features, weights)
    
    return (z)


def     activation(z):

    y = 1.0 / (1.0 + np.exp(-z))
    
    return (y)


def     gradient(features, targets, weights):

    z = pre_activation(features, weights)
    y = activation(z)
    tr_features = features.transpose()
    diff = targets - y
    grad = np.dot(tr_features, diff)

    return (grad)


def     train(features, target, weights):

    epochs = 300
    learning_rate = 0.001
    for epoch in range(epochs):
        if epoch % 10 == 0:
            print (weights[0])
        grad = gradient(features, target, weights)
        weights = weights + learning_rate * grad

    return (weights)


#---------------------------------------------------------------------


def predict_one(x, weights):
    
    return max((x.dot(w), t) for w, t in weights)[1]


def predict(features, weights):

    feature = np.insert(features, 0, 1, axis=1) 

    return [predict_one(i, weights) for i in feature]


def accuracy(data_path, weights):

    data = pd.read_csv(data_path, index_col = "Index")
    features, targets = set_data("dataset_train.csv")

    return sum(predict(features, weights) == targets) / len(targets)   


#---------------------------------------------------------------------

def     main():
    if not len(argv) > 1:
        print('Please input the path as first argument.')
        return
    features, targets = set_data(argv[1])
    features = np.insert(features, 0, 1, axis=1)
    weights_save = []
    for i in np.unique(targets):
        y = np.where(targets == i, 1, 0)    
        weights = init_variables(features)
        weights = train(features, y, weights)
        weights_save.append((weights, i))
    np.save("weights", weights_save)
    print ("Accuracy: ", accuracy("dataset_train.csv", weights_save))


if __name__ == '__main__':
    main()