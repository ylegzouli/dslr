
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
from describe import init_dataset
from sys import argv


#---------------------------------------------------------------------


def     histogram():
    
    dataset = init_dataset('dataset_train.csv')
    data = dataset[1:, :]
    data = data[data[:, 1].argsort()]
    X = np.array(data[:, 16], dtype=float)
    h1 = X[:327]
    h1 = h1[~np.isnan(h1)]
    plt.hist(h1, color='red', alpha=0.5)
    h2 = X[327:856]
    h2 = h2[~np.isnan(h2)]
    plt.hist(h2, color='yellow', alpha=0.5)
    h3 = X[856:1299]
    h3 = h3[~np.isnan(h3)]
    plt.hist(h3, color='blue', alpha=0.5)
    h4 = X[1299:]
    h4 = h4[~np.isnan(h4)]
    plt.hist(h4, color='green', alpha=0.5)
    plt.legend(['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'], loc='upper right', frameon=False)
    plt.title(dataset[0, 16])
    plt.xlabel("Mark")
    plt.ylabel("Nb student")
    plt.show()
    return


def     figure():

    dataset = init_dataset('dataset_train.csv')
    data = dataset[1:, :]
    data = data[data[:, 1].argsort()]
    plt.figure(figsize=(15, 8))
    for i in range(1, 13): 
        X = np.array(data[:, i + 5], dtype=float)
        plt.subplot(4, 4, i)
        h1 = X[:327]
        h1 = h1[~np.isnan(h1)]
        plt.hist(h1, color='red', alpha=0.5)
        h2 = X[327:856]
        h2 = h2[~np.isnan(h2)]
        plt.hist(h2, color='yellow', alpha=0.5)
        h3 = X[856:1299]
        h3 = h3[~np.isnan(h3)]
        plt.hist(h3, color='blue', alpha=0.5)
        h4 = X[1299:]
        h4 = h4[~np.isnan(h4)]
        plt.hist(h4, color='green', alpha=0.5)
        plt.title(dataset[0, i + 5])
        plt.subplots_adjust(hspace = .001)
    plt.show()
    return    


#---------------------------------------------------------------------


if __name__ == "__main__":
    histogram()
    figure()