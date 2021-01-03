
from describe import init_dataset
from sys import argv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import math

#---------------------------------------------------------------------

def histogram():
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
    plt.legend(['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'], loc='upper right', frameon=False)
    plt.title(dataset[0, 16])
    plt.xlabel("Mark")
    plt.ylabel("Nb student")
    plt.show()
    return

def figure():
    dataset = init_dataset('dataset_train.csv')
    data = dataset[1:, :]
    data = data[data[:, 1].argsort()]

    plt.figure(figsize=(15, 8))
    
    X = np.array(data[:, 6], dtype=float)
    plt.subplot(4, 4, 1)
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
    plt.title(dataset[0, 6])
    # ...............................
    
    X = np.array(data[:, 7], dtype=float)
    plt.subplot(4, 4, 2)
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
    plt.title(dataset[0, 7])
    # ...............................
    
    X = np.array(data[:, 8], dtype=float)
    plt.subplot(4, 4, 3)
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
    plt.title(dataset[0, 8])
    # ...............................
    
    X = np.array(data[:, 9], dtype=float)
    plt.subplot(4, 4, 4)
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
    plt.title(dataset[0, 9])
    # ...............................

    X = np.array(data[:, 10], dtype=float)
    plt.subplot(4, 4, 5)
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
    plt.title(dataset[0, 10])
    # ...............................

    X = np.array(data[:, 11], dtype=float)
    plt.subplot(4, 4, 6)
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
    plt.title(dataset[0, 11])
    # ...............................

    X = np.array(data[:, 12], dtype=float)
    plt.subplot(4, 4, 7)
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
    plt.title(dataset[0, 12])
    # ...............................

    X = np.array(data[:, 13], dtype=float)
    plt.subplot(4, 4, 8)
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
    plt.title(dataset[0, 13])
    # ...............................

    X = np.array(data[:, 14], dtype=float)
    plt.subplot(4, 4, 9)
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
    plt.title(dataset[0, 14])
    # ...............................

    X = np.array(data[:, 15], dtype=float)
    plt.subplot(4, 4, 10)
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
    plt.title(dataset[0, 15])
    # ...............................

    X = np.array(data[:, 16], dtype=float)
    plt.subplot(4, 4, 11)
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
    plt.title(dataset[0, 16])
    # ...............................

    X = np.array(data[:, 17], dtype=float)
    plt.subplot(4, 4, 12)
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
    plt.title(dataset[0, 17])
    # ...............................

    X = np.array(data[:, 18], dtype=float)
    plt.subplot(4, 4, 13)
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
    plt.title(dataset[0, 18])
    # ...............................
    
    plt.show()
    return    

#---------------------------------------------------------------------

if __name__ == "__main__":
    histogram()
    figure()