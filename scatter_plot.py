import numpy as np
import pandas as pd
from sys import argv
import matplotlib.pyplot as plt

def     main():
    if not len(argv) > 1:
        print('Please input the data path as first argument.')
        return
    dataset = pd.read_csv(argv[1], index_col = "Index")
    dataset = dataset[["Astronomy", "Defense Against the Dark Arts"]]
    dataset = dataset.dropna()
   
    plt.figure()
    plt.scatter(dataset['Astronomy'], dataset['Defense Against the Dark Arts'], label = 'students')
    plt.legend()
    plt.title("features identiques")
    plt.xlabel("Astronomy")
    plt.ylabel("Defense Against the Dark Arts")
    plt.show()

if __name__ == '__main__':
    main()