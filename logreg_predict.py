import numpy as np
import pandas as pd
from sys import argv

def predict_one(x, weights):
    
    return max((x.dot(w), t) for w, t in weights)[1]


def predict(features, weights):

    feature = np.insert(features, 0, 1, axis=1) 

    return [predict_one(i, weights) for i in feature]

def     main():
    if not len(argv) > 2:
        print('Please input the data path as first argument and the weights file as second arguments.')
        return

#---------------------------------------------------------------------

if __name__ == '__main__':
    main()