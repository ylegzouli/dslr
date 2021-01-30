import numpy as np
import pandas as pd
from sys import argv


def     scale(features):

    for i in range(len(features)):
        features[i] = ( features[i] - features.mean())  / features.std()

    return features


def     set_data(data_path):

    data = pd.read_csv(data_path, index_col = "Index")
    print (data)
    data.drop(["Hogwarts House"], axis=1) 
    # data = data.dropna()
    print (data)

    features = np.array((data.iloc[:,5:]))
    # targets = np.array(data.loc[:,"Hogwarts House"])
    np.apply_along_axis(scale, 0, features)
    print (features)

    col_mean = np.nanmean(features, axis=0)
    print(col_mean)
    inds = np.where(np.isnan(features))
    features[inds] = np.take(col_mean, inds[1])
    
    print (features)
    targets = 0

    return (features, targets)


def predict_one(x, weights):
    
    return max((x.dot(w), t) for w, t in weights)[1]


def predict(features, weights):

    feature = np.insert(features, 0, 1, axis=1) 

    return [predict_one(i, weights) for i in feature]


#---------------------------------------------------------------------


def     main():
    if not len(argv) > 2:
        print('Please input the data path as first argument and the weights file as second arguments.')
        return
    print (argv[1])
    features, weights = set_data(argv[1])
    print (features, weights)

if __name__ == '__main__':
    main()