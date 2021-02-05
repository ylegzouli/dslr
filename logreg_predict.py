import numpy as np
import pandas as pd
from sys import argv
import math
from describe import init_dataset

#---------------------------------------------------------------------

def     scale(features):

    for i in range(len(features)):
        features[i] = ( features[i] - features.mean())  / features.std()

    return features


def     get_mean():
    
    data = init_dataset(argv[1])
    mean = []
    features = data[0]
    dataset = data[1:,0:]
    for i in range(6, len(features)):
        size = 0
        total = 0
        try:
            row = np.array(dataset[:, i], dtype=float)
            row = row[~np.isnan(row)]
            for j in range(0, len(row)):
                size = size + 1
                total = total + row[j]
            if size == 0:
                raise Exception()
            mean.append(total/size)
        except:
            print ("error data")
    
    return (mean)
        

def     replace_nan(features):

    mean = get_mean()
    for j in range(features.shape[1]):
        for i in range(features[:, j].shape[0]):
            if math.isnan(features[i, j]):
                features[i, j] = mean[j]

    return features


def     set_data():

    data = pd.read_csv(argv[1], index_col = "Index")
    features = np.array((data.iloc[:,5:]))
    features = replace_nan(features)
    np.apply_along_axis(scale, 0, features)
    weights = np.load(argv[2], allow_pickle=True)
    targets = np.array(data.loc[:,"Hogwarts House"])

    return (features, weights, targets)


#---------------------------------------------------------------------


def     predict_line(x, weights):
    
    return max((x.dot(w), t) for w, t in weights)[1]


def     predict(features, weights):

    feature = np.insert(features, 0, 1, axis=1) 

    return [predict_line(i, weights) for i in feature]



#---------------------------------------------------------------------


def     main():
    if not len(argv) > 2:
        print('Please input the data path as first argument and the weights file as second arguments.')
        return
    features, weights, test_target = set_data()
    targets = predict(features, weights)
    print (targets) 
    houses = pd.DataFrame(({'Index':range(len(targets)), 'Hogwarts House':targets}))
    houses.to_csv('houses.csv', index=False) 
    # for i in range(len(targets)):
    #     print (targets[i])
    #     print (test_target[i])
    #     if not test_target[i] == targets[i]:
    #         print ("-----> ERROR HERE !")



if __name__ == '__main__':
    main()