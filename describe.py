import numpy as np
import csv
import math
from sys import argv


def     percentile(N, percent):
    N.sort()
    k = (len(N) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return N[int(k)]
    d0 = N[int(f)] * (c - k)
    d1 = N[int(c)] * (k - f)
    return d0 + d1


def     equart_type(X, mean):
    total = 0
    for x in X:
        total = total + (x - mean) ** 2
    return (total / len(X)) ** 0.5


def     init_dataset(data_path):
    dataset = list()
    with open(data_path) as csvfile:
        reader = csv.reader(csvfile)
        try:
            for _ in reader:
                row = list()
                for value in _:
                    try:
                        value = float(value)
                    except:
                        if not value:
                            value = np.nan
                    row.append(value)
                dataset.append(row)
        except csv.Error as e:
            print("error in dataset")
    return np.array(dataset)


def     describe():
    if len(argv) > 1:
        data = init_dataset(argv[1])
    else:
        print('Please input the data path as first argument.')
        return
    features = data[0]
    dataset = data[1:,0:]
    print(f'{"":29} |{"Count":>19} |{"Mean":>19} |{"Std":>19} |{"Min":>19} |{"25%":>19} |{"50%":>19} |{"75%":>19} |{"Max":>19}')
    for i in range(0, len(features)):
        size = 0
        total = 0
        mini = np.inf
        maxi = - np.inf
        row_sort = []
        try:
            print(f'{features[i]:>29}' , end=' |')
            row = np.array(dataset[:, i], dtype=float)
            row = row[~np.isnan(row)]
            for j in range(0, len(row)):
                size = size + 1
                total = total + row[j]
                row_sort.append(row[j])
                if maxi < row[j]:
                    maxi = row[j]
                if mini > row[j]:
                    mini = row[j]
            if size == 0:
                raise Exception()
            row_sort.sort()
            print(f'{size:>19}', end=' |')
            print(f'{total/size:>19}', end=' |')
            print(f'{equart_type(row, total/size):>19}', end=' |')
            print(f'{mini:>19}', end=' |')
            print(f'{row_sort[math.ceil(size/4) - 1]:>19}', end=' |')
            print(f'{row_sort[math.ceil(size/2) - 1]:>19}', end=' |')
            print(f'{row_sort[math.ceil(3*size/4) - 1]:>19}', end=' |')
            print(f'{maxi:>19}', end=' |')
        except:
            print(f'{"No value":>15}', end='')
        print("")


if __name__ == "__main__":
    describe()