import pandas as pd
import seaborn as sns
from sys import argv
import matplotlib.pyplot as plt

def pair_plot():
    if not len(argv) > 1:
        print('Please input the data path as first argument.')
        return
    try:
        df = pd.read_csv(argv[1], index_col = "Index")
        sns.pairplot(df, hue="Hogwarts House")
        plt.show()
    except:
        print('File error')

if __name__ == "__main__":
    pair_plot()
