#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
data = 'dataset_train.csv'
data_test = 'dataset_test.csv'


dataset = pd.read_csv(data, index_col = "Index")

#%%

display(dataset)

# %%
plt.plot(dataset['Astronomy'])


# %%
plt.plot(-dataset['Defense Against the Dark Arts'])

# %%

# %%
