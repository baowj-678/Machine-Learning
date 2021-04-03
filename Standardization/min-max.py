import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt


# 导入数据
wine = pd.read_csv("./datasetswine.csv")

# 0-1标准化
minmax_scale = preprocessing.MinMaxScaler().fit(wine[['Alcohol', 'Malic acid']])
np_minmax = minmax_scale.transform(wine[['Alcohol', 'Malic acid']])

data_before = wine[['Alcohol', 'Malic acid']].to_numpy()
data_after = np_minmax
plt.scatter(data_before[:,0], data_before[:,1])
plt.scatter(data_after[:,0], data_after[:,1])
plt.show()