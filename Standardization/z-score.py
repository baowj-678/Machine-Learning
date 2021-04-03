import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt


# 导入数据
wine = pd.read_csv("./datasetswine.csv")

# z-score标准化
minmax_scale = preprocessing.StandardScaler().fit(wine[['Alcohol', 'Malic acid']])
np_zscore = minmax_scale.transform(wine[['Alcohol', 'Malic acid']])

data_before = wine[['Alcohol', 'Malic acid']].to_numpy()
data_after = np_zscore
plt.scatter(data_before[:,0], data_before[:,1])
plt.scatter(data_after[:,0], data_after[:,1])
ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_position(('data',0))
plt.show()