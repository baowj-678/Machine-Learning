import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt


# 导入数据
wine = pd.read_csv("./datasetswine.csv")

# logistic标准化
data_before = wine[['Alcohol', 'Malic acid']].to_numpy()
data_after = 1 / (1 + np.exp(-data_before))

plt.scatter(data_before[:,0], data_before[:,1])
plt.scatter(data_after[:,0], data_after[:,1])
plt.show()