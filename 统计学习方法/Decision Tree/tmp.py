import numpy as np

a = np.array([[1, '', ''],
       [1, '', '']], dtype=object)

print(isinstance(a[0,0], int))