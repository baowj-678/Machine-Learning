"""
    加载 LibSVM 数据集
@Author: Bao Wenjie
@Date: 2021/2/1
@Email: baowj_678@qq.com
"""
from libsvm.commonutil import svm_read_problem
import numpy as np
from sklearn.datasets import load_svmlight_file

def LoadLibSVM(path:str, x_dtype:np.dtype=np.float, y_dtype:np.dtype=np.int) -> np.ndarray:
    """
        加载 LibSVM 数据集的数据，转成numpy.ndarray类型

    @Param:
    -------
    path: 文件路径\n

    @Return:
    --------
    data_x: [numpy.ndarray](n_samples, n_features)
    data_y: [numpy.ndarray](n_samples,)
    """
    data = load_svmlight_file(path)
    
    data_x = data[0].toarray()
    data_y = (data[1] + 1) / 2  # {-1, 1} -> {0, 1}

    return (data_x.astype(x_dtype), data_y.astype(y_dtype))


def LoadLibSVMData(path:str, features:int, x_dtype:np.dtype=np.float, y_dtype:np.dtype=np.int) -> np.ndarray:
    """
        加载 LibSVM 数据集的数据，转成numpy.ndarray类型

    @Param:
    -------
    path: 文件路径\n
    features: 数据维度

    @Return:
    --------
    data_x: [numpy.ndarray](n_samples, n_features)
    data_y: [numpy.ndarray](n_samples,)
    """
    libsvm_data = None
    try:
        libsvm_data = svm_read_problem(path)
    except IOError:
        raise("文件路径出错")
    libsvm_y, libsvm_x = libsvm_data
    data_y = np.array(libsvm_y, dtype=y_dtype)
    data_x = np.zeros((len(libsvm_x), features), dtype=x_dtype)
    for line, map_ in zip(data_x, libsvm_x):
        for pair in map_.items():
            line[pair[0] - 1] = pair[1]
    return (data_x, data_y)


if __name__ == '__main__':
    a_x, a_y = LoadLibSVMData('dataset\ijcnn1\ijcnn1', 22)
    b_x, b_y = LoadLibSVM('dataset\ijcnn1\ijcnn1')
    print(a_y[990:1000], b_y[990:1000])