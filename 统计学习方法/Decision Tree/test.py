import numpy as np
from sklearn.datasets import load_iris
from DecisionTree import *


if __name__ == "__main__":
    data = load_iris()
    x = data['data']
    y = data['target']
    X = np.c_[x, y]
    tree = DecisionTree(mode='regression')
    print(tree.train(X))
    print(tree(x))
    # print(len((8,9)))

# a = np.array([[1, '', ''],
#        [1, '', '']], dtype=object)

# print(isinstance(a[0,0], int))

    def computeGain(self, X):
        """ 计算信息增益
        :param X numpy(n, dim): 待计算的数据
        :return entropyGain numpy(dim): 每一维的信息增益
        :return splitPoints numpy(dim): 切分点
        """
        dim = X.shape[1] - 1
        n = X.shape[0]
        # 保存每维的最大信息增益
        entropyGain = np.zeros(dim)
        # 保存连续形变量的切分点
        splitPoints = np.zeros(dim)
        # 初始entropy
        entropy0 = self.computeEntropy(X)
        # 计算每维的Gain
        for i in range(dim):
            if isinstance(X[0, i], float):
                # 连续形
                # 按照第i维排序
                nums = list(set(X[:, i]))
                nums = sorted(nums)
                entropy_gain_m_tmp = 0
                split_point_m_tmp = 0
                for j in range(len(nums) - 1):
                    # 切分点
                    split_point = (nums[j] + nums[j + 1])/2
                    X_left = X[X[:, i] <= split_point, :]
                    X_right = X[X[:, i] > split_point, :]
                    entropy_left = self.computeEntropy(X_left)
                    entropy_right = self.computeEntropy(X_right)
                    entropy_gain_tmp = entropy_left* (j + 1)/n +\
                                       entropy_right* (n - j - 1)/n
                    entropy_gain_tmp = entropy0 - entropy_gain_tmp
                    # 更新
                    if entropy_gain_tmp > entropy_gain_m_tmp:
                        entropy_gain_m_tmp = entropy_gain_tmp
                        split_point_m_tmp = split_point
                entropyGain[i] = entropy_gain_m_tmp
                splitPoints[i] = split_point_m_tmp
            else:
                # 离散形
                kinds = set()
                for value in X[:, i]:
                    kinds.add(value)
                # 计算每个分类的熵值
                for value in kinds:
                    X_ = X[X[:, i] == value, :]
                    p = X_.shape[0] / n
                    entropy_ = self.computeEntropy(X_)
                    entropyGain[i] += entropy_ * p
                entropyGain[i] = entropy0 - entropyGain[i]
        return (entropyGain, splitPoints)