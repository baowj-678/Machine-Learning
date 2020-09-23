""" 决策树
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/9/22
"""
import numpy as np
import math


class DecisionTree():
    def __init__(self, algorithm='ID3', mode='classification'):
        super().__init__()
        """ 决策树初始化
        """
        self.ID3 = 'ID3'
        self.C45 = 'C4.5'
        self.CART = 'CART'
        self.classification = 'classification'
        self.regression = 'regression'

        self.targets = None
        self.algorithm = algorithm
        self.mode = mode
        if self.mode == self.classification:
            self.targets = set()
        else:
            self.mode = self.regression
    
    def load(self, path):
        pass

    def train(self, X):
        """ 训练
        @X numpy(n, dim): 待训练数据
        """
        # 生成结果集
        if self.mode == self.classification:
            for y in X[:, -1]:
                self.targets.add(y)
        
        pass

    def computeEntropy(self, X):
        """ 计算数据的熵
        @X numpy(n, dim): 待计算的数据
        """
        assert self.mode == self.classification
        # C_k每个类别的个数
        C_k = {}
        for y in X[:, -1]:
            if y in C_k:
                C_k[y] += 1
            else:
                C_k[y] = 1
        sum_ = X.shape[0]
        # 计算熵
        entropy = 0
        sum__ = 0 # 待删除
        for value in C_k.values():
            sum__ += value
            value = value / sum_
            entropy -= value* math.log(value)
        assert sum__ == sum_
        return entropy

    def computeGain(self, X):
        """ 计算信息增益
        @X numpy(n, dim): 待计算的数据
        @entropyGain numpy(dim): 每一维的信息增益
        """
        # numpy (n - 1): 每个维度的信息增益
        dim = X.shape[1] - 1
        n = X.shape[0]
        entropyGain = np.zeros(n)
        # 初始entropy
        entropy0 = self.computeEntropy(X)
        # 计算每维的Gain
        for i in range(dim):
            if isinstance(X[0, i], float):
                pass
            else:
                kinds = set()
                for value in X[:, i]:
                    kinds.add(value)
                for value in kinds:
                    X_ = X[X[:, i] == value, :]
                    p = X_.shape[0] / n
                    entropy_ = self.computeEntropy(X_)
                    entropyGain[i] += entropy_* p
                entropyGain[i] -= entropy0
        return entropyGain

    def computeGainRatio(self, X):
        """ 计算增益率
        @X numpy(n, dim)
        """
        pass

    def computeGini(self, X):
        """ 计算基尼系数
        @X numpy(n, dim)
        """

    def predict(self):
        pass

    def save(self):
        pass

    def __call__(self, X):
        """ 预测
        @X numpy(batch_size, dim): 待预测数据
        """
        return self.predict(X)
        pass
