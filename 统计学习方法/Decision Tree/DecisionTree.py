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

        # 决策树
        self.Tree = None
        # 预测目标集合
        self.targets = None
        # 算法类型ID3\C45\CART
        self.algorithm = algorithm
        # 模型类型 classification\regression
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
        if self.algorithm == self.classification:
            self.Tree = self.generateTreeClassifier(X)
        elif self.algorithm == self.regression:
            self.Tree = self.generateTreeRegressier(X)
        else:
            raise Exception("模型任务不符合,必须是 'classification' or 'regression' ")
        return self.Tree

    def generateTreeClassifier(self, X):
        """ 递归生成分类决策树
        :param X: 数据
        :return: {}字典
        """
        subTree = {}
        if self.algorithm == self.ID3:
            # ID3算法(信息增益)
            return self.generateTreeID3(X)
        elif self.algorithm == self.C45:
            # C45算法(信息增益率)
            pass
        elif self.algorithm == self.CART:
            # CART算法(基尼系数)
            pass
        else:
            raise Exception("模型算法不符合,必须是 'ID3' or 'C45' or 'CART'")

    def generateTreeID3(self, X):
        """ ID3算法
        :param X: 数据
        :return: {}字典树
        """
        subTree = {}
        dim = X.shape[1] - 1
        # 计算信息增益
        t, entropyGain = self.computeGain(X)
        if t is None:
            # 离散形
            index = np.argmax(entropyGain)
            items = set()
            # 变量取值集合
            for item in X[:, index]:
                items.add(item)
            for item in items:
                subTree[item] = self.generateTreeID3(X[X[:, index] == item, :])
        else:
            # 连续性
            index = np.argmax(entropyGain)

        return subTree

    def generateTreeRegressier(self, X):
        """ 递归生成回归决策树
        :param X: 数据
        :return: {}字典
        """
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
        sum__ = 0  # 待删除
        for value in C_k.values():
            sum__ += value
            value = value / sum_
            entropy -= value * math.log(value)
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
                    entropyGain[i] += entropy_ * p
                entropyGain[i] -= entropy0
        return (1, entropyGain)

    def computeGainRatio(self, X):
        """ 计算增益率
        @X numpy(n, dim)
        """
        dim = X.shape[1] - 1
        n = X.shape[0]
        # (1, dim)
        entropyGains = self.computeEntropy(X)
        # (1, dim)
        entropyGains_ = np.zeros(dim)
        for i in range(dim):
            entropyGains_[i] = self.computeEntropy(X[:, i])

        # 计算GainRatio
        entropyGainsRatio = entropyGains / entropyGains_
        return entropyGainsRatio

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
