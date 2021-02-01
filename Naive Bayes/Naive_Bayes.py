""" 朴素贝叶斯
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/9/27
"""
import numpy as np


class NaiveBayes():
    def __init__(self, dim):
        """ 初始化
        :param dim (int): 数据特征数量
        """
        self.dim = dim
        self.label_map = dict()
        self.attri_map = list()
        self.prop = list()
    
    def train(self, X, Y):
        """ 训练
        """
        self.dim = X.shape[1]
        n = X.shape[0]
        # 统计离散变量取值
        self.__set_attributes(X, Y)
        # 初始化
        for i in range(self.dim):
            self.prop.append(np.ones(len(self.label_map), len(self.attri_map[i])))
        # 计算后验概率
        for i in range(n):
            label = X[i, -1]
            index = self.label_map[label]
            for j in range(dim):
                self.prop[j][index, self.attri_map[X[i, j]]] += 1
            self.prop[j] = self.prop[j]/np.sum(self.prop[j])
        return self.prop

    def set_attributes(self, attributes):
        """ 设置各个特征的属性取值
        :attributes list(list(str)): 各个属性取值
        """
        for attributes_dim in attributes:
            self.attri_map.append(dict())
            tmp_dict = self.attri_map[-1]
            k = 0
            for attribute in attributes_dim:
                if attribute not in tmp_dict:
                    tmp_dict[attribute] = k
                    k += 1
        return self.attri_map
    
    def __set_attributes(self, X, Y):
        """ 重训练数据得到属性取值
        :X numpy(n, dim): 训练数据
        :Y numpy(n): labels
        """
        # 统计label
        k = 0
        for label in Y:
            if label not in self.label_map:
                self.label_map[label] = k
                k += 1
        # 统计属性
        for i in range(self.dim):
            tmp_map = dict()
            k = 0
            for x in X[:, i]:
                if x not in tmp_map:
                    tmp_map[x] = k
                    k += 1
            self.attri_map.append(tmp_map)
        return (self.attri_map, self.label_map)

    def __call__(self, X):
        """ 预测
        """
        return self.predict(X)

    def predict(self, X):
        """ 预测
        """
        ans = np.ones(len(self.label_map))
        for i in range(self.dim):
            ans = ans* self.prop[i][:, self.attri_map[X[i]]]
        return self.label_map[np.argmax(ans)]

    def save(self, path):
        """ 模型保存
        """
        pass

    def load(self, path):
        """ 模型加载
        """
        pass

    def __getitem__(self, X):
        """ 预测单个数据
        """
        pass
    
