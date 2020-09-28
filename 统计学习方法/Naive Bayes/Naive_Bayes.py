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
    
    def train(self, X, Y):
        """ 训练
        """
        pass

    def __call__(self, X):
        """ 预测
        """
        return self.predict(X)

    def predict(self, X):
        """ 预测
        """
        pass

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
    
