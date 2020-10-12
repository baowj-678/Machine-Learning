# Copyright (c) 2020, HUST-AI-pi-team
# All rights reserved.
#
# 随机森林实现
#

import numpy as np
import math
import copy
from Decision_Tree import DecisionTree

class RandomForest():
    def __init__(self, num_tree, algorithm='ID3', mode='classification'):
        super().__init__()
        self.mode = mode
        self.classification = 'classification'
        self.regression = 'regression'
        self.RF = [DecisionTree(algorithm, mode, RF=True) for _ in range(num_tree)]
    
    def train(self, X, lb=0, RF_k=None):
        """ 训练
        :param X numpy(n, dim): 待训练数据
        :param lb (float): 收敛条件
        :param RF_k: 随机森林每次选的子属性个数
        :return tree: 生成的决策树
        """
        if RF_k is None:
            RF_k = int(math.log2(X.shape[1]))
        print('='*8 + '开始训练' + '='*8)
        for tree in self.RF:
            tree.train(X, lb=lb, RF_k=RF_k)
        print('='*8 + '训练完成' + '='*8)
        return self.RF
    
    def predict(self, X):
        """ 预测
        """
        if self.mode == self.classification:
            ans = []
            for tree in self.RF:
                ans.append(tree(X))
            ans = np.r_[ans] #(tree_num, n)
            predict = np.zeros([X.shape[0]])
            for i in range(X.shape[0]):
                label_num = {}
                for j in ans[:, i]:
                    if j in label_num:
                        label_num[j] += 1
                    else:
                        label_num[j] = 1
                label = None
                num = 0
                for item in label_num.items():
                    if item[1] > num:
                        label = item[0]
                predict[i] = label
            return predict
        else:
            # 回归问题
            ans = []
            for tree in self.RF:
                ans.append(tree(X))
            ans = np.r_[ans] #(tree_num, n)
            ans = np.average(ans, dim=0)
            assert ans.shape[0] == X.shape[0]
            return ans

    

        