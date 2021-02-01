# Copyright (c) 2020, HUST-AI-pi-team
# All rights reserved.
#
# 决策树/随机森林 运行文件
#

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


from Decision_Tree import DecisionTree
from Random_Forest import RandomForest


if __name__ == "__main__":
    # 加载数据
    data = load_iris()
    x = data['data']
    y = data['target']
    # 分割数据
    X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40, shuffle=True)
    train_data = np.c_[X_train, y_train]

    ################################# 决策树运行 ###################################
    # 加载决策树模型
    tree = DecisionTree(mode='classification')
    # 训练
    tree.train(train_data)
    # 剪枝
    # tree.pruning(train_data, 0.03)
    # 预测
    y_pre = tree(X_test)


    ################################# 随机森林运行 ###################################
    # # 加载随机森林模型
    # RF = RandomForest(num_tree=10)
    # # 训练随机森林
    # RF.train(train_data, RF_k=3)
    # # 预测
    # y_pre = RF.predict(X_test)

    # 准确率
    print(np.sum(y_pre == y_test)/y_test.shape[0])
