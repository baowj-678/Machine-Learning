""" 感知机的NumPy实现
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/9/20
"""
import numpy as np


class Preceptron:
    def __init__(self, dim, lr=1e-3):
        """ 初始化
        @dim: 输入数据的维度
        """
        self.lr = lr
        if dim == None:
            self.dim = None
            self.weight = None
            self.bias = None
        else:
            self.dim = dim
            self.weight = np.random.normal(0, 1, size=(1, self.dim))
            self.bias = np.random.normal(0, 0.1, size=(self.dim, 1))
        
    def __call__(self, X, Y):
        """ 调用train方法
        """
        return self.train(X, Y)
    
    def train(self, X, Y):
        """ 训练函数
        @X ndarray(batch_size, dim): 训练数据
        @Y ndarray(1, batch_size): 训练结果
        """
        X = np.transpose(X)
        y = np.matmul(self.weight, X) #(1, batch_size)
        if Y is None:
            y[y <= 0] = -1
            y[y > 0] = 1
            return y
        loss = y * Y
        loss[loss < 0] = 0 #(1, batch_size)
        loss_sum = np.sum(loss)
        self.backward(loss, X)
        return loss_sum

    def backward(self, dy, X):
        """ 反向传播
        @dy ndarray(1, batch_size): 各个用例的dy
        @X ndarray(dim, batch_size): 用例
        """
        dx = np.dot(dy, np.transpose(X)) #(1, dim)
        self.weight -= self.lr * dx
        return dx

    def save(self, path):
        """ 保存模型
        @path (str): 模型保存路径
        """
        np.save(path + 'weight.np', self.weight)
        np.save(path + 'bias.np', self.bias)

    def load(self, path):
        """ 加载模型
        @path (str): 模型加载路径 
        """
        self.weight = np.load(path + 'weight.np')
        self.bias = np.load(path + 'bias.np')
        self.dim = self.weight.shape[1]


if __name__ == "__main__":
    from sklearn import datasets, utils

    iris = datasets.load_iris()
    X = np.array(iris['data'])
    Y = np.array(iris['target'])
    X = X[(Y == 1) | (Y == 0)]
    Y = Y[(Y == 1) | (Y == 0)]
    Y[Y == 0] = -1
    X, Y = utils.shuffle(X, Y)
    epochs = 200
    model = Preceptron(4, lr=0.001)
    print(model.weight)
    BATCH = 4
    print('='*8 + 'begin train' + '='*8)
    for epoch in range(epochs):
        itors = len(X) // BATCH
        for itor in range(itors):
            X_feed = X[itor* BATCH: (itor + 1)* BATCH, ]
            Y_feed = Y[itor* BATCH: (itor + 1)* BATCH]
            loss = model(X_feed, Y_feed)
            print('epoch:{}, itor:{}, loss:{}'.format(epoch, itor, loss))
    print('='*8 + 'begin test' + '='*8)
    y = model(X, None)
    print('accuracy:{}'.format(np.sum(Y == y) / Y.shape[0]))
    print(model.weight)
