''' CBOW
@Author: Bao Wenjie
@Date: 2020/8/4
@Email: bwj_678@qq.com
'''

import os
import torch
import numpy as np


class CBOW:
    def __init__(self, dim, c, window):
        ''' 初始化
        @param dim: 词嵌入向量的维度
        @param c: 词典的大小
        @param window: 窗口的大小
        '''
        super().__init__()
        self.dim = dim
        self.c = c
        self.window = window
        self.__w1 = torch.autograd.Variable(torch.randn(dim, c), requires_grad=True)
        self.__w2 = torch.autograd.Variable(torch.randn(c, dim), requires_grad=True)
    
    def train(self,
              center_word,
              samples,
              lr=0.01,
              epoch=5):
        ''' 训练函数
        @param center_word: 中性词的向量
        @param samples [[int]]: 训练样本向量列表
        @param lr: learning_rate
        @param epoch: 训练次数
        '''
        sample_size = len(samples) #训练集大小
        loss = []
        for i in range(epoch):
            for sample in samples:
                '''词嵌入'''
                # 对sos编码
                v_c = torch.zeros(self.dim)
                for j in sample:
                    v_c += self.__w1[:, j]
                v_c = v_c / (2 * self.window)

                x_c = self.__w2 @ v_c
                x_c = torch.exp(x_c) / torch.exp(x_c).sim()
                L = -torch.log(x_c[center_word])
                loss.append(L.item())
                print("Cross Entropy Error: %.4f" % L)

                L.backward()
                self.__w2.data = self.__w2.data - lr * self.__w2.grad.data
                self.__w1.data = self.__w1.data - lr * self.__w1.grad.data
                self.__w2.grad.data.zeros()
                self.__w1.grad.data.zeros()
        return v_c.detach(), loss

class data_helper:
    def __init__(self):
        super().__init__()