""" 决策树
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/9/22
"""
import numpy as np
import math
import copy


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

        # 获得叶节点值函数
        self.getLeaf = None
        # “信息”增量计算函数
        self.gainLoss = None
        # 计算"损失"值
        self.getLoss = None
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

    def load(self, path):
        """ 加载模型
        :param path: 文件路径
        """
        with open(path, mode='r') as file:
            data = file.readline()
        self.Tree = self._loadModel(data)
        return self.Tree
    
    def _loadModel(self, data: str):
        """ 根据文本信息生成决策树
        :param data(str): 文本信息
        """
        if data[0] == '{':
            if data[-1] == '}':
                subTree = {}
                #{}形
                data = data[1:len(data)-1]
                while len(data) > 0:
                    if data[0] != '(':
                        print('='*8 + '无法解析' + '='*8)
                        return
                    right_ = 0
                    # 遍历,找')'
                    for i in range(len(data)):
                        if data[i] == ')':
                            right_ = i
                            break
                    # 不存在
                    if right_ == 0:
                        print('='*8 + '无法解析' + '='*8)
                        return
                    # 找{}
                    left__, right__ = 0, 0
                    left_p, right_p = 0, 0
                    for i in range(right_+1, len(data)):
                        if data[i] == '{':
                            left__ += 1
                            if left__ == 1:
                                left_p = i
                        elif data[i] == '}':
                            right__ += 1
                        if left__ == right__ and left__ > 0:
                            right_p = i
                            break
                    if right_p == 0:
                        for i in range(right_+1, len(data)):
                            if data[i] == ':':
                                left_p = i + 1
                                break
                        right_p = len(data) - 1
                    subData = data[left_p:right_p + 1]
                    subTree[data[0:right_ + 1]] = self._loadModel(subData)
                    while right_p < len(data) and data[right_p] != '(':
                        right_p += 1
                    data = data[right_p:]
                return subTree
            else:
                print('='*8 + '无法解析' + '='*8)
                return
        # label
        else:
            return data[1:]

    def train(self, X, lb=0):
        """ 训练
        :param X numpy(n, dim): 待训练数据
        :param lb (float): 收敛条件
        :return tree: 生成的决策树
        """
        self.lb = lb
        # 生成结果集
        if self.mode == self.classification:
            for y in X[:, -1]:
                self.targets.add(y)
        
        # 模型分类
        if self.mode == self.classification:
            # 分类树
            self.getLeaf = self.countLeafLabel
        elif self.mode == self.regression:
            # 回归树
            self.gainLoss = self.computeGain
            self.getLoss = self.computeGiniRegress
            self.getLeaf = self.computeLeafValue
        else:
            raise Exception("模型任务不符合,必须是 'classification' or 'regression' ")

        # 算法分类
        if self.algorithm == self.ID3:
            # ID3算法(信息增益)
            self.gainLoss = self.computeGain
            self.getLoss = self.computeEntropy
        elif self.algorithm == self.C45:
            # C45算法(信息增益率)
            self.gainLoss = self.computeGainRatio
        elif self.algorithm == self.CART:
            # CART算法(基尼系数)
            self.gainLoss = self.computeGain
            self.getLoss = self.computeGiniClassifier
        else:
            raise Exception("模型算法不符合,必须是 'ID3'  'C4.5' or 'CART'")

        self.Tree = self.generateTree(X)
        return self.Tree



    def generateTree(self, X): 
        """ 递归地生成决策树
        :param X numpy(n, dim): 训练数据
        :return (dict): 字典树
        """
        subTree = {}
        # 计算信息增益
        entropyGain, splitPoints = self.gainLoss(X)
        index = np.argmax(entropyGain)
        # 收敛条件1
        if entropyGain[index] <= self.lb:
            leaf_label = self.getLeaf(X)
            return leaf_label
        if isinstance(X[0, index], float):
            # 连续形
            X_left = X[X[:, index] <= splitPoints[index]]
            X_right = X[X[:, index] > splitPoints[index]]
            # 生成子树
            subTree[(index, -1, splitPoints[index])] = self.generateTree(X_left)
            subTree[(index, 1, splitPoints[index])] = self.generateTree(X_right)
        else:
            # 离散形
            # 获取取值集合
            values = set()
            for value in X[:, index]:
                values.add(value)
            # 生成子树
            for value in values:
                subTree[(index, value)] = self.generateTree(X[X[:, index] == value,:])
        return subTree

    def computeEntropy(self, X):
        """ 计算数据的熵
        :param X numpy(n, dim): 训练数据
        :return entropy (float): 熵值
        """
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
        for value in C_k.values():
            value = value / sum_
            entropy -= value * math.log(value)
        return entropy

    def computeGain(self, X):
        """ 计算信息 基尼 Loss增益
        :param X numpy(n, dim): 待计算的数据
        :return entropyGain numpy(dim): 每一维的信息增益
        :return splitPoints numpy(dim): 切分点
        """
        dim = X.shape[1] - 1
        n = X.shape[0]
        # 保存每维的最大信息增益
        lossGain = np.zeros(dim)
        # 保存连续形变量的切分点
        splitPoints = np.zeros(dim)
        # 初始entropy
        loss_0 = self.getLoss(X)
        # 计算每维的Gain
        for i in range(dim):
            if isinstance(X[0, i], float):
                # 连续形
                # 按照第i维排序
                nums = list(set(X[:, i]))
                nums = sorted(nums)
                loss_gain_m_tmp = 0
                split_point_m_tmp = 0
                for j in range(len(nums) - 1):
                    # 切分点
                    split_point = (nums[j] + nums[j + 1])/2
                    # 划分数据
                    X_left = X[X[:, i] <= split_point, :]
                    X_right = X[X[:, i] > split_point, :]
                    # 计算左右的loss
                    loss_left = self.getLoss(X_left)
                    loss_right = self.getLoss(X_right)
                    loss_gain_tmp = loss_left* (j + 1)/n +\
                                       loss_right* (n - j - 1)/n
                    loss_gain_tmp = loss_0 - loss_gain_tmp
                    # 更新切分点
                    if loss_gain_tmp > loss_gain_m_tmp:
                        loss_gain_m_tmp = loss_gain_tmp
                        split_point_m_tmp = split_point
                lossGain[i] = loss_gain_m_tmp
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
                    loss_ = self.getLoss(X_)
                    lossGain[i] += loss_ * p
                lossGain[i] = loss_0 - lossGain[i]
        return (lossGain, splitPoints)

    def computeGainRatio(self, X):
        """ 计算增益率
        :param X numpy(n, dim): 训练数据
        :return entropyGainsRatio numpy(1, dim): 信息增益率
        """
        dim = X.shape[1] - 1
        n = X.shape[0]
        # (1, dim)
        entropyGains, splitPsoints = self.computeGain(X)
        # (1, dim)
        entropyGains_ = np.zeros(dim)
        for i in range(dim):
            # 信息增益为0
            if entropyGains[i] == 0:
                entropyGains_[i] = float('inf')
                continue
            p_1 = np.sum(X[:, i] <= splitPsoints[i]) / n
            p_2 = 1 - p_1
            entropyGains_[i] = -p_1* math.log(p_1) - p_2* math.log(p_2) 
            # 计算GainRatio
        entropyGainsRatio = entropyGains / entropyGains_
        return (entropyGainsRatio, splitPsoints)

    def countLeafLabel(self, X):
        """ 计算叶子节点的分类值
        :param X numpy(n, dim): 数据 
        :return label_final: 出现最多的类别的标签
        """
        labels = {}
        for label in X[:, -1]:
            if label not in labels:
                labels[label] = 1
            else:
                labels[label] += 1
        label_final = None
        label_cnt = 0
        for item in labels.items():
            if item[1] > label_cnt:
                label_final = item[0]
                label_cnt = item[1]
        return label_final
    
    def computeLeafValue(self, X):
        """ 回归任务计算叶节点的值
        :param X numpy(n, dim): 训练数据
        :return value (int): 叶节点的值
        """
        return np.average(X[:, -1])
        
    def computeGiniClassifier(self, X):
        """ 计算分类任务的基尼系数
        :param X numpy(n, dim)
        :return gini (int):基尼系数 
        """
        # C_k每个类别的个数
        C_k = {}
        for y in X[:, -1]:
            if y in C_k:
                C_k[y] += 1
            else:
                C_k[y] = 1
        sum_ = X.shape[0]
        # 计算基尼系数
        gini = 1
        for value in C_k.values():
            value = value / sum_
            gini -= value* value
        return gini

    def computeGiniRegress(self, X):
        """ 计算回归任务的基尼系数
        :param X numpy(n, dim): 训练数据
        :return gini (int): 基尼系数
        """
        x = X[:, -1]
        aver = np.average(x)
        gini = np.sum(x - aver)^2
        return gini

    def pruning(self, X, alpha=0.3):
        """ 
        :param X numpy()
        """
        self.alpha = alpha
        # deep copy
        copy_tree = copy.deepcopy(self.Tree)
        self.prunedTree = self._pruningRecur(X, copy_tree)
        return self.prunedTree

    def _pruningRecur(self, X, root):
        """
        :param X numpy(n, dim): 
        """
        # not leaf node
        if isinstance(root, dict):
            loss_old = 0
            T = 0
            # go through root's sons
            for item in root.items():
                key = item[0]
                subRoot = item[1]
                if len(key) == 2:
                    # 离散形
                    # subdata
                    sub_X = X[X[:, key[0]] == key[1]]
                    loss, t, sub_root= self._pruningRecur(sub_X, subRoot)
                    root[key] = sub_root
                    loss_old += loss
                    T += t
                elif len(key) == 3:
                    # 连续形
                    if key[1] == 1:
                        # subdata
                        sub_X = X[X[:, key[0]] <= key[1]]
                        loss, t, sub_root= self._pruningRecur(sub_X, subRoot)
                    elif key[1] == -1:
                        sub_X = X[X[:, key[0]] > key[1]]
                        loss, t, sub_root= self._pruningRecur(sub_X, subRoot)
                    root[key] = sub_root
                    loss_old += loss
                    T += t
            loss_new = self.getLoss(X) + self.alpha
            if loss_new < loss_old:
                leaf_label = self.getLeaf(X)
                return (loss_new, T, leaf_label)
            else:
                return (loss_old, T, root)
            pass
        # leaf node
        else:
            c_loss = self.getLoss(X)
            return (c_loss, 1, root)

    def predict(self, X):
        """ 预测
        :param X numpy(n, dim): 待预测的数据
        """
        n = X.shape[0]
        labels = []
        for i in range(n):
            labels.append(self.predictSingle(X[i]))
        return labels

    def predictSingle(self, x):
        """ 预测单条数据
        :x numpy(1, dim): 
        """
        root = self.Tree
        while isinstance(root, dict):
            for item in root.items():
                key = item[0]
                subRoot = item[1]
                if len(key) == 2:
                    # 离散形
                    if x[key[0]] == key[1]:
                        root = subRoot
                        break
                elif len(key) == 3:
                    # 连续形
                    if key[1] == 1 and x[key[0]] >= key[2]:
                        root = subRoot
                        break
                    elif key[1] == -1 and x[key[0]] < key[2]:
                        root = subRoot
                        break
        return root

    def save(self, path):
        """ 保存模型
        :param path: 保存路径
        """
        with open(path, mode='w+') as file:
            file.write(str(self.Tree))

    def __call__(self, X):
        """ 预测
        :X numpy(batch_size, dim): 待预测数据
        """
        return self.predict(X)


if __name__ == "__main__":
    print(tuple('(2,3)'))