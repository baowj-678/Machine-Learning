from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np



def calcEntropy(data_y):
    set_y = set(data_y)
    entropy = 0
    for i in set_y:
        proportion = np.sum(data_y == i) / data_y.shape[0]
        entropy -= proportion * np.log2(proportion)
    return entropy

def getBestPoint(data_x, data_y):#data_x in some dimesion,x y is a row vector
    set_x = np.sort(np.array(list(set(data_x))))
    point = None
    min_entropy = 10000
    # print(set_x)
    for i in range(set_x.shape[0] - 1):
        # print(i,'\n\n\n\n\n')
        now_point = (set_x[i] + set_x[i + 1])/2
        p_left = np.sum(data_x < now_point) / data_x.shape[0]
        left_entropy = calcEntropy(data_y[data_x < now_point])
        # print(p_left, left_entropy)
        p_right = np.sum(data_x > now_point) / data_x.shape[0]
        right_entropy = calcEntropy(data_y[data_x > now_point])
        # print(p_right, right_entropy)
        now_entropy = p_left*left_entropy + p_right*right_entropy
        if min_entropy > now_entropy:
            min_entropy = now_entropy
            point = now_point
    return (point, min_entropy)

def getMajorLabel(data_y):
    set_y = set(data_y)
    majority = 0
    majorLabel = None
    for label in set_y:
        now_count = np.sum(data_y == label)
        if now_count > majority:
            majority = now_count
            majorLabel = label
    return majorLabel

def decisionTree(data_x, data_y, dimension=0):
    entropy = calcEntropy(data_y)
    if data_x.size == 0:
        return getMajorLabel(data_y)
    if len(set(data_y)) == 1:
        return data_y[0]
    point, min_entropy = getBestPoint(data_x[0, :], data_y)
    Gain = entropy - min_entropy
    # print(point)
    # print(data_x, data_y)
    left_x = data_x[:, data_x[0, :] < point]
    left_x = np.delete(left_x, 0, axis=0)
    left_y = data_y[data_x[0, :] < point]
    right_x = data_x[:, data_x[0, :] > point]
    right_x = np.delete(right_x, 0, axis=0)
    right_y = data_y[data_x[0, :] > point]
    Tree = {}
    Tree['>' + str(point)] = decisionTree(right_x, right_y, dimension + 1)
    Tree['<' + str(point)] = decisionTree(left_x, left_y, dimension + 1)
    return Tree

def randomForest(data_x, data_y, numOftree):
    Forest = []
    numsOfdata = data_y.shape[0]
    for i in range(numOftree):
        index = np.random.choice(numsOfdata, numsOfdata, replace=True)
        Forest.append(decisionTree(data_x[:, index], data_y[index], 0))
    return Forest

def testTreeFx(Tree, x, dimension=0):
    if type(Tree) is dict:
        for i in Tree.keys():
            # print(i)
            if eval(str(x[dimension]) +i):
                return testTreeFx(Tree[i], x, dimension + 1)
    else:
        return Tree

def calcLoss(forest, x, y):
    all_sum = y.shape[0]
    print(y.shape)
    correct = 0
    for i in range(y.shape[0]):
        positive = 0
        for j in forest:
            fx = testTreeFx(j, x[:, i])
            if fx == y[i]:
                positive += 1
        if positive / len(forest) > 0.5:
            correct += 1
    return float(correct) / all_sum

def main():
    data = load_iris()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data.data, data.target, test_size=0.5)
    Xtrain = np.transpose(np.array(Xtrain))
    Xtest = np.transpose(np.array(Xtest))
    Ytrain = np.array(Ytrain)
    Ytest = np.array(Ytest)
    print(Xtrain.shape, Ytrain.shape)
    Forest = randomForest(Xtrain, Ytrain, 1000)
    print(calcLoss(Forest, Xtest, Ytest))
    # tree = decisionTree(data_x, data_y)
    # print(Forest)

main()