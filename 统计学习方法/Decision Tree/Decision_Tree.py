import numpy as np
from math import log

#计算Entropy
def clacEntropy(dataSets):
    TargetList = [example[-1] for example in dataSets]
    TargetTypical = set(TargetList)
    Entropy = 0
    for Target in TargetTypical:
        proportion = float(TargetList.count(Target)) / len(TargetList)
        Entropy -= proportion * log(proportion, 2)
    return Entropy

#计算Gini系数
def clacGini(dataSets):
    TargetList = [example[-1] for example in dataSets]
    TargetTypical = set(TargetList)
    Gini = 1
    for Target in TargetTypical:
        proportion = float(TargetList.count(Target)) / len(TargetList)
        Gini -= proportion ** 2
    return Gini

#返回最佳分类属性，基于信息增益
def bestFeature_InfoGain(dataSets, labels):
    BaseEnt = clacEntropy(dataSets)
    BestInfoGain = 0
    BestFeature = None
    for i in range(len(labels)):
        ClassList = np.array([example[i] for example in dataSets])
        ClassTypical = np.unique(ClassList)
        NewEnt = 0
        NewInfoGain = 0
        for branch in ClassTypical:
            proportion = np.sum(ClassList == branch) / len(ClassList)
            NewEnt += proportion * clacEntropy(dataSets[ClassList == branch])
            NewInfoGain = BaseEnt - NewEnt
        if NewInfoGain > BestInfoGain:
            BestInfoGain = NewInfoGain
            BestFeature = labels[i]
    return BestFeature

#返回最佳分类属性，基于增益率
def bestFeature_GainGatio(dataSets, labels):
    BaseEnt = clacEntropy(dataSets)
    BestInfoGain = 0
    BestFeature = None
    for i in range(len(labels)):
        ClassList = np.array([example[i] for example in dataSets])
        ClassTypical = np.unique(ClassList)
        NewEnt = 0
        NewInfoGain = 0
        for branch in ClassTypical:
            proportion = np.sum(ClassList == branch) / len(ClassList)
            NewEnt += proportion * clacEntropy(dataSets[ClassList == branch])
            NewInfoGain = BaseEnt - NewEnt
        if NewInfoGain > BestInfoGain:
            BestInfoGain = NewInfoGain
            BestFeature = labels[i]
    return BestFeature

#返回最佳的分类属性，基于GiniIndex
def bestFeature_GiniIndex(dataSets, labels):
    BaseGini = clacGini(dataSets)
    BestGiniIndex = dataSets.shape[0]
    BestFeature = None
    for i in range(len(labels)):
        ClassList = np.array([example[i] for example in dataSets])
        ClassTypical = np.unique(ClassList)
        NewGiniIndex = 0
        for branch in ClassTypical:
            proportion = np.sum(ClassList == branch) / len(ClassList)
            NewGiniIndex += proportion * clacGini(dataSets[ClassList == branch])
        if NewGiniIndex < BestGiniIndex:
            BestGiniIndex = NewGiniIndex
            BestFeature = labels[i]
    return BestFeature

#删除第column列的数据（从0开始）
def splitDataSets(dataSets, column):
    newDataSets = np.delete(dataSets, column, axis=-1)
    return newDataSets

#计算dataSet中的最多target
def majorFeature(dataSets):
    Features = np.unique(dataSets)
    mostFeature = None
    mostNum = 0
    for feature in Features:
        num = np.sum(dataSets == feature)
        if(num > mostNum):
            mostNum = num
            mostFeature = feature
    return mostFeature
    
#创建决策树
def createTree(dataSets, labels, method='Entropy'):
    #target列向量
    TargetList = [example[-1] for example in dataSets]
    if (TargetList.count(TargetList[0]) == len(TargetList)):
        return TargetList[0]
    if (dataSets.shape[1] <= 1):
        return majorFeature(dataSets)
    #获得的Feature分支
    if method == 'Entropy':
        feature = bestFeature_InfoGain(dataSets, labels)
        print(feature)
    elif method == 'Gini':
        feature = bestFeature_GiniIndex(dataSets, labels)
    elif method == 'GainRatio':
        feature = bestFeature_GainGatio(dataSets, labels)
    #返回该Feature对应的属性column
    column = np.where(labels == feature)
    #获取该column的所有属性
    columnValues = np.array([example[column] for example in dataSets])
    #去重
    values = np.unique(columnValues)
    #删除该标签
    labels = np.delete(labels, column, axis=0)
    DecisionTree = { feature:{} }
    for value in values:
        subDataSets = dataSets[(columnValues == value).reshape(-1),:]
        subDataSets = splitDataSets(subDataSets, column)
        DecisionTree[feature][value] = createTree(subDataSets, labels, method)
    return DecisionTree
        
def main():
    dataSets = np.array([
                ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
                ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
                ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
                ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
                ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
                ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
                ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
                ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
                ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
                ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
                ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
                ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
                ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
                ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
                ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
                ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
                ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
                ])
    labels = np.array(['色泽','根蒂', '敲声', '纹理', '脐部', '触感'])
    # print(dataSets)
    # dataSets = np.array([['长', '粗', '男'],
    #             ['短', '粗', '男'],
    #             ['短', '粗', '男'],
    #             ['长', '细', '女'],
    #             ['短', '细', '女'],
    #             ['短', '粗', '女'],
    #             ['长', '粗', '女'],
    #             ['长', '粗', '女']])
    # labels = np.array(['头发','声音'])  #两个特征
    print(createTree(dataSets, labels,'GainRatio'))
 
main()