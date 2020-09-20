import numpy as np

laplace = 1
def naiveBayes(X, Y, x):
    #X、x为列向量，y为行向量
    #构建bayes树
    C = set(Y)                          #类别集合
    P = {}                              #记录P（x = ajl|y = ck)
    P_ci = {}                           #记录P（y = ck）
    for i in C:
        numsOfci = np.sum(Y == i)       #ci类的样本
        Pci = (numsOfci + laplace) / (Y.shape[0] + laplace)
        P_ci[i] = Pci

        Xci = X[:, Y == i]                   #属于ci类的样本
        P_x_ci = {}                          #ci类

        for j in range(X.shape[0]):          #对x每一维度进行遍历
            aj = set(X[j, :])                #x的j维的所有属性集合
            P_x_ci_j = {}                    #ci类、j维

            for ak in aj:
                Pxij = (np.sum(Xci[j, :] == ak) + laplace) / (numsOfci + laplace)
                P_x_ci_j[ak] = Pxij

            P_x_ci[j] = P_x_ci_j

        P[i] = P_x_ci
    #构建完成
    y = None
    max_py = 0
    for ci in C:
        p_y_ci = P_ci[ci]
        for i in range(X.shape[0]):
            p_y_ci = p_y_ci * P[ci][i][x[i, 0]]
        if p_y_ci > max_py:
            max_py = p_y_ci
            y = ci
    return y

    
        


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
                ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
                ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
                ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
                ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
                ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
                ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']
                ])
    labels = np.array(['色泽','根蒂', '敲声', '纹理', '脐部', '触感'])
    x = np.array(['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑'])[:, np.newaxis]
    y = np.array(['否'])
    X = np.transpose(dataSets[:, :6])
    Y = dataSets[:, 6]
    print(x.shape, y.shape, X.shape, Y.shape)
    print(naiveBayes(X, Y, x))



main()