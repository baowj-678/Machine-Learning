import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def K_means(data, k):
    center_num = np.zeros([k, 1], dtype=np.int32)
    center_temp = np.zeros([k, data.shape[1]])
    labels = np.zeros([data.shape[0], ], dtype=np.int32)
    #初始化聚类中心
    center = data[np.random.choice(data.shape[0], size=k, replace=False), ]          
    is_ok = False
    while not is_ok:
        for i in range(data.shape[0]):
            #计算各点到各聚类中心的距离
            length = np.sum(abs(center - data[i, ])**data.shape[1], axis=1)**(data.shape[1])
            #获取最近的聚类中心的编号
            label = np.argmin(length)
            labels[i] = label                                               
            center_temp[label] += data[i]
            center_num[label] += 1
        #计算新的聚类中心
        center_temp /= center_num
        if (abs(center - center_temp) < 0.001).all():
            is_ok = True
        center = center_temp
        center_num = np.zeros([k, 1], dtype=np.int32)
        center_temp = np.zeros([k, data.shape[1]], dtype=np.float32)
    return (center, labels)

def main():
    # data = np.array([[1.2, 2.3]
    #                 ,[1.1, 3.2]
    #                 ,[3.2, 2.5]
    #                 ,[8.9, 9.6]
    #                 ,[4, 5]
    #                 ,[1, 2]
    #                 ,[2.3, 4.5]
    #                 ,[3.2, 5.6]
    #                 ,[1.2, 0.9]
    #                 ,[1.8, 1.9]
    #                 ,[1.8, 9.0]
    #                 ,[3.4, 7.8]
    #                 ,[9.9, 7.7]
    #                 ,[3.5, 9.0]
    #                 ,[2.3, 4.5]
    #                 ,[7.8, 4.5]
    #                 ,[1.2, 1.2]
    #                 ,[7.8, 3.4]
    #                 ,[0.4, 0.7]]
    #                 )3
    data = np.array(load_iris().data)
    k = 4
    center, labels = K_means(data, k)
    colors = ['skyblue', 'wheat', 'crimson', 'lime']
    for i in range(k):
        label_temp = labels == i
        data_temp = data[label_temp]
        plt.scatter(data_temp[:, 0], data_temp[:, 1], color=colors[i])
        plt.scatter(center[i, 0], center[i, 1], marker='*', color=colors[i])
    plt.show()
    print("\nfinal center:\n", center)
    print("\nfinal labels:\n", labels)

main()


