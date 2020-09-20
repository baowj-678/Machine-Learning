import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

#Xi为列向量
def calcOcirideDistance(X):
    distance = []
    for i in range(X.shape[0]):
        distance.append(np.sum((X[i] - X) ** 2, axis=1))
    distance = np.array(distance)
    return distance

def hierarchicalCluster(data, n=2):
    distance = calcOcirideDistance(data)
    now_n = data.shape[0]
    cluster = []
    for i in range(now_n):
        cluster.append([i])
        distance[i, i] = 10000000
    # print(cluster)
    print(distance)
    while(len(cluster) > n):
        index = np.unravel_index(distance.argmin(),distance.shape)
        cluster_i = None
        cluster_j = None
        for i in range(len(cluster)):
            for j in range(len(cluster[i])):
                if cluster[i][j] == index[0]:
                    cluster_i = i
                elif cluster[i][j] == index[1]:
                    cluster_j = i
        if cluster[cluster_i] == cluster[cluster_j]:
            distance[index[0]][index[1]] = 10000000
            distance[index[1]][index[0]] = 10000000
            continue
        elif cluster[cluster_i] > cluster[cluster_j]:
            for i in range(len(cluster[cluster_j])):
                cluster[cluster_i].insert(-1, cluster[cluster_j][i])
            del(cluster[cluster_j])
        else:
            for i in range(len(cluster[cluster_i])):
                cluster[cluster_j].insert(-1, cluster[cluster_i][i])
            del(cluster[cluster_i])
        distance[index[0]][index[1]] = 10000000
        distance[index[1]][index[0]] = 10000000
    return cluster


def main():
    data = np.array(load_iris().data)
    # plt.scatter()
    print(data[:, 1:3])
    cluster = hierarchicalCluster(data[:,1:3], 5)
    for i in range(len(cluster)):
        data_x = data[cluster[i], 1]
        data_y = data[cluster[i], 2]
        plt.scatter(data_x, data_y)
    plt.show()

main()