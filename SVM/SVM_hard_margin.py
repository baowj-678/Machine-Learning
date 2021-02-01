import numpy as np
from matplotlib import pyplot as plt

class data:
    def __init__(self, dataSets, C, kernal, d, theta):
        #Xi中i的数目，为行向量
        self.n = dataSets.shape[0]
        #X的属性数目，为行向量
        self.m = dataSets.shape[1] - 1
        #X，行向量
        self.X = dataSets[:, :self.m]
        #Y，列向量
        self.Y = np.array(dataSets[:, self.m], dtype=np.int32)[:, np.newaxis]
        #b截距
        self.b = 0
        #alpha,列向量
        self.alpha = np.zeros([self.n, 1])
        #C系数
        self.C = C
        #核函数选择
        self.kernal = kernal
        #多项式核函数的d，高斯核函数的sigma
        self.d = d
        #Sigmoid核的参数
        self.theta = theta

#计算f(x)
def calcFx(dataInfo, x):
    if dataInfo.kernal == 'Linear':
        fx = dataInfo.alpha * dataInfo.Y * np.sum(np.multiply(dataInfo.X, x), axis=1)[:, np.newaxis]
        fx = np.sum(fx) + dataInfo.b
        return fx
    elif dataInfo.kernal == 'Polynomial':
        fx = dataInfo.alpha * dataInfo.Y * np.sum(np.multiply(dataInfo.X, x) ** dataInfo.d, axis=1)[:, np.newaxis]
        fx = np.sum(fx) + dataInfo.b
        return fx
    elif dataInfo.kernal == 'RBF':
        fx = dataInfo.alpha * dataInfo.Y * np.exp(- np.sum((dataInfo.X - x) ** 2 / (2 * dataInfo.d * dataInfo.d), axis=1)[:, np.newaxis])
        fx = np.sum(fx) + dataInfo.b
        return fx
    elif dataInfo.kernal == 'Laplace':
        fx = dataInfo.alpha * dataInfo.Y * np.exp(- np.sum(np.abs(dataInfo.X - x) / dataInfo.d, axis=1)[:, np.newaxis])
        fx = np.sum(fx) + dataInfo.b
        return fx
    elif dataInfo.kernal == 'Sigmoid':
        fx = dataInfo.alpha * dataInfo.Y * np.tanh(dataInfo.d * np.sum(np.multiply(dataInfo.X, x), axis=1)[:, np.newaxis] + dataInfo.theta)
        fx = np.sum(fx) + dataInfo.b
        return fx

def calck(x1, x2, dataInfo):
    if dataInfo.kernal == 'Linear':
        x2 = x2[:, np.newaxis]
        return np.dot(x1, x2)
    elif dataInfo.kernal == 'Polynomial':
        x2 = x2[:, np.newaxis]
        return np.dot(x1, x2) ** dataInfo.d
    elif dataInfo.kernal == 'RBF':
        s = np.sum((x1 - x2) ** 2) / (2 * dataInfo.d ** 2)
        return np.exp(-s)
    elif dataInfo.kernal == 'Laplace':
        s = np.sum(np.abs(x1 - x2)) / dataInfo.d
        return np.exp(-s)
    elif dataInfo.kernal == 'Sigmoid':
        x2 = x2[:, np.newaxis]
        return np.tanh(dataInfo.d * np.dot(x1, x2) + dataInfo.theta)

#返回alpha_new未修改dataInfo
def clipAlpha(data, alphai_new, i, j):
    L = None
    H = None
    alpha_new = None
    if data.Y[i, 0] == data.Y[j, 0]:
        L = max(0, data.alpha[i, 0] + data.alpha[j, 0] - data.C)
        H = min(data.C, data.alpha[i, 0] + data.alpha[j, 0])
    else:
        L = max(0, data.alpha[i, 0] - data.alpha[j, 0])
        H = min(data.C, data.C + data.alpha[i, 0] - data.alpha[j, 0])

    if alphai_new > H:
        alpha_new = H
    elif L <= alphai_new <= H:
        alpha_new = alphai_new
    else:
        alpha_new = L
    return alpha_new

def hard_margin_SVM(dataSets, C, kernal='Linear', d = 1, theta = 1, times = 10):
    dataInfo = data(dataSets, C, kernal, d, theta)
    #是否所有的拉格朗日乘子均满足KKT条件
    is_kkt_all = False
    t = 0
    for time in range(times):
        print(time)
        #选择第一个alpha
        for i in range(dataInfo.n):
            #计算f(xi)
            fxi = calcFx(dataInfo, np.transpose(dataInfo.X[i,:]))
            # 判断alphai是否满足KKT条件
            if(abs(dataInfo.alpha[i]) <= 0.001 and (dataInfo.Y[i] * fxi) >= 1):
                continue
            if(abs(dataInfo.alpha[i] - dataInfo.C) <= 0.001 and (dataInfo.Y[i] * fxi) <= 1):
                continue
            if(0 < dataInfo.alpha[i] < dataInfo.C and abs(dataInfo.Y[i] * fxi - 1) <= 0.001):
                continue
        #如果全部满足KKT，随机选择alphai
            fxi = calcFx(dataInfo, np.transpose(dataInfo.X[i,:]))
            #计算误差Ei
            Ei = fxi - dataInfo.Y[i]
            #Ei Ej最大差值
            deltaE_max = 0
            t += 1
            # print(t)
            Ej = None
            j = None
            #寻找第二个优化变量alphaj
            for k in range(dataInfo.n):
                fxk = calcFx(dataInfo, np.transpose(dataInfo.X[k, :]))
                Ek = fxk - dataInfo.Y[k]
                if abs(Ei -Ek) > deltaE_max:
                    j = k
                    Ej = Ek
                    deltaE_max = abs(Ei - Ej)
            #计算eta
            eta = calck(dataInfo.X[i, :], dataInfo.X[i, :], dataInfo) + calck(dataInfo.X[j, :], dataInfo.X[j, :], dataInfo) - 2 * calck(dataInfo.X[i, :], dataInfo.X[j, :], dataInfo)
            alphai_new = dataInfo.alpha[i] + (dataInfo.Y[i] * (Ej - Ei)) / eta
            alphai_new = clipAlpha(dataInfo, alphai_new, i, j)
            alphaj_new = dataInfo.alpha[j, 0] + dataInfo.Y[i, 0] * dataInfo.Y[j, 0] * (dataInfo.alpha[i, 0] - alphai_new)
            #更新b
            bi = - Ei - dataInfo.Y[i, 0] * (alphai_new - dataInfo.alpha[i, 0]) * calck(dataInfo.X[i, :], dataInfo.X[i, :], dataInfo)\
                - dataInfo.Y[j, 0] * (alphaj_new - dataInfo.alpha[j, 0]) * calck(dataInfo.X[i, :], dataInfo.X[j, :], dataInfo) + dataInfo.b
            bj = - Ej - dataInfo.Y[i, 0] * (alphai_new - dataInfo.alpha[i, 0]) * calck(dataInfo.X[j, :], dataInfo.X[i, :], dataInfo)\
                - dataInfo.Y[j, 0] * (alphaj_new - dataInfo.alpha[j, 0]) * calck(dataInfo.X[j, :], dataInfo.X[j, :], dataInfo) + dataInfo.b
            if 0 < alphai_new < dataInfo.C:
                dataInfo.b = bi
            elif 0 < alphaj_new < dataInfo.C:
                dataInfo.b = bj
            else:
                dataInfo.b = (bi + bj) / 2
            #更新alphai,alphaj
            dataInfo.alpha[i, 0] = alphai_new
            dataInfo.alpha[j, 0] = alphaj_new
    for i in range(dataInfo.n):
        #计算f(xi)
        fxi = calcFx(dataInfo, np.transpose(dataInfo.X[i,:]))
        print('alpha:',dataInfo.alpha[i],'fxi:', fxi, 'Y:', dataInfo.Y[i], '*:', dataInfo.Y[i] * fxi, '\n')
    w = np.sum((dataInfo.alpha * dataInfo.Y * dataInfo.X), axis=0)
    w = w[:, np.newaxis]
    return w, dataInfo.b, dataInfo

def main():
    data = np.array([])
    num = 200
    for i in range(num):
        p = []
        x = np.random.uniform(0, 6)
        y = np.random.uniform(-5, 2)
        if y < ((7 - x * x) / 6):
            p.append(x)
            p.append(y)
            p.append(-1)
        elif y > ((7 - x * x) / 6):
            p.append(x)
            p.append(y)
            p.append(1)
        data = np.append(data, p)
    data = data.reshape([-1, 3])
    print(data.shape)
    data_index = data[:, 2]
    data_plus = data[data_index == 1]
    data_minus = data[data_index == -1]
    plt.scatter(data_plus[:, 0],data_plus[:,1])
    plt.scatter(data_minus[:, 0],data_minus[:,1])
    w, b, dataInfo = hard_margin_SVM(data, C=80, times=10, kernal='Sigmoid', d=2)
    print(w,b)
    x = np.linspace(np.min(data, axis=0)[0], np.max(data, 0)[0], 50)
    y = (-b - w[0]*x)/w[1]
    plt.plot(x,y)
    plt.show()

def test():
    data = np.array([])
    num = 300
    for i in range(num):
        p = []
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 8)
        if y ** 2 < 16 - (x - 5) ** 2:
            p.append(x)
            p.append(y)
            p.append(-1)
        elif y ** 2 > 16 - (x - 5) ** 2:
            p.append(x)
            p.append(y)
            p.append(1)
        data = np.append(data, p)
    data = data.reshape([-1, 3])

    x = np.linspace(1.2, 8.8, 50)
    y = np.sqrt(16 - (x - 5) ** 2)
    plt.plot(x,y)

    w, b, dataInfo = hard_margin_SVM(data, C=0.1, times=30, kernal='RBF', d=1)
    print(w,b)
    data = np.array([])
    num = 200
    for i in range(num):
        p = []
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 8)
        if calcFx(dataInfo, (x, y)) > 0:
            p.append(x)
            p.append(y)
            p.append(1)
        elif calcFx(dataInfo, (x, y)) < 0:
            p.append(x)
            p.append(y)
            p.append(-1)
        data = np.append(data, p)
    data = data.reshape([-1, 3])
    # print(data)
    data_index = data[:, 2]
    data_plus = data[data_index == 1]
    data_minus = data[data_index == -1]
    plt.scatter(data_plus[:, 0],data_plus[:,1])
    plt.scatter(data_minus[:, 0],data_minus[:,1])

    plt.show()
    return 0

test()
