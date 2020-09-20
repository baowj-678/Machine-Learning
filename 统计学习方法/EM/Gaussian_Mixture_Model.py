import  numpy as np
import numpy.linalg as llg
import matplotlib.pyplot as plt

def GaussianF(mu, sigma, data_Y, K):
    Gaussian = []
    sigma_inv = llg.pinv(sigma)
    sigma_value = llg.det(sigma)
    for k in range(K):
        first = 1/(((2*np.pi)**(data_Y.shape[0]/2))*np.sqrt(sigma_value[k]))
        Gaussian_k = []
        for i in range(data_Y.shape[1]):
            second = np.exp(-0.5*np.dot(np.dot((data_Y[:, i] - mu[:, k])[np.newaxis, :], sigma_inv[k,::]), (data_Y[:, i] - mu[:, k])[:np.newaxis]))
            Gaussian_k.append((first*second).tolist())
        Gaussian.append(Gaussian_k)
    Gaussian = np.array(Gaussian)
    Gaussian = np.transpose(Gaussian.reshape((Gaussian.shape[0], Gaussian.shape[1])))
    return Gaussian

def CalcMu(data_Y, gamma, K):
    mu = []
    for i in range(K):
        mu_up = 0
        mu_down = 0
        for j in range(data_Y.shape[1]):
            mu_up += gamma[j, i]*data_Y[:, j]
            mu_down += gamma[j, i]
        mu.append(mu_up/mu_down)
    mu = np.transpose(np.array(mu))
    return mu
    
def Covariance(gamma, data_Y, mu, K):
    sigma = []
    for i in range(K):
        sigma_up = 0
        sigma_down = 0
        for j in range(data_Y.shape[1]):
            y = data_Y[:, j][:, np.newaxis]
            mu_i = mu[:, i][:, np.newaxis]
            sigma_up += gamma[j, i]*np.dot((y - mu_i), np.transpose(y - mu_i))
            sigma_down += gamma[j, i]
        sigma.append(sigma_up/sigma_down)
    sigma = np.array(sigma)
    return sigma

def Mixture(gamma, num, K):
    alpha = []
    for k in range(K):
        alpha_j = 0
        for j in range(num):
            alpha_j += gamma[j, k]
        alpha.append(alpha_j/num)
    return alpha

def CalcGamma(alpha, mu, sigma, data_Y, K):
    Gaussian = GaussianF(mu, sigma, data_Y, K)
    gamma = []
    for j in range(data_Y.shape[1]):
        Gaussian_j = 0
        gamma_j = []
        for k in range(K):
            Gaussian_j += alpha[k]*Gaussian[j, k]
        for k in range(K):
            gamma_j.append(alpha[k]*Gaussian[j, k]/Gaussian_j)
        gamma.append(gamma_j)
    gamma = np.array(gamma)
    return gamma

def GMM(data_Y, K=2):#data_Y为列向量
    alpha = [1/K]*K
    sigma = []
    for i in range(K):
        sigma.append(np.eye(data_Y.shape[0]))
    sigma = np.array(sigma)
    mu = []
    for i in range(K):
        mu.append(data_Y[:,i])
    mu = np.transpose(np.array(mu))
    gamma = CalcGamma(alpha, mu, sigma, data_Y, K)
    for i in range(100):
        mu = CalcMu(data_Y, gamma, K)
        sigma = Covariance(gamma, data_Y, mu, K)
        alpha = Mixture(gamma, data_Y.shape[1], K)
        gamma = CalcGamma(alpha, mu, sigma, data_Y, K)
    print(gamma)
    return gamma

def main():
    data_0 = np.random.normal(loc=1, scale=2, size=(100,2))
    data_1 = np.random.normal(loc=10, scale=2, size=(100,2))
    data_2 = np.random.normal(loc=5, scale=1, size=(100,2))
    data_Y = []
    for i in range(data_0.shape[0]):
        data_Y.append(data_0[i, :])
        data_Y.append(data_1[i, :])
        data_Y.append(data_2[i, :])
    data_Y = np.transpose(data_Y)
    K = 3
    gamma = GMM(data_Y, K)
    index = np.argmax(gamma, axis=1)
    index_ = []
    data_Y_ = []
    for i in range(K):
        data_Y_ = data_Y[:,np.where(index == i)]
        plt.scatter(data_Y_[0, :], data_Y_[1, :])
    plt.show()

main()

