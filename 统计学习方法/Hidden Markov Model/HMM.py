import numpy as np
import matplotlib.pyplot as plt


class HMM():
    def  __init__(self):
        pass

    def train(self, pi, A, B):
        self.pi = pi
        self.A = A
        self.B = B
    
    def setO(self, O):
        self.O = O
        self.K = self.O.shape[0]
    
    def setI(self, I):
        self.I = I
        self.N = self.I.shape[0]

    def setI_index(self, O_index):
        self.I_index = O_index
        #


    def setO_index(self, O_index):
        self.O_index = O_index
        self.T = O_index.shape[0]
    
    def updateAlpha(self, t):
        alpha_temp = np.dot(self.alpha.transpose(), self.A)
        self.alpha = np.multiply(alpha_temp.transpose(), self.B[:, self.O_index[t]])
        return self.alpha

    def updateBeta(self, t):
        beta_temp = np.multiply(self.B[:, self.O_index[t + 1]], self.beta)
        self.beta = np.dot(self.A, beta_temp)
        return self.beta

    def evaluate_direct(self, obervation):
        self.observation2index(obervation)
        pass

    def evaluate_direct_recurse(self, observation, last_i):
        pass

    def evaluate_forward(self, observation):
        self.observation2index(observation)
        # initialize alpha
        self.alpha = np.multiply(self.pi, self.B[:, self.O_index[0]])
        # loop updating
        for t in range(self.T - 1):
            self.updateAlpha(t + 1)
        P = np.sum(self.alpha)
        return P
    
    def evaluate_backward(self, observation):
        self.observation2index(observation)
        # initialize beta
        self.beta = np.ones([self.N, 1], dtype=np.float)
        # loop updating
        for t in range(self.T - 2, -1, -1): 
            self.updateBeta(t)
        P = np.sum(np.multiply(np.multiply(self.pi, self.B[:, self.O_index[0]]), self.beta))
        return P

    def observation2index(self, observation):
        temp = np.zeros([observation.shape[0], 1], dtype=np.int32)
        for i in range(observation.shape[0]):
            temp[i] = np.where(self.O == observation[i])[0]
        self.setO_index(temp)
    

    def calcGama(self, t, N):
        pass
    
    def calcGamaAndXi(self, N, t):
        # calc beta
        # initialization
        self.beta = np.ones([self.pi.shape[0], 1])
        beta = np.zeros([t, N])
        beta[t - 1] = self.beta.transpose()
        for i in range(t - 2, -1, -1):
            self.updateBeta(i)
            beta[i] = self.beta.transpose()

        # calc alpha
        # initialization
        self.alpha = np.multiply(self.pi, self.B[:, self.O_index[0]])
        alpha = np.zeros([t, N])
        alpha[0] = self.alpha.transpose()
        for i in range(t - 1):
            self.updateAlpha(i + 1)
            alpha[i + 1] = self.alpha.transpose()
        # calc Gama
        self.Gama = np.multiply(alpha, beta)
        self.Gama = self.Gama/np.sum(self.Gama, axis=1).reshape([-1, 1])

        # calc Xi
        self.Xi = np.zeros([t - 1, N, N])
        for i in range(t - 1):
            temp = alpha[i, :].reshape([-1, 1])
            temp = np.multiply(temp, self.A)
            temp = np.multiply(temp, self.B[:, self.O_index[i + 1]].transpose())
            temp = np.multiply(temp, beta[i + 1, :])
            self.Xi[i] = temp/np.sum(temp)
        return (self.Gama, self.Xi)

    def calcXi(self, N, t):
        pass

    def learning(self, observation, hidden_num, num=2000):
        # 去重
        # self.seto(np.unique(observation))
        self.observation2index(observation)
        # initialization
        PLUS = 3
        self.N = hidden_num
        self.pi = np.random.normal(size=[self.N, 1]) + PLUS
        self.pi = self.pi/np.sum(self.pi)
        self.A = np.random.normal(size=[self.N, hidden_num]) + PLUS
        self.A = self.A/np.sum(self.A, axis=1).reshape([-1, 1])
        self.B = np.random.normal(size=[hidden_num, self.O.shape[0]]) + PLUS
        self.B = self.B/np.sum(self.B, axis=1).reshape([-1, 1])
        for n in range(num):
            self.calcGamaAndXi(self.N, self.T)
            # update A
            self.A = np.sum(self.Xi, axis=0)/((np.sum(self.Gama, axis=0)-self.Gama[self.T - 1, :]).reshape([-1, 1]))

            # update B
            temp = np.zeros([self.N, self.K])
            for j in range(self.N):
                temp_ = np.sum(np.dot(self.Gama[:, j].reshape([-1, 1]), self.B[j, :].reshape([1, -1])), axis=0)
                temp[j] = temp_ / np.sum(temp_)
            self.B = temp

            # update pi
            self.pi = self.Gama[0, :].reshape([-1, 1])



    def decoding(self, observation):
        # initialization
        self.observation2index(observation)
        self.delta = np.multiply(self.pi, self.B[:, self.O_index[0]])
        self.fi = np.zeros([self.T, self.N])
        # loop
        for i in range(self.T - 1):
            temp = np.multiply(self.delta, self.A)
            temp = np.multiply(temp, self.B[:, self.O_index[i + 1]].reshape([1, -1]))
            self.delta = np.max(temp, axis=0)
            self.fi[i + 1] = np.argmax(temp, axis=0)
        P = np.max(self.delta)
        I_star = np.zeros(self.T, dtype=int)
        I_star[self.T - 1] = np.argmax(self.delta)
        for i in range(self.T - 2, -1, -1):
            I_star[i] = self.fi[i + 1, I_star[i + 1]]
        return (P, I_star)

    def generate_observations_max_posibility(self, length):
        I = np.zeros([length], dtype=int)
        O = np.zeros([length], dtype=self.O.dtype)
        t = 0

        I[t] = np.argmax(self.pi)
        index = np.argmax(self.B[I[t], :])
        O[0] = self.O[index]
        for i in range(1, length, 1):
            I[i] = np.argmax(self.A[I[i - 1], :])
            index = np.argmax(self.B[I[i], :])
            O[i] = self.O[index]
        return O

    def generate_observations(self, length):
        I = np.zeros([length + 1], dtype=int)
        O = np.zeros([length], dtype=self.O.dtype)
        t = 0

        p = np.random.rand()
        p_sum = 0
        for i in range(self.N):
            p_sum += self.pi[i]
            if(p_sum >= p):
                I[t] = i
                break
        for t in range(0, length, 1):
            # generate O[t]
            p_sum = 0
            p = np.random.rand()
            index = 0
            for i in range(self.K):
                p_sum += self.B[I[t], i]
                if(p_sum >= p):
                    index = i
                    break
            O[t] = self.O[index]

            p = np.random.rand()
            p_sum = 0
            for i in range(self.N):
                p_sum += self.pi[i]
                if(p_sum >= p):
                    I[t + 1] = i
        return O
    

x = np.linspace(0, 4, 100).reshape([-1, 1])
y = np.sin(x)
hmm = HMM()
hmm.setO(y)

hmm.learning(y, 1000, num=400)

# print('final:\n')
print('Pi\n', hmm.pi)
print('A\n', hmm.A)
print("B\n", hmm.B)

plt.plot(x, y)
x = np.linspace(0, 31.4, 1000).reshape([-1, 1])
y = hmm.generate_observations(1000)
# print(y)
plt.plot(x, y)
plt.show()
