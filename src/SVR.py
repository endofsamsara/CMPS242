import numpy as np


class SVR:
    def __init__(self, C=1, epsilon=0.001, gamma=0):
        self.valid = None
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.K = None
        self.b = 0
        self.alpha = None
        self.y = None
        self.trainSize = None
        self.X = None

    def _error(self, k):
        pk = self.alpha * self.K[:, k] + self.b
        diff = pk - self.y[k]
        return diff

    def _selectj(self, alpha_i):

        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(np.random.uniform(0, self.trainSize))
        error_j = self._error(alpha_j)

        return alpha_j, error_j

    def _getLim(self, gamma, condition):
        if condition == 1:
            return max(0, gamma - self.C), min(self.C, gamma)
        elif condition == 2:
            return max(0, gamma), min(self.C, self.C + gamma)
        elif condition == 3:
            return max(0, -gamma), min(self.C, self.C - gamma)
        elif condition == 4:
            return max(0, -gamma - self.C), min(self.C, -gamma)

    def _smo(self, i):
        error_i = self._error(i)
        if (abs(error_i) - self.epsilon < -0.001) and (self.alpha[i] != 0) or (
                        abs(error_i) - self.epsilon > 0.001) and (-self.C < self.alpha[i] < self.C):

            j, error_j = self._selectj(i)
            j_old = self.alpha[j]

            eta = -2 * self.K[i, j] + 2
            if eta < 0:
                return 0
            gamma = self.alpha[i] + self.alpha[j]
            LH1 = (max(0, gamma - self.C), min(self.C, gamma))
            LH2 = (max(0, gamma), min(self.C, self.C + gamma))
            LH3 = (max(-self.C, -self.C + gamma), min(0, gamma))
            LH4 = (max(-self.C, gamma), min(0, gamma + self.C))

            errdiff = -error_j + error_i
            can1 = self.alpha[j] + errdiff / eta
            can2 = self.alpha[j] + (errdiff - 2 * self.epsilon) / eta
            can3 = self.alpha[j] + (errdiff + 2 * self.epsilon) / eta
            can4 = self.alpha[j] + errdiff / eta

            # step 5: clip alpha j
            if gamma > 0:
                if LH1[0] <= can1 <= LH1[1]:
                    self.alpha[j] = can1
                elif LH3[0] <= can3 <= LH3[1]:
                    self.alpha[j] = can3
                elif LH2[0] <= can2 <= LH2[1]:
                    self.alpha[j] = can2
                elif can3 < LH3[0]:
                    self.alpha[j] = gamma - self.C
                elif can2 > LH2[1]:
                    self.alpha[j] = self.C
                elif LH3[1] < can3 and can1 < LH1[0]:
                    self.alpha[j] = 0
                elif LH1[1] < can1 and can2 < LH2[0]:
                    self.alpha[j] = gamma
            elif gamma < 0:
                if LH4[0] <= can4 <= LH4[1]:
                    self.alpha[j] = can4
                elif LH3[0] <= can3 <= LH3[1]:
                    self.alpha[j] = can3
                elif LH2[0] <= can2 <= LH2[1]:
                    self.alpha[j] = can2
                elif can3 < LH3[0]:
                    self.alpha[j] = -self.C
                elif can2 > LH2[1]:
                    self.alpha[j] = gamma + self.C
                elif LH3[1] < can3 and can4 < LH4[0]:
                    self.alpha[j] = gamma
                elif LH4[1] < can4 and can2 < LH2[0]:
                    self.alpha[j] = 0
            elif gamma == 0:
                if LH3[0] <= can3 <= LH3[1]:
                    self.alpha[j] = can3
                elif LH2[0] <= can2 <= LH2[1]:
                    self.alpha[j] = can2
                elif can3 < LH3[0]:
                    self.alpha[j] = -self.C
                elif can2 > LH2[1]:
                    self.alpha[j] = self.C
                elif LH3[1] < can3 and can2 < LH2[0]:
                    self.alpha[j] = 0

            self.alpha[i] = gamma - self.alpha[j]

            if abs(j_old - self.alpha[j]) < 0.00001:
                return 0

            b1 = self.b - error_i - (self.alpha[j] - j_old) * (self.K[i, j] - 1)
            b2 = self.b - error_j - (self.alpha[j] - j_old) * (self.K[i, j] - 1)
            if self.alpha[i] not in (-self.C, self.C, 0):
                self.b = b1
            elif self.alpha[j] not in (-self.C, self.C, 0):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0

            return 1
        else:
            return 0

    def fit(self, X, y, maxIter=1000):
        self.X = X
        self.y = y
        self.trainSize = X.shape[0]
        self.alpha = np.zeros(self.trainSize, np.float32)
        self.valid = np.zeros(self.trainSize, np.int8)
        if self.gamma==0:
            self.gamma = 1./X.shape[1]
        print 'calculating kernels'
        K = np.mat(np.zeros([self.trainSize, self.trainSize], np.float32))
        for i in range(self.trainSize):
            if i%100==0:
                print '{:d} rows calculated'.format(i)
            for j in range(self.trainSize):
                if i < j:
                    K[i, j] = np.exp(-self.gamma * (X[i, :] - X[j, :]).power(2).sum())
                elif i == j:
                    K[i, j] = 1
                else:
                    K[i, j] = K[j, i]
        self.K = K
        entireSet = True
        alphaPairsChanged = 1
        iterCount = 0
        while iterCount < maxIter and alphaPairsChanged > 0 :
            alphaPairsChanged = 0

            for i in xrange(self.trainSize):
                alphaPairsChanged += self._smo(i)
            print '---iter:{:d} entire set, alpha pairs changed:{:d}'.format(iterCount, alphaPairsChanged)
            iterCount += 1

    def predict(self, X):
        nzind = np.nonzero(self.alpha)[0]
        nzalpha = np.mat(self.alpha[nzind, ])
        K = np.zeros([len(nzind),X.shape[0]],np.float32)
        for i in range(len(nzind)):
            for j in range(X.shape[0]):
                a = nzind[i]
                K[i, j] = np.exp(-self.gamma * (self.X[a, :] - X[j, :]).power(2).sum())
        return nzalpha*K

    def jduge(self,yp,yg):
        return np.mean(np.square(yp-yg))**.5

