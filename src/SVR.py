import numpy as np
from scipy.sparse import dok_matrix
from scipy.optimize import minimize


class SVR:
    def __init__(self, C=1, epsilon=0.001, gamma=0):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.K = None
        self.w = None
        self.b = None

    def _target(self, theta, y):
        ssize = len(y)
        alpha1 = theta[:ssize]
        alpha2 = theta[ssize:]
        alphadiff = alpha1 - alpha2
        error = 0
        for i in range(ssize):
            error += self.epsilon * (alpha1[i] + alpha2[i])
            if alphadiff[i] == 0:
                continue
            for j in range(ssize):
                if alphadiff[j] == 0:
                    continue
                error += 0.5 * alphadiff[i] * alphadiff[j] * self.K[i, j]
            error -= y[i] * alphadiff[i]

        jacobian = np.zeros(theta.shape)
        for i in range(ssize):
            for j in range(ssize):
                if alphadiff[j] == 0:
                    continue
                jacobian[i] += alphadiff[j] * self.K[i, j]
                jacobian[i + ssize] += -alphadiff[j] * self.K[i, j]
            jacobian[i] += self.epsilon - y[i]
            jacobian[i + ssize] += self.epsilon + y[i]

        return error, jacobian

    def fit(self, X, y):
        if self.gamma==0:
            self.gamma = 1/X.shape[1]
        ssize = len(y)
        self.K = Kmatrix(X, self.gamma)
        cons = ({'type': 'eq',
                 'fun': lambda theta: np.array(sum(theta[:ssize]) - sum(theta[ssize:])),
                 'jac': lambda _: np.append(np.ones(ssize), -np.ones(ssize))},
                {'type': 'ineq',
                 'fun': lambda theta: theta,
                 'jac': lambda _: np.eye(ssize * 2)},
                {'type': 'ineq',
                 'fun': lambda theta: self.C - theta,
                 'jac': lambda _: -np.eye(ssize * 2)})
        res = minimize(self._target, np.zeros(ssize * 2), args=(y,), jac=True, constraints=cons,
                       method='SLSQP', options={'disp': True})
        opt_theta = res.x
        w = (opt_theta[:ssize].T - opt_theta[ssize:].T) * X
        b1 = [y[x] - w*X[x].T - self.epsilon for x in range(ssize) if 0 < opt_theta[x] < self.C]
        b2 = [y[x] - w*X[x].T + self.epsilon for x in range(ssize) if 0 < opt_theta[x + ssize] < self.C]
        b = np.mean(np.append(b1, b2))
        self.w = w
        self.b = b

    def predict(self, X):
        return self.w*X.T+self.b

    def jduge(self,yp,yg):
        return np.mean(np.square(yp-yg))**.5

class Kmatrix:
    def __init__(self, X, gamma):
        self.X = X
        self.gamma = gamma
        self.matrix = dok_matrix((X.shape[0], X.shape[0]), np.float64)

    def __getitem__(self, item):
        if self.matrix.has_key(item):
            return self.matrix[item]
        value = self._rbf(self.X[item[0]].A, self.X[item[1]].A)
        self.matrix[item] = value
        return value

    def _rbf(self, x, y):
        return np.exp(-self.gamma * np.sum(np.square(x-y)))
