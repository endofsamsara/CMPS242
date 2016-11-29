import numpy as np
from scipy.sparse import csr_matrix
import warnings

from sklearn import datasets

from SVR import SVR


def cal(X):
    return 0.01*(X[0]+2*X[1]+3*X[2])

X1=[1,2,3,7,3,2,1,10,8,3,1,2,1,3,1,2,1]
X2=[0,3,4,1,14,19,0,1,2,4,3,0,2,4,0,4,3]
X3=[2,1,5,16,5,0,14,9,3,0,7,4,10,6,8,12,18]

X=np.vstack((X1,X2,X3)).T
X = csr_matrix(X)
y=[]
for i in zip(X1,X2,X3):
    y.append(cal(i))


print y
print X


warnings.simplefilter('error', RuntimeWarning)

print 'loading features'
#X, y = datasets.load_svmlight_file('../data/train_vectors.txt')
rmse=[]
model = SVR(epsilon=0.001,gamma=0.01,C=1)
model.fit(X[:10],y[:10])
print 'testing'
yp = model.predict(X[10:])
rmse += [model.jduge(yp,y[10:])]
print yp,y[10:]
print rmse
