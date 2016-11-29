
import warnings

from sklearn import datasets

from SVR import SVR


warnings.simplefilter('error', RuntimeWarning)

print 'loading features'
X, y = datasets.load_svmlight_file('../data/train_vectors.txt')
rmse=[]
model = SVR(epsilon=0.0001,gamma=0.01,C=1)
model.fit(X[:4000],y[:4000])
print 'testing'
yp = model.predict(X[4000:])
rmse += [model.jduge(yp,y[4000:])]
print rmse
