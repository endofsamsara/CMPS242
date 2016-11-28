
import warnings

from sklearn import datasets

from SVR import SVR


warnings.simplefilter('error', RuntimeWarning)

print 'loading features'
X, y = datasets.load_svmlight_file('../data/train_vectors.txt')
model = SVR(epsilon=0.001,gamma=1)
model.fit(X[:200],y[:200])
print 'testing'
yp = model.predict(X[4000:])
rmse = model.jduge(yp,y[4000:])
print yp,y[4000:]
print rmse
