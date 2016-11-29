from sklearn import datasets,svm

from SVR import SVR

X, y = datasets.load_svmlight_file('../data/train_vectors.txt')
clf = svm.SVR()
clf.fit(X[:4000],y[:4000])
yp = clf.predict(X[4000:])
rmse = SVR().jduge(yp,y[4000:])
print rmse