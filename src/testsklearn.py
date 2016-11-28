from sklearn import datasets,svm

from SVR import SVR

X, y = datasets.load_svmlight_file('../data/train_vectors.txt')
clf = svm.SVR()
clf.fit(X[:90],y[:90])
yp = clf.predict(X[90:])
rmse = SVR().jduge(yp,y[90:])
print yp,y[90:]
print rmse