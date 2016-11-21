import codecs
from sklearn import datasets

import json_to_csv_converter
from PreProcessing import PreProcessing
from LexiconAnalysis import LexiconAnalysis
from VectorWriter import VectorWriter
from SVR import SVR

# json_to_csv_converter.get_review_rating("..\data\yelp_academic_dataset_review.json", 1000)
# a = PreProcessing("..\data\yelp_academic_dataset_review_review.txt").pre_processing()
# la = LexiconAnalysis()
# with codecs.open('..\data\yelp_academic_dataset_review_review_processed.txt', 'r', 'utf-8') as data, codecs.open(
#         '..\data\yelp_academic_dataset_review_rating.txt', 'r', 'utf-8') as label, VectorWriter(
#         '..\data\\train_vectors.txt') as out:
#     for line in data:
#         line = line.strip()
#         rating = label.readline()
#         if not line:
#             continue
#         tokens = line.split(" ")
#         vector = la.generate(tokens)
#         rating = rating.strip()
#         rating = float(rating)
#         out.write_row(rating, vector)
X, y = datasets.load_svmlight_file('../data/train_vectors.txt')
model = SVR()
model.fit(X[:90],y[:90])
yp = model.predict(X[90:])
rmse = model.jduge(yp,y[90:])
print yp,y[90:]
print rmse
