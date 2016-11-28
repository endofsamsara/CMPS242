import codecs

import numpy as np

import json_to_csv_converter
from PreProcessing import PreProcessing
from LexiconAnalysis import LexiconAnalysis
from VectorWriter import VectorWriter
from tfidfVectorize import tfidfVectorizer

json_to_csv_converter.get_review_rating("..\data\yelp_academic_dataset_review.json", 5000)
a = PreProcessing("..\data\yelp_academic_dataset_review_review.txt").pre_processing()
la = LexiconAnalysis()
bow = tfidfVectorizer('../data/yelp_academic_dataset_review_review_processed.txt').tfidf_vectorize()[0]
with codecs.open('..\data\yelp_academic_dataset_review_review_processed.txt', 'r', 'utf-8') as data, codecs.open(
        '..\data\yelp_academic_dataset_review_rating.txt', 'r', 'utf-8') as label, VectorWriter(
    '..\data\\train_vectors.txt') as out:
    c = 0
    for line in data:
        line = line.strip()
        rating = label.readline()
        if not line:
            continue
        tokens = line.split(" ")
        vector = la.generate(tokens)
        vector = np.append(vector, bow[c,:].A)
        c+=1
        rating = rating.strip()
        rating = float(rating)
        out.write_row(rating, vector)

