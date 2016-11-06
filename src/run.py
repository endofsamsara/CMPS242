import codecs

import json_to_csv_converter
from PreProcessing import PreProcessing
from LexiconAnalysis import LexiconAnalysis
from VectorWriter import VectorWriter

json_to_csv_converter.get_review_rating("..\data\yelp_academic_dataset_review.json", 1000)
a = PreProcessing("..\data\yelp_academic_dataset_review_review.txt").pre_processing()
la = LexiconAnalysis()
with codecs.open('..\data\yelp_academic_dataset_review_review_processed.txt', 'r', 'utf-8') as data, codecs.open(
        '..\data\yelp_academic_dataset_review_rating.txt', 'r', 'utf-8') as label, VectorWriter(
        '..\data\\train_vectors.txt') as out:
    for line in data:
        line = line.strip()
        rating = label.readline()
        if not line:
            continue
        tokens = line.split(" ")
        vector = la.generate(tokens)
        rating = rating.strip()
        rating = float(rating)
        out.write_row(rating, vector)
