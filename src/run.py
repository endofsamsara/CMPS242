import json_to_csv_converter
from PreProcessing import PreProcessing

json_to_csv_converter.get_review_rating("..\data\yelp_academic_dataset_review.json", 1000)
a = PreProcessing("..\data\yelp_academic_dataset_review_review.csv").pre_processing()
