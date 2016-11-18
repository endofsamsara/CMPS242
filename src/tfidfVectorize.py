import codecs
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

class tfidfVectorizer:
    def __init__(self, input_name="in.txt"):
        self._in_file_name = input_name

    def tfidf_vectorize(self):
        #this funtion turn target txt.file to a sparse matrix res in csr format
        with codecs.open(self._in_file_name, "r", "utf-8") as raw:
            res = self.spa(var)

    def fit(self, file):
        vectorized = TfidfVectorizer()
        vectorized.min_df = 1
        matrix = vectorized.fit_transform(file)
        return matrix.toarray()

    def spa(self, vec):
        print(sparse.csr_matrix(vec))
        return [sparse.csr_matrix(vec)]

