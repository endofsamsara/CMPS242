from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


class PreProcessing:
    def __init__(self):
        pass

    _in_file_name, _out_file_name = "in.csv", "out.csv"

    _partial_list = {
        "n't": "not",
        "'m": "am",
        "'s": "is",
        "'re": "are",
        "'ll": "will",
        "'ve": "have",
        "ca": "can",
        "wo": "will"
    }

    # run the whole process of pre processing
    def pre_processing(self):
        with open(self._in_file_name, "r") as raw, open(self._out_file_name, "w") as out:
            for line in raw:
                line = line.strip()
                var = self.tokenize(line)
                res = self.normalize(var)
                out.write(" ".join(res)+"\n")
            out.flush()

    # split punctuations and abbreviations apart
    def tokenize(self, review):
        return [word.lower() for t in sent_tokenize(review) for word in word_tokenize(t)]

    # transfer a word to its original form
    # manually change some partial forms resulted from tokenizing to original form
    def normalize(self, words):
        wnl = WordNetLemmatizer()
        return [wnl.lemmatize(self._partial_list.get(x, x)) for x in words]

