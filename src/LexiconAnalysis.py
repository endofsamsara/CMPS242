import codecs
import numpy as np

from nltk.stem.wordnet import WordNetLemmatizer


class LexiconAnalysis:
    negation_begins = {'not', 'no', 'none', 'nothing', 'nobody', 'nowhere',
                       'few', 'little', 'hardly', 'rarely', 'scarcely', 'seldom',
                       'neither', 'nor', 'never'}
    sentence_ends = {'.', ';', '!', '?', '...'}

    def __init__(self, mask=0):
        self.mask = mask
        print 'loading corpus'
        wnl = WordNetLemmatizer()
        if not self.mask & 0b1:
            self.nrc = {}
            with codecs.open("../corpus/Yelp-restaurant-reviews-AFFLEX-NEGLEX-unigrams.txt", 'r', 'utf-8') as nrc_corpus:
                for line in nrc_corpus:
                    line = line.strip()
                    items = line.split('\t')
                    lexicon = items[0].split('_')
                    temp = lexicon[0].find("'")
                    if temp > 0:
                        lexicon[0] = lexicon[0][0:temp]
                    lexicon[0] = wnl.lemmatize(lexicon[0])
                    temp = '_'.join(lexicon)
                    self.nrc[temp] = float(items[1])

        if not self.mask & 0b10:
            self.mqs = {}
            with codecs.open("../corpus/subjclueslen1-HLTEMNLP05.tff", 'r', 'utf-8') as mqs_corpus:
                for line in mqs_corpus:
                    line = line.strip()
                    #items = line.split('\t')
                    items = line.split(' ')

                    #make sure starts from index 6
                    items[2] = items[2][6:]
                    if items[0] == "type=strongsubj":
                        if items[5] == "priorpolarity=positive":
                            self.mqs[items[2]] = 1.0
                        elif items[5] == "priorpolarity=negative":
                            self.mqs[items[2]] = -1.0
                        elif items[5] == "priorpolarity=neutral":
                            self.mqs[items[2]] = 0.0
                        else:
                            self.mqs[items[2]] = 2

                    if items[0] == "type=weaksubj":
                        if items[5] == "priorpolarity=positive":
                            self.mqs[items[2]] = 0.5
                        elif items[5] == "priorpolarity=negative":
                            self.mqs[items[2]] = -0.5
                        elif items[5] == "priorpolarity=neutral":
                            self.mqs[items[2]] = 0.0
                        else:
                            self.mqs[items[2]] = 2



    def generate(self, review):
        tokens, sentences = self.marking(review)
        features = []
        if not self.mask & 0b1:
            scores = np.array([self.nrc[t] if t in self.nrc else 0 for t in tokens])
            total = float(scores.size)
            # most positive score
            features += [max(scores + [0])]
            # most negative score
            features += [-min(scores + [0])]
            # positive words proportion
            pc = sum(scores > 0)
            features += [pc / total]
            # negative words proportion
            nc = sum(scores < 0)
            features += [nc / total]
            # strong positive words proportion (>1)
            features += [sum(scores > 1) / total]
            # strong negative words proportion (<-1)
            features += [sum(scores < -1) / total]
            # avg score among those with a score
            features += [sum(scores) / (pc + nc)]
            # proportion of position sentences (that have positive total score)
            sscores = np.array([np.sum(scores[sentences[x]:sentences[x + 1]]) for x in range(len(sentences) - 1)])
            stotal = float(sscores.size)
            features += [sum(sscores > 0) / stotal]
            # proportion of negative sentences (that have positive total score)
            features += [sum(sscores < 0) / stotal]

        if not self.mask & 0b10:
            new_tokens = np.array([token.split('_')[0] for token in tokens])

            scores = np.array([self.mqs[t] if t in self.mqs and self.mqs[t] < 2 else 0 for t in new_tokens])
            total = float(scores.size)
            # most positive score
            #features += [max(scores + [0])]
            # most negative score
            #features += [-min(scores + [0])]
            # positive words proportion
            pc = sum(scores > 0)
            features += [pc / total]
            # negative words proportion
            nc = sum(scores < 0)
            features += [nc / total]
            # strong positive words proportion (>1)
            features += [sum(scores == 1.0) / total]
            # strong negative words proportion (<-1)
            features += [sum(scores == -1.0) / total]
            # avg score among those with a score
            features += [sum(scores) / (pc + nc)]
            # proportion of position sentences (that have positive total score)
            sscores = np.array([np.sum(scores[sentences[x]:sentences[x + 1]]) for x in range(len(sentences) - 1)])
            stotal = float(sscores.size)
            features += [sum(sscores > 0) / stotal]
            # proportion of negative sentences (that have positive total score)
            features += [sum(sscores < 0) / stotal]

        return features

    def marking(self, review):
        state = 0
        tokens = review[:]
        sentences = [0]
        for i in range(len(tokens)):
            if tokens[i] in LexiconAnalysis.sentence_ends or i == len(tokens) - 1:
                if i > 1 and i - 1 in sentences:
                    sentences.remove(i - 1)
                sentences += [i]
                state = 0

            if state == 0 and tokens[i] in LexiconAnalysis.negation_begins:
                state = 1
            elif state == 1:
                tokens[i] += '_NEGFIRST'
                state = 2
            elif state == 2:
                tokens[i] += '_NEG'
        return tokens, sentences


