from gensim.corpora.dictionary import Dictionary
from gensim.utils import tokenize
import numpy as np
import re, sys, os

abstract = '../baselines/wikipediacompanycorpus/download_corpus/abstract.pkl'
vocab= '../baselines/wikipediacompanycorpus/download_corpus/vocab2000_abstarct.gensim'

class MyTokens(object):
    def __init__(self, FilePath, MaxIt = -1):
        self.FilePath = FilePath
        self.MaxIt = MaxIt
    def __iter__(self):
        reader = open(abstract)
        i = 0
        p = re.compile('\\\\\w')
        for line in reader:
            if self.MaxIt > 0  and i > self.MaxIt: 
                break
            else: 
                i += 1
                try: 
                    text = p.sub(' ', line)
                    if len(text) > 30:
                        yield tokenize(text, lowercase=True)
                    else:
                        continue
                except:
                    continue

class MyCorpus(object):
    def __init__(self, dictionary, TokenGenerator):
        self.dictionary = dictionary
        self.TokenGenerator = TokenGenerator
    def __iter__(self):
        for doc in self.TokenGenerator:
            yield self.dictionary.doc2bow(doc)

class BatchGnerator(object):
    def __init__(self, dataPath, dictionary, batch_size):
        self.dictionary = dictionary
        self.token = MyTokens(FilePath = dataPath)
        self.vocab_size = len(dictionary)
        self.batch_size = batch_size
    
    def __iter__(self):
        # if self.batch_size > 0:
        i = -1
        for doc in self.token:
            bow = self.dictionary.doc2bow(doc)
            i += 1
            j = i%self.batch_size
            if j == 0:
                batch = np.zeros([self.batch_size, self.vocab_size])
            for word_id, count in bow:
                batch[j][word_id] = count
            if j == self.batch_size-1:
                yield batch
            else:
                continue