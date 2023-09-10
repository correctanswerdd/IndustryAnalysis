from gensim.corpora.dictionary import Dictionary
from gensim.utils import tokenize
import pandas as pd
import numpy as np
import re, sys, os

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../util')

train_yelp = '../HAN+sentLDA/clf_datasets/yelp_review_full_2015_csv/train.csv'
test_yelp = '../HAN+sentLDA/clf_datasets/yelp_review_full_2015_csv/test.csv'
vocab_yelp = '../MyLVAE/yelp/vocab2000.gensim'

class MyTokens(object):
    def __init__(self, FilePath, MaxIt = -1):
        self.FilePath = FilePath
        self.MaxIt = MaxIt
        self.y = []
    def __iter__(self):
        reader = pd.read_csv(self.FilePath, header=0, names=['rating', 'review'], iterator=True, chunksize=1)
        i = -1
        p = re.compile('\\\\\w')
        for row in reader:
            if self.MaxIt > 0  and i > self.MaxIt: 
                break
            else: 
                i += 1
                try: 
                    text = p.sub(' ', row['review'][i])
                    if len(text) > 30:
                        yield tokenize(text, lowercase=True)
                        self.y.append(row['rating'][i])
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
        # else:
        #     wholebatch = []
        #     for doc in self.token:
        #         bow = self.dictionary.doc2bow(doc)
        #         doc = np.zeros([self.batch_size, self.vocab_size])
        #         for word_id, count in bow:
        #             batch[j][word_id] = count


if __name__ == '__main__':
    t = MyTokens(FilePath = train_yelp, MaxIt = -1)
    # dictionary = Dictionary(t)
    # print('original dictionary size: ', len(dictionary))
    # dictionary.save('../yelp/fullvocab.gensim')

    dictionary = Dictionary.load('../MyLVAE/yelp/fullvocab.gensim')

    from nltk.corpus import stopwords
    stopWords = set(stopwords.words('english'))
    print(stopWords)

    bad_ids = []
    for w in stopWords:
        try:
            bad_ids.append(dictionary.token2id[w])
        except:
            continue

    dictionary.filter_tokens(bad_ids=bad_ids)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=2000, keep_tokens=None)
    print('filtered dictionary size: ', len(dictionary))
    dictionary.save_as_text('../MyLVAE/yelp/vocab2000.txt')
    dictionary.save('../MyLVAE/yelp/vocab2000.gensim')