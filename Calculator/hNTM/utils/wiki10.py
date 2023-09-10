import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

MAX_NUM_WORD = 1000
EMBEDDING_SIZE = 100
num_classes = 73

def read_vector(filename):
    wordVectors = []
    vocab = []
    fileObject = open(filename, 'r')
    for i, line in enumerate(fileObject):
        if i==0 or i==1: # first line is a number (vocab size)
            continue
        line = line.strip()
        word = line.split()[0]
        vocab.append(word)
        wv_i = []
        for j, vecVal in enumerate(line.split()[1:]):
            wv_i.append(float(vecVal))
        wordVectors.append(wv_i)
    wordVectors = np.asarray(wordVectors)
    vocab_dict = dict(zip(vocab, range(1, len(vocab)+1))) # no 0 id; saved for padding
    print("Vectors read from: "+filename)
    return wordVectors, vocab_dict, vocab

def load_bow_by_batch(batch_ids, reduced_vocab, x_path):
    docs = []
    for i, ID in enumerate(batch_ids):
        with open(x_path+'/'+ID) as r_f:
            text = r_f.read() 
            docs.append(text.strip())
    count_vect = CountVectorizer(vocabulary=reduced_vocab)
    bow = count_vect.transform(docs).toarray()
    return bow

def load_rnn_data_by_batch(batch_ids, vocab_dict, x_path):
    X = np.zeros((len(batch_ids), MAX_NUM_WORD), dtype=np.int32)
    for i, ID in enumerate(batch_ids):
        with open(x_path+'/'+ID) as r_f:
            text = r_f.read() 
            doc = [vocab_dict[w] for w in text.split() if w in vocab_dict]
            if len(doc)>MAX_NUM_WORD:
                X[i, :] = doc[:MAX_NUM_WORD]
            else:
                X[i, :len(doc)] = doc
    return X