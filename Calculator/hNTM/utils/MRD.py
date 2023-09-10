import pickle
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

MAX_NUM_WORD = 500
EMBEDDING_SIZE = 100

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

def read_corpus(x_path, y_path, vocab_dict, reduced_vocab, num_docs):
    y = np.load(y_path)
    X = np.zeros((num_docs, MAX_NUM_WORD), dtype=np.int32)
    docs = []
    with open(x_path) as r_f:
        for i,l in enumerate(r_f):
            doc = [vocab_dict[w] for w in l.split() if w in vocab_dict]
            if len(doc)>MAX_NUM_WORD:
                X[i, :] = doc[:MAX_NUM_WORD]
            else:
                X[i, :len(doc)] = doc
            docs.append(l.strip())
    count_vect = CountVectorizer(vocabulary=reduced_vocab)
    bow = count_vect.transform(docs).toarray()
    return X, y, bow

def load_data(scaling=False):
    wv_matrix, vocab_dict, vocab = read_vector(
        "../TAM_AISTATS2020/NTM_attention/movie_review_data/scaledata/processed/MRDscale_Vec.txt")
    vocab_size = len(vocab)
    print("original vocab size: ", vocab_size)

    reduced_vocab = []
    with open("../TAM_AISTATS2020/NTM_attention/movie_review_data/scaledata/processed/topic_model_vocab.txt") as r_f:
        for line in r_f:
            reduced_vocab.append(line.strip())
    
    train_x_rnn, train_y, train_x_bow = read_corpus(
        "../TAM_AISTATS2020/NTM_attention/movie_review_data/scaledata/processed/train_corpus.txt", 
        '../TAM_AISTATS2020/NTM_attention/movie_review_data/scaledata/processed/train_y.npy', 
        vocab_dict, reduced_vocab, 3337)
    
    test_x_rnn, test_y, test_x_bow = read_corpus(
        "../TAM_AISTATS2020/NTM_attention/movie_review_data/scaledata/processed/test_corpus.txt", 
        '../TAM_AISTATS2020/NTM_attention/movie_review_data/scaledata/processed/test_y.npy', 
        vocab_dict, reduced_vocab, 1669)
    
    if scaling:
        scaler = StandardScaler()
        train_y = np.squeeze(scaler.fit_transform(np.expand_dims(train_y, -1)))
        test_y = np.squeeze(scaler.transform(np.expand_dims(test_y, -1)))
        mean = scaler.mean_[0]
        std = scaler.scale_[0]
    else:
        mean = 0.0
        std = 1.0

    return wv_matrix, vocab_dict, vocab, reduced_vocab, train_x_rnn, train_y, train_x_bow, test_x_rnn, test_y, test_x_bow, mean, std
    

       