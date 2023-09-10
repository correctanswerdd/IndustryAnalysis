import tensorflow as tf
import numpy as np
import sys, os, time
from _20news import *
from yelp import *
import wiki10
import MRD

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../util')

class Dataset(object):

    def __init__(self, dataset, batch_size = 64, max_word_num = 500):
        self.dataset = dataset
        self.batch_size = batch_size 
        self.max_word_num = max_word_num
        self.embedding_size = 100
        if self.dataset == "20news":
            self.load20news()
        elif self.dataset == "yelp":
            self.loadyelp()
        elif self.dataset == "wiki10":
            self.loadwiki()
        elif self.dataset == "wiki10_stemmed":
            self.loadwiki(stemmed = True)
        elif self.dataset == "MRD":
            self.loadMRD()
    
    def load20news(self, train=True, test=True):
        self.pretrained_embed, vocab_dict, self.vocab = read_vector(
            "../HNTM/data/preprocessed/embedding/20newsVec.txt")
        self.vocab_size = len(self.vocab)
        self.train_x, self.train_y, _, self.num_train_docs = read_topical_atten_data(
            "../HNTM/data/preprocessed/train-processed.tab", vocab_dict, self.vocab)
        self.num_train_batch = self.num_train_docs//self.batch_size
        self.test_x, self.test_y, _, self.num_test_docs = read_topical_atten_data(
            "../HNTM/data/preprocessed/test-processed.tab", vocab_dict, self.vocab)
        self.num_test_batch = self.num_test_docs//self.batch_size

    def loadyelp(self, train=True, test=True):
        self.dictionary = Dictionary.load(vocab_yelp)
        self.vocab_size = len(self.dictionary)
        if train and test:
            self.train_gen = BatchGnerator(train_yelp, self.dictionary, self.batch_size)
            self.train_y = self.train_gen.token.y
            self.test_gen = BatchGnerator(test_yelp, self.dictionary, self.batch_size)
            self.test_y = self.test_gen.token.y

    def loadwiki(self, stemmed = False, train=True, test=True):
        # load data ids and labels
        self.train_ids = np.load('data/wiki10/train_ids.npy')
        self.test_ids = np.load('data/wiki10/test_ids.npy')
        if stemmed:
            text_path = 'data/wiki10/stemmed_text'
            vocab_path = 'data/wiki10/stemmed_vocab_10k.txt'
        else:
            text_path = 'data/wiki10/text'
            vocab_path = 'data/wiki10/topic_model_vocab_20000.txt'
        # load vocab
        self.vocab = [] 
        with open(vocab_path) as r_f:
            for i, l in enumerate(r_f):
                self.vocab.append(l.strip())
                if i == self.vocab_size-1:
                    break
        self.num_train_batch = len(self.train_ids)//self.batch_size
        self.num_test_batch = len(self.test_ids)//self.batch_size
        def load_batch(batch_ids):
            return wiki10.load_bow_by_batch(batch_ids, self.vocab, text_path)
        self.load_batch = load_batch
    
    def loadMRD(self):
        _, _, _, self.vocab, _, _, self.train_x_bow, _, _, self.test_x_bow, _, _ = MRD.load_data()
        self.num_train_batch = len(self.train_x_bow)//self.batch_size
        self.num_test_batch = len(self.test_x_bow)//self.batch_size



