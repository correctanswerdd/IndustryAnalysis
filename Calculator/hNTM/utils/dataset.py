import tensorflow as tf
import numpy as np
import sys, os, time
from gensim.corpora import Dictionary


# from Calculator.utils import load_filebase  # cannot import modules in Calculator
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../util')

class Dataset(object):
    def __init__(self, year, for_training, batch_size=None, vocab_size=2000, full="full",
                 file_base="../../10kdata/"):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.file_base = file_base
        self.full = full
        self.for_training = for_training
        self.load10k(year)

    def load10k(self, year):
        import pickle
        #         self.vocab = Dictionary.load('./10-k/russel3000/10k_vocab2000_{year}.dict'.format(year=year))
        #         data = pickle.load(open("./10-k/russel/10k_corpus_{year}.pkl".format(year=year), 'rb'))
        vocab_dir = self.file_base + f'vocab{self.vocab_size}{self.full}_{year}.dict'
        corpus_dir = self.file_base + f"corpus_{self.vocab_size}{self.full}_training_{year}.pkl"
        self.vocab = Dictionary.load(vocab_dir)
        data = pickle.load(open(corpus_dir, 'rb'))

        BOW = np.zeros([len(data), self.vocab_size])
        for i, bow in enumerate(data):
            for word_id, count in bow:
                BOW[i][word_id] = count

        if self.for_training:
            self.num_train_batch = int(len(data) * 0.7) // self.batch_size
            self.train_x_bow = BOW[:self.num_train_batch * self.batch_size]
            self.num_test_batch = len(data) // self.batch_size - self.num_train_batch
            self.test_x_bow = BOW[self.num_train_batch * self.batch_size:]
        else:  # for k_means
            if int(len(data)) % self.batch_size == 0:
                self.num_train_batch = int(len(data)) // self.batch_size
                self.train_x_bow = BOW
                self.num_test_batch = int(len(data)) // self.batch_size
                self.test_x_bow = BOW
            else:
                # 补全最后一个batch
                comp_size = int(len(data)) % self.batch_size
                complementary = np.zeros((self.batch_size - comp_size, self.vocab_size))

                self.num_train_batch = int(len(data)) // self.batch_size + 1
                self.train_x_bow = np.vstack((BOW, complementary))

                self.num_test_batch = self.num_train_batch
                self.test_x_bow = self.train_x_bow
