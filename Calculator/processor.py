#!/usr/bin/env python3

from gensim.corpora import Dictionary
import gensim.downloader
from gensim import models, similarities
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from Calculator.io import read_csv, save_pkl, save_csv, read_pkl
from Calculator.utils import load_filebase, tokenize, paragraph_embedding


def doc2token(df):
    documents = df["document"].values
    token_list = []
    pbar = tqdm(documents, desc="Document to Tokens", leave=False)
    for i, doc in enumerate(pbar):
        pbar.set_description("Processing {i}".format(i=i))
        pbar.refresh()
        tokens = tokenize(doc)
        token_list.append(tokens)
    return token_list


def doc2token_filter(df, word_lst):
    documents = df["document"].values
    token_list = []
    pbar = tqdm(documents, desc="Document to Tokens", leave=False)
    for i, doc in enumerate(pbar):
        pbar.set_description("Processing {i}".format(i=i))
        pbar.refresh()
        tokens = tokenize(doc)
        tokens = [w for w in tokens if w in word_lst]
        token_list.append(tokens)
    return token_list


def doc2tfidf(df, extreme):
    """
    document -> tokens -> tfidf vector
    """
    doc2tokenn = doc2token(df)
    dictionary = Dictionary(doc2tokenn)
    dictionary.filter_extremes(keep_n=extreme)
    corpus = [dictionary.doc2bow(text) for text in doc2tokenn]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf


def wordcount2embedding(df, extreme):
    doc2tokenn = doc2token(df)
    dictionary = Dictionary(doc2tokenn)
    dictionary.filter_extremes(keep_n=extreme)
    corpus = [dictionary.doc2bow(text) for text in doc2tokenn]
    embedding = []
    for doc in corpus:
        vector = np.zeros(extreme)
        for i, count in doc:
            vector[i] = count
        embedding.append(vector)
    return embedding


def trainingset_generation_doc2vec(year, full_tokens):
    if full_tokens:
        full = "full"
        df = read_csv(type="index2file", year=year)
    else:
        full = "512"
        df = read_csv("documents_paragraphs", year=year)
        df = df[["cik", "short_file"]].copy()
        df = df.rename(columns={"short_file": "document"})
    doc2tokenn = doc2token(df)
    filename = load_filebase() + f"doc2vec_{full}_training_{year}.pkl"
    save_pkl(filename, doc2tokenn)


def trainingset_generation_doc2vec_filter(year, full_tokens, extreme):
    if full_tokens:
        full = "full"
        df = read_csv(type="index2file", year=year)
    else:
        full = "512"
        df = read_csv("documents_paragraphs", year=year)
        df = df[["cik", "short_file"]].copy()
        df = df.rename(columns={"short_file": "document"})
    dictionary = Dictionary.load(load_filebase() + f'vocab{extreme}{full}_{year}.dict')
    word_lst = dictionary.token2id.keys()
    doc2tokenn = doc2token_filter(df, word_lst)
    filename = load_filebase() + f"doc2vec_{extreme}{full}_training_{year}.pkl"
    save_pkl(filename, doc2tokenn)


def trainingset_generation_vocab_corpus(year, extreme, full_tokens):
    if full_tokens:
        full = "full"
        df = read_csv(type="index2file", year=year)
    else:
        full = "512"
        df = read_csv("documents_paragraphs", year=year)
        df = df[["cik", "short_file"]].copy()
        df = df.rename(columns={"short_file": "document"})
    doc2tokenn = doc2token(df)
    dictionary = Dictionary(doc2tokenn)
    dictionary.filter_extremes(keep_n=extreme)
    corpus = [dictionary.doc2bow(text) for text in doc2tokenn]
    dictionary.save(load_filebase() + f'vocab{extreme}{full}_{year}.dict')
    filename = load_filebase() + f"corpus_{extreme}{full}_training_{year}.pkl"
    save_pkl(filename, corpus)


def trainingset_generation_wordcount(year, extreme, full_tokens):
    if full_tokens:
        full = "full"
        df = read_csv(type="index2file", year=year)
    else:
        full = "512"
        df = read_csv("documents_paragraphs", year=year)
        df = df[["cik", "short_file"]].copy()
        df = df.rename(columns={"short_file": "document"})
    embedding = wordcount2embedding(df=df, extreme=extreme)
    filename = load_filebase() + f"wordcount_{extreme}{full}_{year}.pkl"
    save_pkl(filename, embedding)


def trainingset_generation_tfidf(year, extreme, full_tokens):
    if full_tokens:
        full = "full"
        df = read_csv(type="index2file", year=year)  # df: cik,
    else:
        full = "512"
        df = read_csv("documents_paragraphs", year=year)
        df = df[["cik", "short_file"]].copy()
        df = df.rename(columns={"short_file": "document"})
    corpus_tfidf = doc2tfidf(df, extreme=extreme)
    filename = load_filebase() + f"tfidf_{extreme}{full}_{year}.pkl"
    save_pkl(filename, corpus_tfidf)


def trainingset_generation(year, model):
    """
    format: model={name}_{extreme}{full}
    """
    if model == "doc2vec_512":
        trainingset_generation_doc2vec(year, False)
    elif model == "doc2vec_full":
        trainingset_generation_doc2vec(year, True)
    elif model == "doc2vec_2000full":
        trainingset_generation_doc2vec_filter(year, True, 2000)
    elif model == "lda_2000512":  # 512 tokens
        trainingset_generation_vocab_corpus(year, 2000, False)
    elif model == "lda_2000full":  # full tokens
        trainingset_generation_vocab_corpus(year, 2000, True)
    elif model == "hntm_2000full":  # full tokens
        trainingset_generation_vocab_corpus(year, 2000, True)
    elif model == "hntm_20000full":  # full tokens
        trainingset_generation_vocab_corpus(year, 20000, True)
    elif model == "hntm_2000512":  # 512 tokens
        trainingset_generation_vocab_corpus(year, 2000, False)
    elif model == "wordcount_2000full":
        trainingset_generation_wordcount(year, 2000, True)
    elif model == "wordcount_20000full":
        trainingset_generation_wordcount(year, 20000, True)
    elif model == "wordcount_2000512":
        trainingset_generation_wordcount(year, 2000, False)
    elif model == "wordcount_20000512":
        trainingset_generation_wordcount(year, 20000, False)
    elif model == "bowtfidf_2000512":
        trainingset_generation_tfidf(year, 2000, True)
    elif model == "bowtfidf_2000512":  # 512 tokens
        trainingset_generation_tfidf(year, 2000, False)
    elif model == "bowtfidf_20000512":  # 512 tokens
        trainingset_generation_tfidf(year, 20000, False)
    else:
        print(f"model_{model} need not have training set.")
    print("dataset generation finished!")
