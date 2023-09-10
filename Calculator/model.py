import gensim
import gensim.downloader
import numpy as np
from nltk import sent_tokenize
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from pathlib import Path
from gensim.test.utils import get_tmpfile
from sentence_transformers import SentenceTransformer, InputExample, datasets, losses
from re import compile, sub
from gensim.corpora import Dictionary
from gensim.models import ldamodel
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, AdamW, BertModel, RobertaTokenizer, RobertaModel, \
    LongformerTokenizer, LongformerModel, RobertaForMaskedLM
import tensorflow as tf
import sys, os, time

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '../hNTM') 失败
from .hNTM.model import TopicModel
from .hNTM.utils.dataset import Dataset
from .hNTM.util import read_configuration

from Calculator.io import read_csv, read_pkl, save_pkl, save_csv
from Calculator.utils import load_filebase, tokenize, paragraph_embedding, topic_concatenate, get_split
from SimCSE.simcse import SimCSE


def doc2vec_get_dataset(data, tdmodel):
    x = []
    for i, text in enumerate(data):
        paragraph = tdmodel(text, tags=[i])
        x.append(paragraph)
    return x


def doc2vec_train(x_train, model_file_base, size=200):
    Path(model_file_base).mkdir(parents=True, exist_ok=True)
    model = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model.train(x_train, total_examples=model.corpus_count, epochs=100)
    model.save(model_file_base + 'model_dm')
    return model


def doc2vec_inference(x_train, model_file_base):
    model = Doc2Vec.load(model_file_base + '/model_dm')
    infered_vectors_list = []
    for text, label in tqdm(x_train, leave="True", desc=f"doc2vec embedding"):
        infered_vectors_list.append(model.infer_vector(text))
    return infered_vectors_list


def SBERT_embedding(model, paragraph):
    # 删词规则
    r0 = "[^\x00-\x7f]"

    # filter out the paragraph
    para = sub(r0, '', paragraph)
    return model.encode(para, show_progress_bar=False)


def longformer_embedding(model, tokenizer, text):
    """
    * last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) – Sequence of hidden-states at the output of the last layer of the model.
    * pooler_output (torch.FloatTensor of shape (batch_size, hidden_size)) – Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    """
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    return output.pooler_output.cpu().detach().numpy()


def SBERT_inference(documents):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embedding = []
    for doc in tqdm(documents, leave="True", desc=f"sbert embedding"):
        paras = doc.split("\n")
        doc_embedding = []
        for para in paras:
            tokens = tokenize(para)
            if tokens and para and len(tokens) > 20:
                doc_embedding.append(SBERT_embedding(model, para))
        if not doc_embedding:
            # plan B
            r0 = "[\n]"
            doc_ = sub(r0, '', doc)
            paras = doc_.split("\n")
            doc_embedding = []
            for para in paras:
                tokens = tokenize(para)
                if tokens and para and len(tokens) > 20:
                    doc_embedding.append(SBERT_embedding(model, para))

        assert doc_embedding
        doc_embedding = np.array(doc_embedding)
        doc_embedding = np.mean(doc_embedding, axis=0)
        embedding.append(doc_embedding)
    return embedding


def SBERT_inference_v2(documents):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embedding = []
    for doc in tqdm(documents, leave="True", desc=f"sbert embedding"):
        sents_embedding = []
        sents = sent_tokenize(doc)
        for s in sents:
            sents_embedding.append(SBERT_embedding(model, s))
        doc_embedding = np.array(sents_embedding)
        doc_embedding = np.mean(doc_embedding, axis=0)
        embedding.append(doc_embedding)
    return embedding


def SBERT_inference_v3(documents):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embedding = []
    for doc in tqdm(documents, desc="SBERT512 embedding", leave=True):
        embedding.append(model.encode(doc, show_progress_bar=False))
    return embedding


def SBERT_inference_v4(documents, model):
    embedding = []
    for doc in tqdm(documents, desc="SBERT512 embedding", leave=True):
        embedding.append(model.encode(doc, show_progress_bar=False))
    return embedding


def lda_inference(model, corpus, num_topics):
    topics = []
    for i in range(len(corpus)):
        doc_topics, word_topics, phi_values = model.get_document_topics(corpus[i], per_word_topics=True)
        dist = np.zeros(num_topics)
        for j in range(num_topics):
            t, p = doc_topics[j]
            dist[j] = p
        topics.append(dist)
    return topics


def simcse_inference(documents, model):
    embedding = []
    for doc in tqdm(documents, leave="True", desc="SimCSE embedding"):
        embedding.append(model.encode(doc).cpu().detach().numpy())
    return embedding


def word2vec_inference(documents, model):
    embedding = []
    for doc in tqdm(documents, leave="True", desc="word2vec embedding"):
        tokens = tokenize(doc)
        para_embed = paragraph_embedding(tokens, model)
        embedding.append(para_embed)
    return embedding


def word2vec_inference_filter(documents, model, word_lst):
    embedding = []
    for doc in tqdm(documents, leave="True", desc="word2vec embedding"):
        tokens = tokenize(doc)
        tokens = [w for w in tokens if w in word_lst]
        para_embed = paragraph_embedding(tokens, model)
        embedding.append(para_embed)
    return embedding


def bert_inference(documents, name):
    print(f"model is {name}.")
    if name == "bert":
        model = BertModel.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif name == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
    elif name == "longformer":
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        embedding = []
        for doc in tqdm(documents, desc="longformer embedding", leave=True):
            encoded_input = tokenizer(doc, return_tensors='pt', padding=True, truncation=True, max_length=4096,
                                      add_special_tokens=True)
            output = model(**encoded_input)
            embedding.append(np.squeeze(output["pooler_output"].cpu().detach().numpy()))
        return embedding
    elif "roberta" in name:
        model = RobertaModel.from_pretrained(name)
        tokenizer = RobertaTokenizer.from_pretrained(name)
    elif "bert" in name:
        model = BertModel.from_pretrained(name)
        tokenizer = BertTokenizer.from_pretrained(name)
    else:
        return
    embedding = []
    for doc in tqdm(documents, leave="True", desc=f"{name} embedding"):
        encoded_input = tokenizer(doc, return_tensors='pt', padding=True, truncation=True, max_length=512,
                                  add_special_tokens=True)
        output = model(**encoded_input)
        embedding.append(np.squeeze(output["pooler_output"].cpu().detach().numpy()))
    return embedding


def bert_split_inference(data):
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    embedding = []
    for chunks in tqdm(data, leave="True", desc=f"full bert embedding"):
        doc_embedding = []
        for piece in chunks:
            encoded_input = tokenizer(piece, return_tensors='pt', padding=True, truncation=True, max_length=512,
                                      add_special_tokens=True)
            output = model(**encoded_input)
            doc_embedding.append(np.squeeze(output["pooler_output"].cpu().detach().numpy()))
        embedding.append(np.mean(doc_embedding, axis=0))
    return embedding


def hntm_inference(year, extreme, full):
    FILE_OF_CKPT = os.path.join(load_filebase(), f"{year}_hntm_{extreme}{full}_out/best-model.ckpt")
    parameters = read_configuration(
        file_name=os.path.join(load_filebase(), f"{year}_hntm_{extreme}{full}_out/config.ini")
    )
    d = Dataset(year=year, for_training=False, batch_size=parameters["batch_size"], full=full,
                file_base=load_filebase(), vocab_size=extreme)
    with tf.Graph().as_default() as g:
        ###########################################
        """        Build Model Graphs           """
        ###########################################
        with tf.variable_scope("topicmodel") as scope:
            m = TopicModel(d,
                           latent_sizes=parameters["num_topics"],
                           layer_sizes=parameters["layer_sizes"],
                           embedding_sizes=parameters["embedding_sizes"])
            print('built the graph for training.')
            scope.reuse_variables()

        ###########################################
        """              Init                   """
        ###########################################
        init_op = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init_op)
        saver = tf.train.Saver()

        print("... restore from the last check point.")
        saver.restore(sess, FILE_OF_CKPT)

        topic_distributions = []
        X_train = d.train_x_bow

        for i in range(d.num_train_batch):
            batch_topic_distributions = sess.run([m.topic_dist],
                                                 feed_dict={m.x: X_train[i * d.batch_size: (i + 1) * d.batch_size],
                                                            m.is_train: False})
            new_topic_distributions = topic_concatenate(batch_topic_distributions, parameters["batch_size"])
            topic_distributions.extend(new_topic_distributions)
        sess.close()

    return topic_distributions


def training_doc2vec(year, full):
    filename = load_filebase() + f"doc2vec_{full}_training_{year}.pkl"
    data = read_pkl(filename)
    TaggededDocument = gensim.models.doc2vec.TaggedDocument
    x_train = doc2vec_get_dataset(data, TaggededDocument)
    model_file_base = load_filebase() + f"{year}_doc2vec_{full}/"
    doc2vec_train(x_train, model_file_base)


def training_doc2vec_filter(year, full, extreme):
    filename = load_filebase() + f"doc2vec_{extreme}{full}_training_{year}.pkl"
    data = read_pkl(filename)
    TaggededDocument = gensim.models.doc2vec.TaggedDocument
    x_train = doc2vec_get_dataset(data, TaggededDocument)
    model_file_base = load_filebase() + f"{year}_doc2vec_{extreme}{full}/"
    doc2vec_train(x_train, model_file_base)


def training_lda(year, n_topics, extreme, full_tokens):
    if full_tokens:
        full = "full"
    else:
        full = "512"
    dictionary = Dictionary.load(load_filebase() + f'vocab{extreme}{full}_{year}.dict')
    filename = load_filebase() + f"corpus_{extreme}{full}_training_{year}.pkl"
    data = read_pkl(filename)
    model = ldamodel.LdaModel(data,
                              id2word=dictionary,
                              num_topics=n_topics,
                              alpha='asymmetric',
                              minimum_probability=1e-8)
    filename = load_filebase() + f"{year}_lda{full}.model"
    model.save(filename)


def training_bert(year, name, tokenizer, model, batch_size):
    with open(f'SimCSE/data/one_line_doc_{year}.txt', 'r') as fp:
        text = fp.read().split("\n")
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    inputs['labels'] = inputs.input_ids.detach().clone()

    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
    selection = []
    for i in range(mask_arr.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )
    for i in range(mask_arr.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

    class ReportDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        def __len__(self):
            return len(self.encodings.input_ids)

    dataset = ReportDataset(inputs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    optim = AdamW(model.parameters(), lr=1e-5)
    epochs = 1

    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    model.save_pretrained(f"{name}_{year}")
    tokenizer.save_pretrained(f"{name}_{year}")


def training_sbert(year, batch_size):
    dataset = pd.read_csv(f'SimCSE/data/2col_for_simcse_{year}.csv')
    train_samples = []
    for idx, row in tqdm(dataset.iterrows()):
        train_samples.append(InputExample(
            texts=[row['texta'], row['textp']]
        ))
    loader = datasets.NoDuplicatesDataLoader(
        train_samples, batch_size=batch_size)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    loss = losses.MultipleNegativesRankingLoss(model)
    epochs = 1
    warmup_steps = int(len(loader) * epochs * 0.1)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=f'{load_filebase()}/{year}_sbert',
        show_progress_bar=False
    )


def training(model, year, batch_size=8):
    if model == "doc2vec_full":
        training_doc2vec(year, "full")
    elif model == "doc2vec_2000full":
        training_doc2vec_filter(year, "full", 2000)
    elif model == "doc2vec_512":
        training_doc2vec(year, "512")
    elif model == "lda_2000512":
        training_lda(year, 100, 2000, False)
    elif model == "lda_2000full":
        training_lda(year, 100, 2000, True)
    elif model == "bert-mlm":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        training_bert(year, "bert-mlm", tokenizer, model, batch_size)
    elif model == "roberta-mlm":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForMaskedLM.from_pretrained("roberta-base")
        training_bert(year, "roberta-mlm", tokenizer, model, batch_size)
    elif model == "sbert":
        training_sbert(year, batch_size)


def inference_doc2vec(year, full):
    model_file_base = load_filebase() + f"{year}_doc2vec_{full}/"
    filename = load_filebase() + f"doc2vec_{full}_training_{year}.pkl"
    data = read_pkl(filename)
    TaggededDocument = gensim.models.doc2vec.TaggedDocument
    x_train = doc2vec_get_dataset(data, TaggededDocument)
    embedding = doc2vec_inference(x_train, model_file_base)
    filename = load_filebase() + f"doc2vec_{full}_inference_{year}.pkl"
    save_pkl(filename, embedding)


def inference_doc2vec_filter(year, full, extreme):
    model_file_base = load_filebase() + f"{year}_doc2vec_{extreme}{full}/"
    filename = load_filebase() + f"doc2vec_{extreme}{full}_training_{year}.pkl"
    data = read_pkl(filename)
    TaggededDocument = gensim.models.doc2vec.TaggedDocument
    x_train = doc2vec_get_dataset(data, TaggededDocument)
    embedding = doc2vec_inference(x_train, model_file_base)
    filename = load_filebase() + f"doc2vec_{extreme}{full}_inference_{year}.pkl"
    save_pkl(filename, embedding)


def inference_sbert(year):
    data = read_csv("documents_paragraphs", year=year)
    embedding = SBERT_inference_v3(data["short_file"].values)
    filename = load_filebase() + f"sbert_inference_{year}.pkl"
    save_pkl(filename, embedding)


def inference_sbert_2col(year):
    data = read_csv("documents_paragraphs", year=year)
    model = SentenceTransformer(f"{load_filebase()}/{year}_sbert")
    embedding = SBERT_inference_v4(data["short_file"].values, model)
    filename = load_filebase() + f"sbert_2col_inference_{year}.pkl"
    save_pkl(filename, embedding)


def inference_sbert_2col_test(year):
    data = read_csv("documents_paragraphs_test", year=year)
    model = SentenceTransformer(f"{load_filebase()}/{year}_sbert")
    embedding = SBERT_inference_v4(data["short_file"].values, model)
    filename = load_filebase() + f"sbert_2col_test_inference_{year}.pkl"
    save_pkl(filename, embedding)


def inference_lda(year, n_topics, extreme, full_tokens):
    if full_tokens:
        full = "full"
    else:
        full = "512"
    filename = load_filebase() + f"{year}_lda{full}.model"
    model = ldamodel.LdaModel.load(filename)
    filename = load_filebase() + f"corpus_{extreme}{full}_training_{year}.pkl"
    data = read_pkl(filename)
    embedding = lda_inference(model, data, n_topics)
    filename = load_filebase() + f"lda_{extreme}{full}_inference_{year}.pkl"
    save_pkl(filename, embedding)


def inference_glove(year, model, full_tokens):
    if model == "glove":
        glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')
    elif model == "word2vec":
        glove_vectors = gensim.downloader.load('word2vec-google-news-300')
    else:
        print(f"wrong model_{model}!")
        return

    if full_tokens:
        full = "full"
        df = read_csv(type="index2file", year=year)  # df: cik, file_index
        documents = df["document"].values
    else:
        full = "512"
        data = read_csv(type="documents_paragraphs", year=year)
        documents = data["short_file"].values

    embedding = word2vec_inference(documents, glove_vectors)
    filename = load_filebase() + f"{model}_{full}_inference_{year}.pkl"
    save_pkl(filename, embedding)


def inference_glove_filter(year, model, full_tokens, extreme):
    if model == "glove":
        glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')
    elif model == "word2vec":
        glove_vectors = gensim.downloader.load('word2vec-google-news-300')
    else:
        print(f"wrong model_{model}!")
        return

    if full_tokens:
        full = "full"
        df = read_csv(type="index2file", year=year)  # df: cik, file_index
        documents = df["document"].values
    else:
        full = "512"
        data = read_csv(type="documents_paragraphs", year=year)
        documents = data["short_file"].values

    dictionary = Dictionary.load(load_filebase() + f'vocab{extreme}{full}_{year}.dict')
    word_lst = dictionary.token2id.keys()
    embedding = word2vec_inference_filter(documents, glove_vectors, word_lst)
    filename = load_filebase() + f"{model}_{extreme}{full}_inference_{year}.pkl"
    save_pkl(filename, embedding)


def inference_simcse(year, model, name):
    data = read_csv("documents_paragraphs", year=year)
    embedding = simcse_inference(data["short_file"].values, model)
    filename = load_filebase() + f"{name}_inference_{year}.pkl"
    save_pkl(filename, embedding)


def inference_simcse_test(year, model, name):
    data = read_csv("documents_paragraphs_test", year=year)
    embedding = simcse_inference(data["short_file"].values, model)
    filename = load_filebase() + f"{name}_inference_{year}.pkl"
    save_pkl(filename, embedding)


def inference_bert(year, model):
    data = read_csv("documents_paragraphs", year=year)  # cik, short_file
    embedding = bert_inference(data["short_file"].values, f"{model}_{year}")
    filename = load_filebase() + f"{model}_inference_{year}.pkl"
    save_pkl(filename, embedding)


def inference_bert_full(year, model):
    data = read_csv(type="index2file", year=year)[["cik", "document"]]
    data["document_split"] = data['document'].apply(get_split)
    embedding = bert_split_inference(data["document_split"].values)
    filename = load_filebase() + f"{model}_inference_{year}.pkl"
    save_pkl(filename, embedding)


def inference_hntm(year, extreme, full):
    embedding = hntm_inference(year, extreme, full)
    filename = load_filebase() + f"hntm_{extreme}{full}_inference_{year}.pkl"
    save_pkl(filename, embedding)


def inference(model, year):
    if model == "doc2vec_full":
        inference_doc2vec(year, "full")
    elif model == "doc2vec_2000full":
        inference_doc2vec_filter(year, "full", 2000)
    elif model == "doc2vec_512":
        inference_doc2vec(year, "512")
    elif model == "hntm_20000full":  # full
        inference_hntm(year, 20000, "full")
    elif model == "hntm_full":  # full
        inference_hntm(year, 2000, "full")
    elif model == "hntm_512":  # 512
        inference_hntm(year, 2000, "512")
    elif model == "longformer":  # 512
        inference_bert(year, model)
    elif model == "bert_full":
        inference_bert_full(year, model)
    elif model == "bert":
        inference_bert(year, model)
    elif model == "bert-mlm":
        inference_bert(year, model)
    elif model == "roberta-mlm":
        inference_bert(year, model)
    elif model == "roberta":
        inference_bert(year, model)
    elif model == "roberta-mlm":
        inference_bert(year, model)
    elif model == "sbert":
        inference_sbert(year)
    elif model == "sbert_2col":
        inference_sbert_2col(year)
    elif model == "sbert_2col_test":
        inference_sbert_2col_test(year)
    elif model == "lda_2000512":
        inference_lda(year, 100, 2000, False)
    elif model == "lda_2000full":
        inference_lda(year, 100, 2000, True)
    elif model == "glove_full":  # 完整文档
        inference_glove(year, "glove", True)
    elif model == "glove_2000full":  # 完整文档
        inference_glove_filter(year, "glove", True, 2000)
    elif model == "glove_512":  # 512
        inference_glove(year, "glove", False)
    elif model == "word2vec_512":  # 512
        inference_glove(year, "word2vec", True)
    elif model == "bert_r2":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}")
        inference_simcse(year, model, "bert_r2")
    elif model == "bert_r2_epoch":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_3epoch")
        inference_simcse(year, model, "bert_r2_epoch")
    elif model == "bert_r2_v2":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_v2")
        inference_simcse(year, model, "bert_r2_v2")
    elif model == "bert_baseline":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_baseline")
        inference_simcse(year, model, "bert_baseline")
    elif model == "bert_baseline_mlm":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_baseline_domlm")
        inference_simcse(year, model, "bert_baseline_mlm")
    elif model == "bert_domlm":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm")
        inference_simcse(year, model, "bert_domlm")
    elif model == "bert_contrastive":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_contrastive")
        inference_simcse(year, model, "bert_contrastive")
    elif model == "bert_contrastive_v2":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_contrastive_v2")
        inference_simcse(year, model, "bert_contrastive_v2")
    elif model == "bert_contrastive_v3":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_contrastive_v3")
        inference_simcse(year, model, "bert_contrastive_v3")
    elif model == "bert_2col":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col")
        inference_simcse(year, model, "bert_2col")
    elif model == "bert_2col_test":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col")
        inference_simcse_test(year, model, "bert_2col_test")
    elif model == "bert_2col_thsim098":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_highsimallr2_0.85_0.98")
        inference_simcse(year, model, "bert_2col_thsim098")
    elif model == "bert_2col_thsim097":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_highsimallr2_0.85_0.97")
        inference_simcse(year, model, "bert_2col_thsim097")
    elif model == "bert_2col_thsim095":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_highsimallr2_0.85_0.95")
        inference_simcse(year, model, "bert_2col_thsim095")
    elif model == "bert_2col_thsim093":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_highsimallr2_0.85_0.93")
        inference_simcse(year, model, "bert_2col_thsim093")
    elif model == "bert_2col_thsim09":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_highsimallr2_0.85_0.9")
        inference_simcse(year, model, "bert_2col_thsim09")
    elif model == "bert_2col_th095":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_th0.95")
        inference_simcse(year, model, "bert_2col_th095")
    elif model == "bert_2col_th09":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_th0.9")
        inference_simcse(year, model, "bert_2col_th09")
    elif model == "bert_2col_th087":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_th0.87")
        inference_simcse(year, model, "bert_2col_th087")
    elif model == "bert_2col_th083":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_th0.83")
        inference_simcse(year, model, "bert_2col_th083")
    elif model == "bert_2col_th08":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_th0.8")
        inference_simcse(year, model, "bert_2col_th08")
    elif model == "bert_2col_th075":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_th0.75")
        inference_simcse(year, model, "bert_2col_th075")
    elif model == "bert_2col_th07":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_th0.7")
        inference_simcse(year, model, "bert_2col_th07")
    elif model == "bert_2col_th04":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_th0.4")
        inference_simcse(year, model, "bert_2col_th04")
    elif model == "bert_2col_th01":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_th0.1")
        inference_simcse(year, model, "bert_2col_th01")
    elif model == "bert_2col_lseq512_lr5e-6":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_lseq512_lr5e-6")
        inference_simcse(year, model, "bert_2col_lseq512_lr5e-6")
    elif model == "bert_2col_lseq512_lr1e-6":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_lseq512_lr1e-6")
        inference_simcse(year, model, "bert_2col_lseq512_lr1e-6")
    elif model == "bert_2col_lr5e-6":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_lr5e-6")
        inference_simcse(year, model, "bert_2col_lr5e-6")
    elif model == "bert_2col_dropout0":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_dropout0")
        inference_simcse(year, model, "bert_2col_dropout0")
    elif model == "bert_2col_dropout0_lr5e-6":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_dropout0_lr5e-6")
        inference_simcse(year, model, "bert_2col_dropout0_lr5e-6")
    elif model == "bert_2col_allsim":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_allsim")
        inference_simcse(year, model, "bert_2col_allsim")
    elif model == "bert_2col_allsim09":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_allsim_th09")
        inference_simcse(year, model, "bert_2col_allsim09")
    elif model == "bert_2col_allsim093":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_allsim_th093")
        inference_simcse(year, model, "bert_2col_allsim093")
    elif model == "bert_2col_allsim095":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_allsim_th095")
        inference_simcse(year, model, "bert_2col_allsim095")
    elif model == "bert_2col_allsim099":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_allsim_th099")
        inference_simcse(year, model, "bert_2col_allsim099")
    elif model == "bert_2col_allr2":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_allr2")
        inference_simcse(year, model, "bert_2col_allr2")
    elif model == "bert_2col_allr209":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_allr2_th09")
        inference_simcse(year, model, "bert_2col_allr209")
    elif model == "bert_2col_allr2093":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_allr2_th093")
        inference_simcse(year, model, "bert_2col_allr2093")
    elif model == "bert_2col_allr2095":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_allr2_th095")
        inference_simcse(year, model, "bert_2col_allr2095")
    elif model == "bert_2col_allr2099":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_2col_allr2_th099")
        inference_simcse(year, model, "bert_2col_allr2099")
    elif model == "bert_domlm_2col":  # wmlm=0.1
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col")
        inference_simcse(year, model, "bert_domlm_2col")
    elif model == "bert_domlm_2col_wmlm02":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col_wmlm02")
        inference_simcse(year, model, "bert_domlm_2col_wmlm02")
    elif model == "bert_domlm_2col_wmlm05":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col_wmlm05")
        inference_simcse(year, model, "bert_domlm_2col_wmlm05")
    elif model == "bert_domlm_2col_wmlm1":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col_wmlm1")
        inference_simcse(year, model, "bert_domlm_2col_wmlm1")
    elif model == "bert_domlm_2col_wmlm2":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col_wmlm2")
        inference_simcse(year, model, "bert_domlm_2col_wmlm2")
    elif model == "bert_domlm_2col_wmlm5":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col_wmlm5")
        inference_simcse(year, model, "bert_domlm_2col_wmlm5")
    elif model == "bert_domlm_2col_wmlm5_lseq512":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col_wmlm5_lseq512")
        inference_simcse(year, model, "bert_domlm_2col_wmlm5_lseq512")
    elif model == "bert_domlm_2col_wmlm5_35m":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col_wmlm5_35m")
        inference_simcse(year, model, "bert_domlm_2col_wmlm5_35m")
    elif model == "bert_domlm_2col_wmlm5_095th":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col_wmlm5_095th")
        inference_simcse(year, model, "bert_domlm_2col_wmlm5_095th")
    elif model == "bert_domlm_2col_wmlm5_09th":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col_wmlm5_09th")
        inference_simcse(year, model, "bert_domlm_2col_wmlm5_09th")
    elif model == "bert_domlm_2col_wmlm5_07th":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col_wmlm5_07th")
        inference_simcse(year, model, "bert_domlm_2col_wmlm5_07th")
    elif model == "bert_domlm_2col_wmlm5_065th":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col_wmlm5_065th")
        inference_simcse(year, model, "bert_domlm_2col_wmlm5_065th")
    elif model == "bert_domlm_2col_wmlm5_06th":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col_wmlm5_06th")
        inference_simcse(year, model, "bert_domlm_2col_wmlm5_06th")
    elif model == "bert_domlm_2col_wmlm5_05th":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col_wmlm5_05th")
        inference_simcse(year, model, "bert_domlm_2col_wmlm5_05th")
    elif model == "bert_domlm_2col_wmlm5_03th":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col_wmlm5_03th")
        inference_simcse(year, model, "bert_domlm_2col_wmlm5_03th")
    elif model == "bert_domlm_2col_wmlm10":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-bert-base-uncased_{year}_domlm_2col_wmlm10")
        inference_simcse(year, model, "bert_domlm_2col_wmlm10")
    elif model == "simcse":
        model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        inference_simcse(year, model, "simcse")
    elif model == "roberta_domlm":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-roberta-base_{year}_domlm")
        inference_simcse(year, model, "roberta_domlm")
    elif model == "roberta_2col":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-roberta-base_{year}_2col")
        inference_simcse(year, model, "roberta_2col")
    elif model == "roberta_2col_lseq512":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-roberta-base_{year}_2col_lseq512")
        inference_simcse(year, model, "roberta_2col_lseq512")
    elif model == "roberta_2col_lseq512_lr5e-6":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-roberta-base_{year}_2col_lseq512_lr5e-6")
        inference_simcse(year, model, "roberta_2col_lseq512_lr5e-6")
    elif model == "roberta_domlm_2col_wmlm5":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-roberta-base_{year}_domlm_2col_wmlm5")
        inference_simcse(year, model, "roberta_domlm_2col_wmlm5")
    elif model == "roberta_baseline":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-roberta-base_{year}_baseline")
        inference_simcse(year, model, "roberta_baseline")
    elif model == "roberta_r2":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-roberta-base_{year}")
        inference_simcse(year, model, "roberta_r2")
    elif model == "roberta_domlm_2col":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-roberta-base_{year}_domlm_2col")
        inference_simcse(year, model, "roberta_domlm_2col")
    elif model == "longformer_2col":  # contrastive->ap来自同一篇文章
        model = SimCSE(f"SimCSE/result/my-sup-simcse-longformer-base_{year}_2col")
        inference_simcse(year, model, "longformer_2col")
    elif model == "longformer_domlm":  # 丢弃
        model = SimCSE(f"SimCSE/result/my-sup-simcse-longformer-base_{year}_domlm")
        inference_simcse(year, model, "longformer_domlm")
    elif model == "longformer_domlm_2col_wmlm5":
        model = SimCSE(f"SimCSE/result/my-sup-simcse-roberta-base_{year}_domlm_2col_wmlm5")
        inference_simcse(year, model, "longformer_domlm_2col_wmlm5")
    else:
        print(f"no such model_{model}")
