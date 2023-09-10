import numpy as np
from gensim import similarities
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from numpy import dot
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from re import compile, sub


def topic_concatenate(batch_topic_distributions, BATCH_SIZE):
    """
    input:
    batch_topic_distributions[0][0].shape = (batch_size, size of topic level 0)  # size of topic level 0 is largest
    batch_topic_distributions[0][1].shape = (batch_size, size of topic level 1)
    batch_topic_distributions[0][2].shape = (batch_size, size of topic level 2)
    ...

    output:
    new_topic_distributions.shape = (batch_size, size of topic level 0 + size of topic level 1 + ...)
    """

    topic_sizes = []
    for i in range(len(batch_topic_distributions[0])):
        topic_sizes.append(batch_topic_distributions[0][i].shape[1])

    length_concatenate_topics = 0
    for i in topic_sizes:
        length_concatenate_topics += i

    new_topic_distributions = []
    for i in range(BATCH_SIZE):
        temp = np.zeros(length_concatenate_topics)
        start = 0
        for j in range(len(batch_topic_distributions[0])):
            temp[start: start + topic_sizes[j]] = batch_topic_distributions[0][j][i]
            start += topic_sizes[j]

        new_topic_distributions.append(temp)

    return new_topic_distributions


def return_past(df_return, pivot_10kdate, pivot_cik, t):
    # print(f"{pivot_10kdate - pd.DateOffset(years=t)} to {pivot_10kdate}")
    # Select the rows between two dates
    in_range_df = df_return[
        df_return["DATE"].isin(
            pd.date_range(start=(pivot_10kdate - pd.DateOffset(years=t)), end=pivot_10kdate, freq='D', closed='left'))]
    len_past_cik = len(in_range_df[in_range_df["cik"] == pivot_cik])
    if len_past_cik < 12 * t:
        # check whether THIS FIRM have past 12t returns.
        print(f"cik{pivot_cik} not having enough previous 12*{t} return-values. only {len_past_cik} months.")
        return None
    else:
        # filter out firms not having enough returns.
        # ps: not exactly 12*t. sometime may bt 12*t+1, which is allowed.
        in_range_df = in_range_df.groupby('cik').filter(lambda x: len(x) >= 12 * t)
        return in_range_df  # (cik, date, return)


def return_future(df_return, pivot_10kdate, pivot_cik, t):
    # Select the rows between two dates
    in_range_df = df_return[
        df_return["DATE"].isin(
            pd.date_range(start=pivot_10kdate, end=(pivot_10kdate + pd.DateOffset(years=t)), freq='D', closed='right'))]

    len_past_cik = len(in_range_df[in_range_df["cik"] == pivot_cik])
    if len_past_cik < 12 * t:
        # check whether THIS FIRM have future 12t returns.
        print("this cik not having future ciks. only {} months.".format(len_past_cik))
        return None
    else:
        # filter out firms not having enough returns.
        # ps: not exactly 12*t. sometime may bt 12*t+1, which is allowed.
        in_range_df = in_range_df.groupby('cik').filter(lambda x: len(x) >= 12 * t)
        return in_range_df


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def tokenize(document):
    wnl = WordNetLemmatizer()
    # 删词规则
    r0 = "[\d]+"
    r1 = """[•!/_,$&%^*()<>+""''``?@|:;~{}#]+|[“”‘’]|(-[-]+)"""
    r2 = "[^\x00-\x7f]"

    # 停用词
    stop = set(stopwords.words('english'))

    # split
    sents = sent_tokenize(document)
    tokens = []
    for sent in sents:
        # 正则过滤
        sent = sub(r0, '', sent)
        sent = sub(r1, ' ', sent)
        sent = sub(r2, '', sent)

        # 单词按照词性还原（n-单复数还原，v-时态还原，a-比较级还原）
        tagged_sent = pos_tag(word_tokenize(sent))  # 单词在句子中的词性
        lemmas_sent = []
        for w, t in tagged_sent:
            wordnet_pos = get_wordnet_pos(t) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(w, pos=wordnet_pos))

        filter_sent = [w.lower() for w in lemmas_sent if w not in stop and len(w) > 3]
        tokens.extend(filter_sent)
    return tokens


def cosine_similarity(a, b):
    return dot(a, b) / max(norm(a) * norm(b), 1e-8)


def paragraph_embedding(tokens, glove_vectors):
    para_embed = []
    for word in tokens:
        if word in glove_vectors.vocab:
            para_embed.append(glove_vectors[word])
    para_embed = np.array(para_embed)
    para_embed = np.mean(para_embed, axis=0)
    return para_embed


def load_filebase():
    # file_base = "10kdata/russel3000/"
    file_base = "10kdata/"
    return file_base


def draw_result(lst_iter, lst1, lst2, title):
    plt.plot(lst_iter, lst1, '-b', label='model1')
    plt.plot(lst_iter, lst2, '-r', label='model2')
    plt.xlabel("n repeat")
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()


def draw(repeat, lst1, lst2):
    # iteration num
    lst_iter = range(repeat)
    draw_result(lst_iter, lst1, lst2, "sample comparison")


def draw_result_one(lst_iter, lst, title):
    plt.plot(lst_iter, lst, '-r')
    plt.xlabel("time")
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()


def draw_one(repeat, lst):
    # iteration num
    lst_iter = range(repeat)
    draw_result_one(lst_iter, lst, "r2 with time")


def get_split(text1, chunk_size=450, overlap=50):
    l_total = []
    l_parcial = []
    if len(text1.split()) // (chunk_size - overlap) > 0:
        n = len(text1.split()) // (chunk_size - overlap)
    else:
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = text1.split()[:chunk_size]
            l_total.append(" ".join(l_parcial))
        else:
            l_parcial = text1.split()[w * (chunk_size - overlap):w * (chunk_size - overlap) + chunk_size]
            l_total.append(" ".join(l_parcial))
    return l_total
