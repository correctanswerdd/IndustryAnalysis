import numpy as np
import pandas as pd
import random
from tqdm import tqdm

from Calculator.io import read_csv, load_embeddings, save_csv


def add_label(df, th):
    def addlabel_(rank):
        if rank >= th:
            return "high"
        else:
            return "low"

    df['label'] = df["value"].apply(addlabel_)
    return df


def add_to_samples(data, sent1, sent2, label):
    if sent1 not in data:
        data[sent1] = {'high': [], 'low': []}
    data[sent1][label].append(sent2)
    return data


def triplet_cik2paragraphs(year, high, th, th_sim):
    df_paragraphs = read_csv(type="documents_paragraphs", year=year)  # (cik, short_file)
    filename = f"./SimCSE/data/triplet_ciks_for_simcse_{year}_{high}_{th}_{th_sim}.csv"
    df_triplet = read_csv(type="cik_triplets", filename=filename)

    # merge
    df_ = df_paragraphs.rename(columns={"cik": "cika"})
    df_result = pd.merge(df_, df_triplet, on=["cika"], how="inner")
    df_result = df_result.rename(columns={"short_file": "texta"})
    df_ = df_paragraphs.rename(columns={"cik": "cikp"})
    df_result = pd.merge(df_, df_result, on=["cikp"], how="inner")
    df_result = df_result.rename(columns={"short_file": "textp"})
    df_ = df_paragraphs.rename(columns={"cik": "cikn"})
    df_result = pd.merge(df_, df_result, on=["cikn"], how="inner")
    df_result = df_result.rename(columns={"short_file": "textn"})
    return df_result[["texta", "textp", "textn"]]


def col_cik2paragraphs(year, high, th, th_sim):
    df_paragraphs = read_csv(type="documents_paragraphs", year=year)  # (cik, short_file)
    filename = f"./SimCSE/data/triplet_ciks_for_simcse_{year}_{high}_{th}_{th_sim}.csv"
    df_triplet = read_csv(type="cik_triplets", filename=filename)

    # merge
    df_ = df_paragraphs.rename(columns={"cik": "cika"})
    df_result = pd.merge(df_, df_triplet, on=["cika"], how="inner")
    df_result = df_result.rename(columns={"short_file": "texta"})
    df_ = df_paragraphs.rename(columns={"cik": "cikp"})
    df_result = pd.merge(df_, df_result, on=["cikp"], how="inner")
    df_result = df_result.rename(columns={"short_file": "textp"})
    return df_result[["texta", "textp"]]


class OriDataset(object):
    def __init__(self, year, base, high, th, th_sim):
        self.siamese_dataset = None
        self.triplet_dataset = None
        self.firm_peers = None

        df_base = self.init_base(year)
        self.create_siamese(year, df_base, base, high, th, th_sim)
        self.create_firm_peers()

    def init_base(self, year):
        """
        return all firms in this year
        return_df: (cik, embeddingidx). embeddingidx is for fetching bert embedding
        """
        ciks = read_csv(type="index2file", year=year)[["cik", "index"]]
        print("{yearfirm}firm num in dataset: {num}".format(yearfirm=year, num=len(ciks)))
        return ciks

    def create_siamese(self, year, df_base, base, high, th, th_sim):
        """
                NO SHUFFLE
            df_base = (cik, embeddingidx)
            base = doc2vec/SBERT/BERT(using)
            high = highsimallr2/allsim/allr2
            if high==highsimallr2 then th=0.85
            if high!=highsimallr2 then th=0.97

        return_df(df_result): (ciki, embeddingdix_x, cikj, embeddingidx_j, r2, label)
        """
        df1 = read_csv(type=high, year=year, t=3, base=base, th_sim=th_sim)
        if high == "highsimallr2":
            df1 = df1.rename(columns={"r2_rank": "value"})
        elif high == "allsim":
            df1 = df1.rename(columns={"sim_rank": "value"})
        elif high == "allr2":
            df1 = df1.rename(columns={"r2_rank": "value"})
        else:
            print(f"wrong type {high}")
            return
        # reset label
        df1 = add_label(df=df1, th=th)
        # cut table
        df1 = df1[["ciki", "cikj", "label"]]

        # merge df_base on ciki&cikj. df_base的数据仅来源于(year-t)~year之间有12t个return的firm.
        df_ = df_base.rename(columns={"cik": "ciki"})
        df_result = pd.merge(df1, df_, on=["ciki"], how="inner")  # (ciki, index, cikj, label)
        print("after merging on ciki, rows left: {num}".format(num=len(df_result)))
        df_ = df_base.rename(columns={"cik": "cikj"})
        df_result = pd.merge(df_result, df_, on=["cikj"],
                             how="inner")  # (ciki, index_x, cikj, index_y, label)
        print("after merging on cikj, rows left: {num}".format(num=len(df_result)))
        df_result = df_result.dropna(subset=['index_x', 'index_y'], how="any")
        print("after dropping nan rows, rows left: {num}".format(num=len(df_result)))

        # 只有这些firm的embedding会受到调整 in this year
        trainedcik = df_result.drop_duplicates(subset=["ciki"])[["ciki", "index_x"]]
        print("trained cik in {year} is {num}".format(year=year, num=len(trainedcik)))

        # original dataset: (index_x, index_y, label)
        self.siamese_dataset = df_result[["ciki", "index_x", "cikj", "index_y", "label"]]

    def create_firm_peers(self):
        df_result = self.siamese_dataset  # (ciki, index_x, cikj, index_j, label)
        data = {}
        for idx, row in df_result.iterrows():
            if row["ciki"] != row["cikj"]:
                embedding1_ = row["ciki"]
                embedding2_ = row["cikj"]
                data = add_to_samples(data, embedding1_, embedding2_, row['label'])
        self.firm_peers = data

    def create_triplet_shuffle(self, num_sample):
        data = self.firm_peers
        allsamples = []
        for sent1, others in data.items():
            pos_ciks = others['high']
            neg_ciks = others['low']
            if len(pos_ciks) > 0 and len(neg_ciks) > 0:
                for i in range(num_sample):
                    aidx = sent1
                    pidx = random.choice(pos_ciks)
                    nidx = random.choice(neg_ciks)
                    row = aidx, pidx, nidx
                    allsamples.append(row)

                    # opposite1
                    aidx = random.choice(pos_ciks)
                    pidx = sent1
                    nidx = random.choice(neg_ciks)
                    row = aidx, pidx, nidx
                    allsamples.append(row)

        self.triplet_dataset = pd.DataFrame(allsamples, columns=['cika', 'cikp', 'cikn'])

    def get_siamese(self):
        return self.siamese_dataset

    def get_triplet(self):
        return self.triplet_dataset

    def get_firm_peers(self):
        return self.firm_peers
