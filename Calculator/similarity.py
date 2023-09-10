from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from gensim import similarities
import random
from tqdm import tqdm
import torch

from Calculator.utils import load_filebase, cosine_similarity
from Calculator.io import read_pkl, read_csv, save_csv, load_embeddings, read_txt


def select_random_peers(year):
    if year != 2018:
        future_t = 1
    else:
        future_t = 3
    ciks, _, _ = return_future36m(year=year, t=future_t)
    data = []
    pbar = tqdm(ciks, desc="Firms", leave=False)
    for i, cik_i in enumerate(pbar):
        candidates = [x for x in ciks if x != cik_i]
        # randomly find 10 peers
        choices = random.sample(candidates, 10)
        for j in range(10):
            row = cik_i, choices[j]
            data.append(row)
    df = pd.DataFrame(data, columns=['focal', 'peer'])

    df_cik2id = read_csv(type="index2file", year=year)[["index", "cik"]]  # id, cik
    df_ = df_cik2id.rename(columns={"cik": "focal"})
    df_result = pd.merge(df, df_, on=["focal"], how="inner")  # (ciki, cikj, indexi)
    df_result = df_result.rename(columns={"index": "indexi"})
    df_ = df_cik2id.rename(columns={"cik": "peer"})
    df_result = pd.merge(df_result, df_, on=["peer"], how="inner")  # (ciki, cikj, indexi, indexj)
    df_result = df_result.rename(columns={"index": "indexj"})
    assert len(df_result) == len(df)
    return df_result


def return_future36m(year, t):
    """
    df1.values  # 不保证cik含36m
    result[["cik", "RETXcik", "year", "year", "month"]]  # 每个含36m的cik+return
    result.drop_duplicates(subset=['cik'], keep='last')["cik"].values  # 含36m的cik们
    """
    df2 = read_csv(type="return_future", year=year, t=t)
    df1 = read_csv(type="index2file", year=year)
    result = pd.merge(df2, df1, how='inner', on="cik")
    return df1["cik"].values, \
           result[["cik", "RETXcik", "year", "month"]], \
           result.drop_duplicates(subset=['cik'], keep='last')["cik"].values


def same_sector_candidates(cik, df_cname, ciks36m):
    df_cname = df_cname[df_cname["cik"].isin(list(ciks36m))]
    ser = df_cname[df_cname["cik"] == cik]["Sector"]
    if ser.empty:
        # candidates = list(ciks36m)
        # candidates.remove(cik)
        # print(f"case1+{len(ciks36m)}")
        return [x for x in ciks36m if x != cik]
    else:
        orig_sector = df_cname[df_cname["cik"] == cik]["Sector"].iat[0]
        candidates = df_cname[df_cname["Sector"] == orig_sector]["cik"].values
        if candidates.shape[0] >= 10:
            # print(f"case2+{len(candidates)}")  大部分为case2
            return [x for x in candidates if x != cik]
        else:  # 若candidate不足10个，则从没有sector信息的cik中抽几个补足剩下的
            # allcandidates = list(ciks36m)
            # allcandidates.remove(cik)
            allcandidates = [x for x in ciks36m if x != cik]
            n = 10 - candidates.shape[0]
            supplement = random.sample(allcandidates, n)
            # print(f"case3+{len(list(candidates) + supplement)}")
            return list(candidates) + supplement


def similarity_each_peer(embedding, pivot_idx, ciks, condition):
    data = []
    for i, cik_i in enumerate(ciks):
        if cik_i not in condition:
            continue
        else:
            row = cik_i, cosine_similarity(embedding[i], embedding[pivot_idx])
            data.append(row)
    df = pd.DataFrame(data)
    df = df.rename(columns={0: "cik", 1: "sim"})
    assert len(df) == len(condition)
    return df


def similarity_each_peer_v2(embedding, pivot_idx, df_10k, th):
    """
    return: cik(with high sim1), sim1
    """
    data = []
    for i, row in df_10k.iterrows():
        row = row["cik"], cosine_similarity(embedding[row["index"]], embedding[pivot_idx])
        data.append(row)

    df = pd.DataFrame(data)
    df = df.rename(columns={0: "cik", 1: "sim1"})
    df["sim1_rank"] = df['sim1'].rank(pct=True)
    df = df[df["sim1_rank"] >= th]
    return df


def similarity_each_peer_tfidf(tfidf, pivot_idx, ciks, condition):
    index = similarities.MatrixSimilarity(tfidf)
    sim_list = index[tfidf[pivot_idx]]
    data = []

    for i, cik_i in enumerate(ciks):
        if cik_i not in condition:
            continue
        else:
            row = cik_i, sim_list[i]
            data.append(row)
    df = pd.DataFrame(data)
    df = df.rename(columns={0: "cik", 1: "sim1"})
    return df


def find_peers_random1(year, future_t=3):
    df_siamese = read_csv(type="highsimallr2", year=year, t=3, base="bert")
    cikslist = list(df_siamese.drop_duplicates(subset=["ciki"])["ciki"].values)
    ciks, _, ciks36m = return_future36m(year=year, t=future_t)
    condition = [value for value in ciks36m if value in cikslist]
    data = []

    pbar = tqdm(ciks, desc="Firms", leave=False)
    for i, cik_i in enumerate(pbar):
        if cik_i not in condition:
            continue
        # candidates = cikslist
        # candidates.remove(cik_i)   ### 不可以这么remove！candidates是cikslist的copy，对candidates的改变会直接影响cikslist
        candidates = [x for x in cikslist if x != cik_i]
        # candidates = [x for x in ciks if x != cik_i]

        # randomly find 10 peers
        choices = random.sample(candidates, 10)
        row = cik_i, choices[0], choices[1], choices[2], choices[3], choices[4], choices[5], choices[6], choices[7], \
              choices[8], choices[9]
        data.append(row)

    df = pd.DataFrame(data, columns=['cik', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    filename = load_filebase() + f"10peers_random1_{year}.csv"
    save_csv(df, filename)
    return


def find_peers_random2(year, sector, future_t=3):
    df_siamese = read_csv(type="highsimallr2", year=year, t=3, base="bert")
    cikslist = df_siamese.drop_duplicates(subset=["ciki"])["ciki"].values
    ciks, _, ciks36m = return_future36m(year=year, t=future_t)
    condition = [value for value in ciks36m if value in cikslist]
    df_cname = read_csv(type=sector)
    # df_cname = df_cname_[df_cname_["cik"].isin(list(ciks36m))]

    data = []
    pbar = tqdm(ciks, desc="Firms", leave=False)
    for i, cik_i in enumerate(pbar):
        if cik_i not in condition:
            continue
        choices = random.sample(
            same_sector_candidates(cik=cik_i,
                                   df_cname=df_cname,
                                   ciks36m=cikslist), 10)
        row = cik_i, choices[0], choices[1], choices[2], choices[3], choices[4], choices[5], choices[6], choices[7], \
              choices[8], choices[9]
        data.append(row)

    df = pd.DataFrame(data, columns=['cik', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    filename = load_filebase() + "10peers_random2_{year}.csv".format(year=year, case=0)
    save_csv(df, filename)
    return


def find_peers_bowtfidf(year, model, future_t=3):
    """
    format: model=tfidf_{extreme}{full}
    """
    df_siamese = read_csv(type="highsimallr2", year=year, t=3, base="bert")
    cikslist = df_siamese.drop_duplicates(subset=["ciki"])["ciki"].values
    ciks, _, ciks36m = return_future36m(year, t=future_t)
    tfidf = load_embeddings(model=model, year=year)
    data = []
    condition = [value for value in ciks36m if value in cikslist]
    pbar = tqdm(ciks, desc="Firms", leave=False)
    for i, cik_i in enumerate(pbar):
        if cik_i not in condition:
            continue
        df = similarity_each_peer_tfidf(tfidf=tfidf, pivot_idx=i, ciks=ciks, condition=cikslist)
        df = df.sort_values(by='sim1', ascending=False)
        df = df[1:]

        # sort and find 10 peers
        cos_sim_list = df["cik"].values
        row = cik_i, cos_sim_list[0], cos_sim_list[1], cos_sim_list[2], cos_sim_list[3], cos_sim_list[4], \
              cos_sim_list[5], cos_sim_list[6], cos_sim_list[7], cos_sim_list[8], cos_sim_list[9]
        data.append(row)

    df = pd.DataFrame(data, columns=['cik', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    filename = load_filebase() + f"10peers_{model}_{year}.csv"
    save_csv(df, filename)
    return


def find_peers_r2baseline(year, future_t=3):
    df_siamese = read_csv(type="highsimallr2", year=year, t=3, base="bert")
    cikslist = df_siamese.drop_duplicates(subset=["ciki"])["ciki"].values
    ciks, _, ciks36m = return_future36m(year=year, t=future_t)
    data = []

    for ciki in cikslist:
        if ciki not in ciks36m:
            continue
        df_ = df_siamese[df_siamese["ciki"] == ciki]
        df_ = df_.sort_values(by='r2', ascending=False)
        df_ = df_[1:]  # 删除本身

        # sort and find 10 peers
        cos_sim_list = df_["cikj"].values
        row = ciki, cos_sim_list[0], cos_sim_list[1], cos_sim_list[2], cos_sim_list[3], cos_sim_list[4], \
              cos_sim_list[5], cos_sim_list[6], cos_sim_list[7], cos_sim_list[8], cos_sim_list[9]
        data.append(row)

    df = pd.DataFrame(data, columns=['cik', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    filename = load_filebase() + "10peers_{type}_{year}.csv".format(type="r2baseline", year=year)
    save_csv(df, filename)
    return


def find_peers(year, model, future_t=3, th_sim=None):
    """
    format: model={name}_{extreme}{full}
    """
    ciks, _, ciks36m = return_future36m(year=year, t=future_t)
    df_siamese = read_csv(type="highsimallr2", year=year, t=3, base="bert", th_sim=th_sim)
    cikslist = df_siamese.drop_duplicates(subset=["ciki"])["ciki"].values
    condition = [value for value in ciks36m if value in cikslist]
    embedding = load_embeddings(model=model, year=year)
    data = []

    pbar = tqdm(ciks, desc="Firms", leave=False)
    for i, cik_i in enumerate(pbar):
        if cik_i not in condition:
            continue
        df = similarity_each_peer(embedding=embedding, pivot_idx=i, ciks=ciks, condition=cikslist)
        df = df.sort_values(by='sim', ascending=False)
        df = df[1:]

        # sort and find 10 peers
        cos_sim_list = df["cik"].values
        row = cik_i, \
              cos_sim_list[0], cos_sim_list[1], cos_sim_list[2], cos_sim_list[3], cos_sim_list[4], \
              cos_sim_list[5], cos_sim_list[6], cos_sim_list[7], cos_sim_list[8], cos_sim_list[9]
        data.append(row)

    df = pd.DataFrame(data, columns=['cik', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    filename = load_filebase() + f"10peers_{model}_{year}.csv"
    save_csv(df, filename)
    return


def find_peers_information(year, model):
    """
    format: model={name}_{extreme}{full}
    """
    ciks, _, ciks36m = return_future36m(year=year, t=3)
    df_siamese = read_csv(type="highsimallr2", year=year, t=3, base="bert")
    df_cname = read_csv(type="cname")
    cikslist = df_siamese.drop_duplicates(subset=["ciki"])["ciki"].values
    condition = [value for value in ciks36m if value in cikslist]
    embedding = load_embeddings(model=model, year=year)
    data = []

    pbar = tqdm(ciks, desc="Firms", leave=False)
    for i, cik_i in enumerate(pbar):
        if cik_i not in condition:
            continue
        df = similarity_each_peer(embedding=embedding, pivot_idx=i, ciks=ciks, condition=cikslist)
        df = df.sort_values(by='sim', ascending=False)
        df = df[1:]

        # sort and find 10 peers
        firm_lst = df["cik"].values
        sim_lst = df["sim"].values
        for j in range(10):
            row = cik_i, firm_lst[j], sim_lst[j]
            data.append(row)

    df = pd.DataFrame(data, columns=['focal', 'peer', 'sim'])  # focal, peer
    df_cname_ = df_cname.rename(columns={'cik': 'focal'})  # focal, description, sector
    result = pd.merge(df, df_cname_, how="inner", on=["focal"])  # focal, description, sector, peer
    result = result.rename(
        columns={'Description': 'focal_Description', 'Industry': 'focal_Industry', 'Sector': 'focal_Sector'})
    df_cname_ = df_cname.rename(columns={'cik': 'peer'})  # peer, description, sector
    result = pd.merge(result, df_cname_, how="inner", on=["peer"])  # focal, desp, sec, peer, decp, sec
    result = result.rename(
        columns={'Description': 'peer_Description', 'Industry': 'peer_Industry', 'Sector': 'peer_Sector'})
    filename = load_filebase() + f"10peers_{model}_{year}_information.csv"
    save_csv(result, filename)
    return


def save_all_peers(year, base):
    """
    base: SBERT/doc2vec/bert/...
    """
    embedding = load_embeddings(model=base, year=year)
    ciks = list(read_csv(type="index2file", year=year)["cik"].values)  # id, cik
    df_result = None
    pbar = tqdm(ciks, desc="Firms", leave=False)
    for i, cik_i in enumerate(pbar):
        df = similarity_each_peer(embedding=embedding, pivot_idx=i, ciks=ciks, condition=ciks)
        df["sim_rank"] = df['sim'].rank(pct=True)
        df["focal"] = [cik_i] * len(df)
        df_result = pd.concat([df_result, df])

    df_result = df_result.rename(columns={"focal": "ciki", 'cik': 'cikj'})
    filename = load_filebase() + f"allcomp_allsim_{year}_{base}.csv"
    save_csv(df=df_result, filename=filename)


def calculate_distance_changing(year, model):
    df_siamese = read_csv(type="highsimallr2", year=year, t=3, base="bert")  # ciki, cikj, r2, r2_rank
    df_cik2id = read_csv(type="index2file", year=year)[["index", "cik"]]  # id, cik
    embedding = load_embeddings(model=model, year=year)
    df_ = df_cik2id.rename(columns={"cik": "ciki"})
    df_result = pd.merge(df_siamese, df_, on=["ciki"], how="inner")  # (ciki, cikj, r2, r2_rank, index)
    df_result = df_result.rename(columns={"index": "indexi"})
    df_ = df_cik2id.rename(columns={"cik": "cikj"})
    df_result = pd.merge(df_result, df_, on=["cikj"], how="inner")  # (ciki, cikj, r2, r2_rank, indexi, indexj)
    df_result = df_result.rename(columns={"index": "indexj"})
    assert len(df_result) == len(df_siamese)

    th = 0.85

    def add_label(r2rank):
        if r2rank >= th:
            return "high"
        else:
            return "low"

    embedding = load_embeddings(model=model, year=year)

    def cal_sim(x):
        return cosine_similarity(embedding[x[0]], embedding[x[1]])

    df_result["sim"] = df_result[["indexi", "indexj"]].apply(cal_sim, axis=1)
    df_result["label"] = df_result["r2_rank"].apply(add_label)
    df_high = df_result[df_result['label'] == 'high']
    df_low = df_result[df_result['label'] == 'low']
    df_high_mean = df_high[['ciki', 'sim']].groupby(['ciki']).mean()
    df_low_mean = df_low[['ciki', 'sim']].groupby(['ciki']).mean()

    df_high_mean["difference"] = df_high_mean["sim"] - df_low_mean["sim"]
    return df_high_mean['difference'].mean(axis=0)


def calculate_distance_changing_distribution(year, model):
    df_result = distribution_alignment(year, model)
    df_high = df_result[df_result['label'] == 'high']
    df_low = df_result[df_result['label'] == 'low']
    df_high_mean = df_high[['ciki', 'sim']].groupby(['ciki']).mean()
    df_low_mean = df_low[['ciki', 'sim']].groupby(['ciki']).mean()

    df_high_mean["difference"] = df_high_mean["sim"] - df_low_mean["sim"]

    df_high_mean = df_high_mean[np.abs(df_high_mean['difference'] - df_high_mean['difference'].mean()) <= (
                3 * df_high_mean['difference'].std())]
    # keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.

    return df_high_mean['difference'].tolist()


def draw_distance_changing(model1, model1_name, model2, model2_name):
    distance = []
    years = []
    models = []
    for year in [2014, 2015, 2016, 2017, 2018]:
        lst_high_mean = calculate_distance_changing_distribution(year, model1)
        distance += lst_high_mean
        years += [year] * len(lst_high_mean)
        models += [model1_name] * len(lst_high_mean)
    for year in [2014, 2015, 2016, 2017, 2018]:
        lst_high_mean = calculate_distance_changing_distribution(year, model2)
        distance += lst_high_mean
        years += [year] * len(lst_high_mean)
        models += [model2_name] * len(lst_high_mean)
    return pd.DataFrame({'distance': distance, 'model': models, 'year': years})


def distribution_alignment(year, model):
    df_siamese = read_csv(type="highsimallr2", year=year, t=3, base="bert")  # ciki, cikj, r2, r2_rank
    df_cik2id = read_csv(type="index2file", year=year)[["index", "cik"]]  # id, cik
    df_ = df_cik2id.rename(columns={"cik": "ciki"})
    df_result = pd.merge(df_siamese, df_, on=["ciki"], how="inner")  # (ciki, cikj, r2, r2_rank, index)
    df_result = df_result.rename(columns={"index": "indexi"})
    df_ = df_cik2id.rename(columns={"cik": "cikj"})
    df_result = pd.merge(df_result, df_, on=["cikj"], how="inner")  # (ciki, cikj, r2, r2_rank, indexi, indexj)
    df_result = df_result.rename(columns={"index": "indexj"})
    assert len(df_result) == len(df_siamese)

    th = 0.85

    def add_label(r2rank):
        if r2rank >= th:
            return "high"
        else:
            return "low"

    embedding = load_embeddings(model=model, year=year)

    def cal_sim(x):
        return cosine_similarity(embedding[x[0]], embedding[x[1]])

    df_result["sim"] = df_result[["indexi", "indexj"]].apply(cal_sim, axis=1)
    df_result['sim_norm'] = (df_result["sim"] - df_result["sim"].min()) / (
            df_result["sim"].max() - df_result["sim"].min())
    df_result["label"] = df_result["r2_rank"].apply(add_label)
    return df_result


def distribution_uniformity(year, model):
    df_result = select_random_peers(year)  # (focal_firm, peer_firm)*N
    embedding = load_embeddings(model=model, year=year)

    def cal_sim(x):
        return cosine_similarity(embedding[x[0]], embedding[x[1]])

    df_result["sim"] = df_result[["indexi", "indexj"]].apply(cal_sim, axis=1)
    df_result['sim_norm'] = (df_result["sim"] - df_result["sim"].min()) / (
            df_result["sim"].max() - df_result["sim"].min())
    return df_result


def calculate_alignment(year, model):
    df_result = distribution_alignment(year, model)
    df_result = df_result[df_result['label'] == 'high']
    # df_result['square'] = df_result['sim_norm'] ** 2
    # return df_result['square'].mean(axis=0)
    x = torch.from_numpy(df_result['sim_norm'].values).reshape(-1, 1)
    return x.norm(p=2, dim=1).pow(2).mean()


def calculate_uniformity(year, model):
    df_result = distribution_uniformity(year, model)
    x = torch.from_numpy(df_result["sim_norm"].values).reshape(-1, 1)
    return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log().item()


def draw_align_uniform(year, model):
    df_align = distribution_alignment(year, model)
    df_uniform = distribution_uniformity(year, model)
    return


def draw_distributions(year, model1, model2):
    df_result = select_random_peers(year)
    embedding1 = load_embeddings(model=model1, year=year)
    embedding2 = load_embeddings(model=model2, year=year)

    def cal_sim1(x):
        return cosine_similarity(embedding1[x[0]], embedding1[x[1]])

    def cal_sim2(x):
        return cosine_similarity(embedding2[x[0]], embedding2[x[1]])

    df_result["sim1"] = df_result[["indexi", "indexj"]].apply(cal_sim1, axis=1)
    df_result["sim2"] = df_result[["indexi", "indexj"]].apply(cal_sim2, axis=1)
    df_result['sim_norm1'] = (df_result["sim1"] - df_result["sim1"].min()) / (
            df_result["sim1"].max() - df_result["sim1"].min())
    df_result['sim_norm2'] = (df_result["sim2"] - df_result["sim2"].min()) / (
            df_result["sim2"].max() - df_result["sim2"].min())

    df_result = df_result.rename(columns={'sim_norm1': 'BERT', 'sim_norm2': 'CLIA-BERT'})
    return df_result[['BERT', 'CLIA-BERT']]
