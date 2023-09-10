#!/usr/bin/env python3
import pandas as pd
import pickle

from Calculator.utils import load_filebase


def save_pkl(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def read_csv(type, year=None, filename=None, base=None, t=None, th_sim=None):
    if type == "index2file":
        assert year is not None
        file_index = "../russell3000-10K/updated_fileindex.csv"
        df_ = pd.read_csv(file_index)  # (cik, date, item1_index)
        df_ = df_.dropna(subset=['item1'])
        df_ = df_.reset_index(drop=True)  # drop=True: avoid the old index being added as a column:
        df_['date'] = df_['date'].apply(str)
        df = df_[["cik", "date", "item1"]].copy()

        data = []
        base = "../russell3000-10K/"
        for index, row in df.iterrows():
            if row["date"][:4] == str(year):
                path2item1 = row["item1"]
                with open(base + path2item1) as f:
                    filecontent = f.read()
                    t = row['cik'], filecontent, row["date"]
                    data.append(t)
        df = pd.DataFrame(data, columns=['cik', 'document', "date"])
        df = df.drop_duplicates(subset=['cik'])  # duplicate cik两份年报 => 保留其中一个即可
        df = df.reset_index()  # df=(cik, index, document, date). index->embedding_index
        df['index'] = df.index
        df["date"] = pd.to_datetime(df['date'], format='%Y%m%d')  # text发表日期
        return df
    elif type == "10peers":
        # print("reading 10peer.csv")
        df = pd.read_csv(filename)
        return df
    elif type == "return_future":
        assert year is not None and t is not None
        file_cik2monthlyreturn = "../russell3000-10K/russell3000_return.csv"
        df1 = pd.read_csv(file_cik2monthlyreturn)
        df1['DATE'] = df1['DATE'].apply(str)
        df1 = df1.dropna(subset=['CIK', 'RETX'], how="any")  # CIK和RETX有nan值
        df1['CIK'] = df1['CIK'].astype({'CIK': 'int64'})  # CIK导入的时候为float
        df1 = df1[df1.RETX != 'C']  # 删除RETX含C的row
        df1 = df1[df1.RETX != 'B']  # 删除RETX含B的row
        df1['RETX'] = df1['RETX'].astype({'RETX': 'float64'})  # CIK导入的时候为float
        df1 = df1.rename(columns={"CIK": "cik", "RETX": "RETXcik"})

        # generate df2=(CIK, return, year, month) from df1(cik, return, date)
        df2 = df1[['cik', 'RETXcik']].copy()
        df2["year"] = df1["DATE"].apply(lambda x: x[:4])
        df2["month"] = df1["DATE"].apply(lambda x: x[4: 6])

        # filter out year=2014->(2015+2016+2017)
        df2 = df2[(df2.year <= str(year + t)) & (df2.year >= str(year + 1))]

        # Removing Rows on Count condition. 保证留下来的cik全都是return等于36个月
        df2 = df2.groupby('cik').filter(lambda x: len(x) == 12 * t)
        return df2  # df2("cik", "RETXcik", "year", "month")
    elif type == "return_all":  ##############################
        # file_cik2monthlyreturn = "../Data/russell3000-10K/russell3000_return.csv"
        # df1 = pd.read_csv(file_cik2monthlyreturn)
        # # df1['DATE'] = df1['DATE'].apply(str)
        # df1 = df1.dropna(subset=['CIK', 'RETX'], how="any")  # CIK和RETX有nan值
        # df1['CIK'] = df1['CIK'].astype({'CIK': 'int64'})  # CIK导入的时候为float
        # df1 = df1[df1.RETX != 'C']  # 删除RETX含C的row
        # df1 = df1[df1.RETX != 'B']  # 删除RETX含B的row
        # df1['RETX'] = df1['RETX'].astype({'RETX': 'float64'})  # CIK导入的时候为float
        # df1 = df1.rename(columns={"CIK": "cik", "RETX": "RETXcik"})
        #
        # # generate df=(CIK, return, DATE)
        # filename = load_filebase() + "allreturn.csv"
        # save_csv(df=df1[['cik', 'RETXcik', 'DATE']], filename=filename)

        filename = load_filebase() + "allreturn.csv"
        df = pd.read_csv(filename)
        df["DATE"] = pd.to_datetime(df['DATE'], format='%Y%m%d')
        return df
    elif type == "cname":  ######################
        file_cik2cname = "../russell3000-10K/russell3000_constituents_cik.csv"
        df_cname = pd.read_csv(file_cik2cname)
        df_cname = df_cname.dropna(subset=['cik'])  # CIK有nan值
        df_cname['cik'] = df_cname['cik'].astype({'cik': 'int64'})  # CIK导入的时候为float
        df_cname = df_cname[["Description", "Industry", "Sector", "cik"]]
        df_cname = df_cname.drop_duplicates(subset=['cik'])  # 有重复的cik
        return df_cname
    elif type == "cname_v2":  ######################
        file_cik2cname = "../russell3000-10K/russell3000_constituents_cik.csv"
        df_cname = pd.read_csv(file_cik2cname)
        df_cname = df_cname.dropna(subset=['cik'])  # CIK有nan值
        df_cname['cik'] = df_cname['cik'].astype({'cik': 'int64'})  # CIK导入的时候为float
        df_cname = df_cname[["Symbol", "Description", "Industry", "cik"]]
        df_cname = df_cname.drop_duplicates(subset=['cik'])  # 有重复的cik

        file_symbol2sector = "../russell3000-10K/russell3000_ticker_GIC.csv"
        df_gsector = pd.read_csv(file_symbol2sector)
        df_gsector = df_gsector[["Symbol", "gsector"]]
        df_gsector = df_gsector.rename(columns={"gsector": "Sector"})

        df_result = pd.merge(df_cname, df_gsector, on=["Symbol"], how="inner")
        return df_result[["Description", "Industry", "cik", "Sector"]]
    elif type == "allsim":
        assert base is not None and year is not None
        filename = load_filebase() + f"allcomp_allsim_{year}_{base}.csv"
        df = pd.read_csv(filename)
        return df
    elif type == "allr2":
        assert t is not None and year is not None
        filename = load_filebase() + f"allcomp_allr2_{year}_t{t}.csv"
        df = pd.read_csv(filename)
        return df
    elif type == "highsimallr2":
        assert t is not None and base is not None and year is not None and th_sim is not None
        filename = load_filebase() + f"allcomp_highsim_{year}_{base}_t{t}_th{th_sim}_allr2.csv"
        df = pd.read_csv(filename)
        n_ciks = len(df.drop_duplicates(subset=['ciki'], keep='last')["ciki"].values)
        print(f"number of this year firm that have past 12*{t} month returns: {n_ciks}")
        return df
    elif type == "highsimallr2_35m":
        assert t is not None and base is not None and year is not None
        filename = load_filebase() + "allcomp_highsim1_{year}_{base}_t{t}_allr2_35.csv".format(t=t, base=base, year=year)
        df = pd.read_csv(filename)
        n_ciks = len(df.drop_duplicates(subset=['ciki'], keep='last')["ciki"].values)
        print(f"number of this year firm that have past 12*{t} month returns: {n_ciks}")
        return df
    elif type == "return_past_10kspe":
        assert year is not None and t is not None
        # filename = load_filebase() + "allcomp_pastreturn_{year}_t{t}.csv".format(t=t, year=year)
        filename = load_filebase() + "allcomp_priorreturn_{year}_t{t}.csv".format(t=t, year=year)
        df = pd.read_csv(filename)
        return df
    elif type == "return_past_10kspe_v2":
        assert year is not None and t is not None
        filename = load_filebase() + "allcomp_priorreturn_{year}_t{t}_35.csv".format(t=t, year=year)
        df = pd.read_csv(filename)
        return df
    elif type == "documents_paragraphs":
        filename = f"./SimCSE/data/simcse_inference_{year}.csv"
        df = pd.read_csv(filename)
        return df
    elif type == "documents_paragraphs_test":
        filename = f"./SimCSE/data/simcse_inference_{year}_test.csv"
        df = pd.read_csv(filename)
        return df
    elif type == "cik_triplets":
        assert filename is not None
        # filename = f"./SimCSE/data/cik_triplets_for_simcse_{year}.csv"
        df = pd.read_csv(filename)
        return df
    else:
        print("wrong type name!")
        return


def save_csv(df, filename):
    df.to_csv(filename, encoding='utf-8', index=False)


def read_pkl(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def read_txt(filename):
    with open(filename, 'r') as reader:
        return reader.read()


def flatten(t):
    return [item for sublist in t for item in sublist]


def chunks(lst, n):
    x = []
    for i in range(0, len(lst), n):
        x.append(flatten(lst[i:i + n]))
    return x


def load_embeddings(model, year):
    if "tfidf" in model or "wordcount" in model:
        filename = load_filebase() + f"{model}_{year}.pkl"
        embedding = read_pkl(filename)
    else:
        filename = load_filebase() + f"{model}_inference_{year}.pkl"
        embedding = read_pkl(filename)
    return embedding
