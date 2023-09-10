import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from gensim import similarities
import random
from tqdm import tqdm
from scipy import stats

from Calculator.io import read_csv, read_pkl, load_embeddings, save_csv
from Calculator.similarity import return_future36m, similarity_each_peer_v2
from Calculator.utils import load_filebase, draw, return_past, return_future


def result(df10peers, year, t, check_all_r2):
    returnresult = read_csv("return_future", year=year, t=t)  # cik, RETX, year, month
    dftemp = pd.merge(returnresult, df10peers, on=["cik"], how="inner")  # cik, RETX, year, month, 0...9
    resultt = returnresult  # cik, RETX, year, month
    last_name = "cik"

    for i in range(10):
        resultt = resultt.rename(columns={last_name: str(i), "RETX" + last_name: "RETX" + str(i)})  # returnresult改名
        dftemp = pd.merge(dftemp, resultt, on=[str(i), 'year', 'month'], how="left")  # 更新df_temp，每个loop都给它加一列
        last_name = str(i)

    if check_all_r2:
        len1 = len(dftemp)
        resultt = dftemp.dropna(
            subset=["RETX0", "RETX1", "RETX2", "RETX3", "RETX4", "RETX5", "RETX6", "RETX7", "RETX8", "RETX9"],
            how="any")
        len2 = len(resultt)
        assert len1 == len2  # 因为参与统计的所有cik全部存在未来36m的return，所以result里面不可能存在nan。如果存在，一定哪里代码写错了
    else:
        resultt = dftemp.dropna(
            subset=["RETX0", "RETX1", "RETX2", "RETX3", "RETX4", "RETX5", "RETX6", "RETX7", "RETX8", "RETX9"],
            how="all")
    return resultt


def average(last_step, check_all_r2):
    matrix = last_step[
        ["RETX0", "RETX1", "RETX2", "RETX3", "RETX4", "RETX5", "RETX6", "RETX7", "RETX8", "RETX9"]].values
    if check_all_r2:
        mean = np.mean(matrix, axis=1)
    else:
        mean = np.nanmean(matrix, axis=1)

    output = last_step[["cik", "RETXcik", "month", "year"]].copy()
    output["RETXmean"] = mean
    output["month"] = output['month'].astype({'month': 'int64'})
    output["year"] = output['year'].astype({'year': 'int64'})
    return output


def regression_all(output):
    # Load the diabetes dataset
    X, y = output["RETXmean"].values.reshape(-1, 1), output["RETXcik"].values
    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    # Make predictions using the original set
    y_pred = regr.predict(X)

    print(r2_score(y, y_pred))


def regression_permonth(output, is_return):
    r2s = []
    for month in range(1, 13):
        X, y = output[output.month == month]["RETXmean"].values.reshape(-1, 1), \
               output[output.month == month]["RETXcik"].values

        regr = linear_model.LinearRegression()
        regr.fit(X, y)
        y_pred = regr.predict(X)
        r2 = r2_score(y, y_pred)
        r2s.append(r2)

    if not is_return:
        print(np.nanmean(r2s))
    else:
        return np.nanmean(r2s)


def regression_permonth_v2(output, is_return):
    r2s = []
    years = output.drop_duplicates(subset=["year"])["year"].values
    # print(years)

    for year in years:
        for month in range(1, 13):
            X, y = output[(output.month == month) & (output.year == year)]["RETXmean"].values.reshape(-1, 1), \
                   output[(output.month == month) & (output.year == year)]["RETXcik"].values

            regr = linear_model.LinearRegression()
            regr.fit(X, y)
            y_pred = regr.predict(X)
            r2 = r2_score(y, y_pred)
            r2s.append(r2)
    # draw_one(repeat=36, lst=r2s)
    if not is_return:
        print(np.nanmean(r2s))
    else:
        return np.nanmean(r2s)


def regression_permonth_v3(output, is_return):
    pearsonrs = []
    years = output.drop_duplicates(subset=["year"])["year"].values

    for year in years:
        for month in range(1, 13):
            X, y = output[(output.month == month) & (output.year == year)]["RETXmean"].values, \
                   output[(output.month == month) & (output.year == year)]["RETXcik"].values
            p = stats.pearsonr(X, y)[0]
            pearsonrs.append(p)

    if not is_return:
        print(np.nanmean(pearsonrs))
    else:
        return np.nanmean(pearsonrs)


def r2_each_peer(pivot_cik, inrange_return, t):
    data = []
    dfy = inrange_return[inrange_return["cik"] == pivot_cik].sort_values(by=['DATE'])[: 12 * t]
    y = dfy["RETXcik"].values
    ciks = inrange_return.drop_duplicates(subset=["cik"])["cik"].values

    for cik in ciks:
        dfx = inrange_return[inrange_return["cik"] == cik].sort_values(by=['DATE'])[: 12 * t]
        X = dfx["RETXcik"].values.reshape(-1, 1)
        regr = linear_model.LinearRegression()
        regr.fit(X, y)
        y_pred = regr.predict(X)
        row = pivot_cik, cik, r2_score(y, y_pred)
        data.append(row)

    df = pd.DataFrame(data, columns=["ciki", "cikj", "r2"])
    return df


#####################  ciki, cikj, r2, r2_rank ##########################
def return_past_all(year, t):
    """
    df_10k中每一个cik的10k date，往前数t年
    若pivot cik在12t这段时间里有cik，则：12t这段时间里有return的其他cik全都统计一下
    否则跳过这个pivot cik
    """
    # this year 10k dataframe
    df_10k_ = read_csv(type="index2file", year=year)  # df=(cik, date, index)
    df_10k = df_10k_[["index", "cik", "date"]].copy()
    # RETURN table: df=(cik, RETXcik, DATE)
    df_return = read_csv("return_all")

    pbar = tqdm(df_10k.iterrows(), desc="Firms", leave=False)
    df_return_past = None
    count = 0

    for i, row in pbar:
        # print(row["date"])
        df_inrange = return_past(df_return=df_return, pivot_10kdate=row["date"], t=t, pivot_cik=row["cik"])
        if df_inrange is None:
            # if pivot firm doesnt have past 12t month return, skip this firm
            continue
        else:
            count += 1
            pivot_lst = [row["cik"]] * len(df_inrange)
            df_inrange["pivot"] = pivot_lst
            df_return_past = pd.concat([df_return_past, df_inrange])

    assert count == len(df_return_past.drop_duplicates(subset=['pivot'], keep='last')["pivot"].values)

    filename = load_filebase() + "allcomp_priorreturn_{year}_t{t}.csv".format(t=t, year=year)
    save_csv(df=df_return_past, filename=filename)


def save_highsim_allr2(year, base, t, th):
    """
    base: SBERT/doc2vec/bert/...
    t: previous t years
    """
    # this year 10k dataframe
    df_10k_ = read_csv(type="index2file", year=year)  # df=(cik, date, index)
    df_10k = df_10k_[["index", "cik", "date"]].copy()
    # this year embedding
    embedding = load_embeddings(model=base, year=year)
    # RETURN table: df=(pivot, cik, RETXcik, DATE)
    df_return = read_csv(type="return_past_10kspe", year=year, t=t)

    ####### start ####################
    pbar = tqdm(df_10k.iterrows(), desc="Firms", leave=False)
    df_result = None
    for i, row in pbar:
        # query RETURN dataframe: all firms whose r2 in_range
        df_inrange = df_return[df_return["pivot"] == row["cik"]]  # pivot, cik, date, return
        if df_inrange.empty:
            # if pivot firm doesnt have past 12t month return, skip this firm
            continue

        # this year dataframe: each firm with peers whose sim >= th
        # df_10k_local: firm_cik whose r2 is in past range and cik is in this year. (pivot, cik, return, date)
        # df_highsim1: (cik, sim1, sim1_rank>=th)
        df_10k_local = df_10k[df_10k["cik"].isin(list(df_inrange["cik"].values))]
        df_highsim1 = similarity_each_peer_v2(embedding=embedding, pivot_idx=row["index"], df_10k=df_10k_local, th=th)

        # firms whose r2 in_range and sim1>=th and have embedding in embedyear
        result_1 = df_inrange[df_inrange["cik"].isin(list(df_highsim1["cik"].values))]

        # r2 dataframe: each firm with peers and their r2
        result_2 = r2_each_peer(pivot_cik=row["cik"], inrange_return=result_1, t=t)
        result_2["r2_rank"] = result_2['r2'].rank(pct=True)

        # concat
        df_result = pd.concat([df_result, result_2])

    filename = load_filebase() + f"allcomp_highsim_{year}_{base}_t{t}_th{th}_allr2.csv"
    save_csv(df=df_result, filename=filename)


def save_all_r2(year, t):
    """
    t: previous t years
    """
    # this year 10k dataframe
    df_10k_ = read_csv(type="index2file", year=year)  # df=(cik, date, index)
    df_10k = df_10k_[["index", "cik", "date"]].copy()
    # RETURN table: df=(pivot, cik, RETXcik, DATE)
    df_return = read_csv(type="return_past_10kspe", year=year, t=t)

    ####### start ####################
    pbar = tqdm(df_10k.iterrows(), desc="Firms", leave=False)
    df_result = None
    for i, row in pbar:
        # query RETURN dataframe: all firms whose r2 in_range
        df_inrange = df_return[df_return["pivot"] == row["cik"]]  # pivot, cik, date, return
        if df_inrange.empty:
            # if pivot firm doesnt have past 12t month return, skip this firm
            continue

        # r2 dataframe: each firm with peers and their r2
        result_r2 = r2_each_peer(pivot_cik=row["cik"], inrange_return=df_inrange, t=t)
        result_r2["r2_rank"] = result_r2['r2'].rank(pct=True)

        # concat
        df_result = pd.concat([df_result, result_r2])

    filename = load_filebase() + f"allcomp_allr2_{year}_t{t}.csv"
    save_csv(df=df_result, filename=filename)


#####################  calculate_r2 ##########################
def calculate_metric(year, model, t, r2):
    """
    format: model={name}_{extreme}{full} or =r2baseline or =random1 or =random2
    """
    filename = load_filebase() + f"10peers_{model}_{year}.csv"
    df10peers = read_csv(type="10peers", filename=filename)
    r = result(df10peers=df10peers, year=year, check_all_r2=False, t=t)
    output = average(r, check_all_r2=False)
    # regression_all(output)
    # return regression_permonth(output, is_return=True)
    if r2:
        return regression_permonth_v2(output, is_return=True)
    else:  # pearson
        return regression_permonth_v3(output, is_return=True)


def view_metric_compare_sample(year, model1, model2, repeat):
    print("calculating r2 for {year} data".format(year=year))
    filename = load_filebase() + "10peers_{model}_{year}.csv".format(model=model1, year=year)
    df10peers1 = read_csv(type="10peers", filename=filename)
    filename = load_filebase() + "10peers_{model}_{year}.csv".format(model=model2, year=year)
    df10peers2 = read_csv(type="10peers", filename=filename)

    lst_r2_1 = []
    lst_r2_2 = []
    for i in range(repeat):
        df10peers1_sample = df10peers1.sample(n=10, replace=True)
        ciks1 = df10peers1_sample["cik"].values
        df10peers2_sample = df10peers2[df10peers2['cik'].isin(list(ciks1))]
        while len(df10peers2_sample) < 10:
            df10peers1_sample = df10peers1.sample(n=10, replace=True)
            ciks1 = df10peers1_sample["cik"].values
            df10peers2_sample = df10peers2[df10peers2['cik'].isin(list(ciks1))]

        print("firms are:", ciks1)
        r = result(df10peers1_sample, year=year, check_all_r2=False, t=3)
        output = average(r, check_all_r2=False)
        lst_r2_1.append(regression_permonth(output, is_return=True))

        r = result(df10peers2_sample, year=year, check_all_r2=False, t=3)
        output = average(r, check_all_r2=False)
        lst_r2_2.append(regression_permonth(output, is_return=True))

    draw(repeat=repeat, lst1=lst_r2_1, lst2=lst_r2_2)
