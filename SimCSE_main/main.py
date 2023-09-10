import numpy as np
from SimCSE_main.dataset import OriDataset, triplet_cik2paragraphs, col_cik2paragraphs
from Calculator.io import save_csv, read_csv


def create_triplet_dataset_ciks(year, base, th_sim, high, th):
    """
        3-column supervised training set for SimCSE. (cika, cikp, cikn).
        baseline: th=0.85, th_sim=0.97, and high="highsimallr2"
    """
    ori_dataset = OriDataset(year=year, base=base, high=high, th=th, th_sim=th_sim)
    ori_dataset.create_firm_peers()
    ori_dataset.create_triplet_shuffle(num_sample=20)
    cik_triplets = ori_dataset.get_triplet()  # (cika, cikp, cikn)
    filename = f"./SimCSE/data/triplet_ciks_for_simcse_{year}_{high}_{th}_{th_sim}.csv"
    save_csv(df=cik_triplets, filename=filename)


def triplet2paragraphs(year, high, th, th_sim):
    """
    3-column supervised training set for SimCSE. (texta, textp, textn).
    """
    df = triplet_cik2paragraphs(year, high, th, th_sim)
    filename = f"./SimCSE/data/triplets_for_simcse_{year}_{high}_{th}.csv"
    save_csv(df=df, filename=filename)


def col2paragraphs(year, high, th, th_sim):
    """
    2-column supervised training set for SimCSE. (texta, textp).
    """
    df = col_cik2paragraphs(year, high, th, th_sim)
    filename = f"./SimCSE/data/2col_for_simcse_{year}_{high}_{th}_{th_sim}.csv"
    save_csv(df=df, filename=filename)


def line2paragraphs(year):
    """
    1-column unsupervised training set for SimCSE. [text1, text2, ....].
    """
    df = read_csv(type="documents_paragraphs", year=year)
    documents = df["short_file"].values

    for doc in documents:
        x = doc.replace("\n", "")
        with open(f'./SimCSE/data/one_line_doc_{year}.txt', 'a') as the_file:
            the_file.write(x + '\n')


def documentsample(year, window_size):
    """
    Sample 512 tokens from full text. Should NOT be used casually!
    """
    assert window_size <= 512
    df = read_csv(type="index2file", year=year)  # df: cik, file_index
    print(len(df))

    def document_cut(text):
        tokens = text.split(" ")
        num_tokens = len(tokens)
        if num_tokens <= window_size:
            return " ".join(tokens)
        window_start = np.random.randint(0, num_tokens - window_size + 1)
        window_end = window_start + window_size
        return " ".join(tokens[window_start:window_end])

    df['short_file'] = df["document"].apply(document_cut)

    filename = f"./SimCSE/data/simcse_inference_{year}.csv"
    save_csv(df=df, filename=filename)