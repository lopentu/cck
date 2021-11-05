import sys
from pathlib import Path
if "../src" not in sys.path:
    sys.path.append("../src")
from dietcoke import Growth, Author, Vocabulary, dynaspan_lst, corpus_lst

import time
from datetime import datetime
import random
import pandas as pd
import json
import numpy as np
import re
from scipy import sparse
from tqdm.auto import tqdm

def get_dzg_lookup():
    print('Mapping urn and dzg category ...')
    fp = Path('../data/dzg/all_urns_dzg.json')
    if fp.exists():
        with open(fp, 'r', encoding='utf-8') as f:
            dzg = json.load(f)
            
        dzg_lookup = {}
        for line in dzg:
            dzg_lookup[line['urn']] = line['dzg']
    else:
        dzg_lookup = {}
        print('No mapping for urn and dzg category ...')
    print('Done.\n-----')
    return dzg_lookup

def get_urn_lookup():
    dzg_lookup = get_dzg_lookup()

    print('Creating urn lookup ...')
    urn_lookup = {}
    for corpus in tqdm(corpus_lst()):
        corpus.read_corpus()
        od_corpus = sorted(corpus.corpus, key=lambda x: Author(x.author).rep_year)
        for line in od_corpus:
            rep_year = Author(line.author).rep_year
            if rep_year == -9999: continue
            urn_lookup[line.urn] = {
                'title': line.obj['title'],
                'mid_year': rep_year,
                'author_norm': Author(line.author).name_norm[0],
                'dzg': dzg_lookup.get(line.urn, np.nan)
            }
    urns = list(urn_lookup.keys())

    print('Done.\n-----')
    return [urns, urn_lookup]

def calc_chunked_freq(cur_corpus, fn_out):
    ldf_lst = []
    for line in cur_corpus:
        try:
            growth_obj = Growth(''.join(line.text), 10000)
            ldf = growth_obj.get_freq_df
            ldf = ldf.astype(int)
            ldf.columns = [f'{line.urn}_{col}' for col in ldf.columns]
            ldf_lst.append(ldf)
            time.sleep(0.1)
        except Exception as e:
            print('---', line.urn)
            print(e)

    cdf = ldf_lst[0].join(ldf_lst[1:], how='outer') \
        .fillna(0).astype(int) \
        .reset_index() \
        .rename({'index': 'char'}, axis=1)
    # cdf.to_csv(fn_out)

    vocab = Vocabulary('../data/dictionary.txt').dictionary
    chars_emp_df = pd.DataFrame({'char': vocab, 'col': [-1 for _ in range(len(vocab))]})
    cdf = chars_emp_df.merge(cdf, how='left').drop(['char', 'col'], axis=1) \
            .fillna(0)
    mat = sparse.csr_matrix(cdf, dtype=int)
    sparse.save_npz(Path(fn_out).stem + '.npz', mat)

    with open(Path(fn_out).stem + '.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(cdf.columns))

    print(mat.shape)
    print(len(list(cdf.columns)))
    

def sample_vocab():
    print('Sampling characters from vocabulary ...')
    random.seed(2021)
    vocab = Vocabulary('../data/dictionary.txt')
    chars_select = ["子","曰","者","於","為","有","其","人","一","而","以","也","不","之"]
    vocab_n = [char for char in vocab.dictionary[:20000] if not char in chars_select]
    chars = random.sample(vocab_n, k=1000-len(chars_select))
    chars += chars_select
    chars_index = [vocab.encode(char) for char in chars]
    
    print(f'{len(chars_select)} chars selected.')
    print(f'{len(chars) - len(chars_select)} chars sampled, first 5 chars:', chars[:5])

    with open(out_folder / 'char_lookup.json', 'w', encoding='utf-8') as f:
        json.dump(dict(zip(chars, chars_index)), f, ensure_ascii=False, indent=2)
    print('Done.\n-----')
    return [chars, chars_index]

def sample_freq_mat(chars, chars_index, urns_index):
    print('Sampling freq mat ...')
    with open('../data/chunked_freq/text_slice_lookup.json', 'r', encoding='utf-8') as f:
        text_slice_lookup = json.load(f)

    dfs = {}
    for fp in tqdm(Path('.').glob('../data/chunked_freq/*.npz')):
        mat = sparse.load_npz(fp)[chars_index,].todense()
        df = pd.DataFrame(mat)
        df.columns = text_slice_lookup[fp.stem]

        urn_cols = [col for col in df.columns if col.split('_')[0] in urns_index]
        df = df[urn_cols]

        df['char'] = chars
        df_long = pd.melt(df, id_vars=['char'], var_name='urn_textslice', value_name='raw_freq')
        dfs[fp.stem] = df_long
    print('Done.\n-----')
    return dfs

def combine_dfs(dfs):
    print('Combining dfs ...')
    od_keys = sorted(dfs.keys(), key=lambda fn: (str(dynaspan_lst.index(str(fn).split('_')[2])) + '_' + str(fn).split('_')[-1]))
    df = pd.concat([dfs[key] for key in od_keys])
    print('Done.\n-----')
    return df

def add_meta(df, urn_lookup):
    print('Adding meta ...')
    df_lst = []
    for rowid in tqdm(range(0, df.shape[0], 1000000)):
        chunked_df = df.iloc[rowid:rowid+1000000,]

        urns, text_slice = zip(*[n.split('_') for n in chunked_df['urn_textslice']])
        chunked_df['urn'] = urns
        chunked_df['text_slice'] = text_slice
        chunked_df.drop('urn_textslice', axis=1, inplace=True)

        for meta in ['dzg', 'title', 'mid_year', 'author_norm']:
            lst = [urn_lookup.get(urn, {meta: np.nan}) for urn in chunked_df['urn']]
            chunked_df[meta] = [lookup.get(meta, np.nan) for lookup in lst]
        df_lst.append(chunked_df)
    print('Done.\n-----')
    return df_lst

def main():
    # for corpus in corpus_lst():
    #     print(corpus.dynaspan)
    #     corpus.read_corpus()
    #     od_corpus = sorted(corpus.corpus, key=lambda x: Author(x.author).rep_year)
    #     if corpus.dynaspan == '清':
    #         for i in range(0, len(od_corpus), 300):
    #             print(f'Processing split corpus at {i}, {i+300}')
    #             calc_chunked_freq(od_corpus[i:i+300], f'chunked_freq_{corpus.dynaspan}_{str(i).zfill(4)}_{str(i+300).zfill(4)}.csv')
    #     else:
    #         calc_chunked_freq(od_corpus, f'chunked_freq_{corpus.dynaspan}.csv')
    #     # break
    #     # time.sleep(300)

    # save_npz_from_csv()
    # save_text_slice_lookup()

    script_start_time = time.time()
    global out_folder
    out_folder = Path(datetime.today().strftime('%Y%m%d-%H%M%S'))
    out_folder.mkdir(exist_ok=True)

    urns, urn_lookup = get_urn_lookup()

    chars, chars_index = sample_vocab()
    dfs = sample_freq_mat(chars, chars_index, urns_index=urns)

    df = combine_dfs(dfs)
    df_lst = add_meta(df, urn_lookup)

    print('Concating dfs ...')
    df_lst2 = []
    for i in tqdm(range(0, len(df_lst), 10)):
        df_lst2.append(pd.concat(df_lst[i:i+10]))
    
    df = pd.concat(df_lst2)
    df = df.sort_values('mid_year')

    print('Saving gam_df.parquet')
    df.to_parquet(out_folder / 'gam_df.parquet')
    print('Done.\n-----')

    test_df = pd.read_parquet(out_folder / 'gam_df.parquet')
    print('Shape of gam_df.parquet:', test_df.shape)
    print(test_df.head())
    print(test_df.sample(n=10))

    print('Script time:', time.time() - script_start_time)

if __name__ == '__main__':
    main()

# def save_npz_from_csv():
#     def dfchunk2mat(fp):
#         df_lst = pd.read_csv(fp, chunksize=1000)
        
#         # df = pd.concat(df_lst, ignore_index=True)
#         cmat = None
#         for df in tqdm(df_lst):
#             df = df.drop('Unnamed: 0', axis=1).rename({'index': 'char'}, axis=1)
#             df = chars_emp_df.merge(df, how='left') \
#                     .drop(['char', 'col'], axis=1) \
#                     .fillna(0)

#             mat = sparse.csr_matrix(df, dtype=int)
#             if cmat is None:
#                 cmat = mat
#             else:
#                 cmat += mat
#         print('Done reading df chunks and creating a combined matrix ...', cmat.shape)
#         return cmat

#     def df2mat(fp):
#         df = pd.read_csv(fp) \
#                 .drop('Unnamed: 0', axis=1).rename({'index': 'char'}, axis=1)
#         print('Done reading the df ...')

#         df = chars_emp_df.merge(df, how='left') \
#                 .drop(['char', 'col'], axis=1) \
#                 .fillna(0)
#         mat = sparse.csr_matrix(df, dtype=int)
#         print('Done creating the matrix ...', mat.shape)
#         return mat

#     dic = Vocabulary('../data/dictionary.txt')
#     vocab = dic.dictionary
#     chars_emp_df = pd.DataFrame({'char': vocab, 'col': [-1 for _ in range(len(vocab))]})

#     for fp in Path('.').glob('*.csv'):
#         print(fp)
#         if re.search('宋元|明', str(fp)):
#             mat = dfchunk2mat(fp)
#         else:
#             mat = df2mat(fp)

#         start_time = time.time()
#         sparse.save_npz(fp.stem + '.npz', mat)
#         print('Time elapsed for saving matrix:', time.time() - start_time)
#         time.sleep(120)

# def save_text_slice_lookup():
#     import json

#     text_slice_lookup = {}
#     for fp in Path('.').glob('*.csv'):
#         with open(fp, 'r') as f:
#             line = f.readline()
#         text_slice_lookup[fp.stem] = line.split(',')[2:]

#     with open('text_slice_lookup.json', 'w', encoding='utf-8') as f:
#         json.dump(text_slice_lookup, f, ensure_ascii=False, indent=2)