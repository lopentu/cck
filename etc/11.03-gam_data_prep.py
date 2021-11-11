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
from itertools import chain
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
    author_profile = pd.read_csv('../data/author_time/author_profile.csv')
    author_profile = author_profile[['urn', 'title', 'mid_year', 'name']] \
                        .rename({'name': 'author_norm'}, axis=1)
    author_profile = author_profile[np.isnan(author_profile['mid_year']) == False]
    author_profile['dzg'] = author_profile['urn'].apply(lambda x: dzg_lookup.get(x, np.nan))

    urn_lookup = {}
    for _, row in author_profile.iterrows():
        urn_lookup[row['urn']] = dict(row)

    urns = list(urn_lookup.keys())

    print('Done.\n-----')
    return [urns, urn_lookup]

def calc_chunked_freq(cur_corpus, fn_out):
    ldf_lst = []
    for line in cur_corpus:
        try:
            growth_obj = Growth(''.join(line.text), 100000)
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
    chars_select += list(chain.from_iterable([n.split(';') for n in ["布;燈", "鐘;錶;磐;篪", "槍;刀;劍;戟;炮", "鯨", "心", "城;池", "快;慢", "籽", "日;山;涉;踄", "大;小;高", "貫;毌;擐;關", "矢;誓", "歌;唱;和"]]))
    #vocab_n = [char for char in vocab.dictionary[:20000] if not char in chars_select]
    chars = []#random.sample(vocab_n, k=1000-len(chars_select))
    chars += chars_select
    chars_index = [vocab.encode(char) for char in chars]
    
    print(f'{len(chars_select)} chars selected.')
    print(f'{len(chars) - len(chars_select)} chars sampled, first 5 chars:', chars[:5])

    with open(out_folder / 'char_lookup.json', 'w', encoding='utf-8') as f:
        json.dump(dict(zip(chars, chars_index)), f, ensure_ascii=False, indent=2)
    print('Done.\n-----')
    return [chars, chars_index]

def sample_freq_mat(chars, chars_index, urns_index):
    random.seed(2021)

    print('Sampling freq mat ...')
    with open('../data/chunked_freq/text_slice_lookup.json', 'r', encoding='utf-8') as f:
        text_slice_lookup = json.load(f)

    dfs = {}
    for fp in tqdm(Path('.').glob('../data/chunked_freq/*.npz')):
        mat = sparse.load_npz(fp).todense()[chars_index,]
        df = pd.DataFrame(mat)
        df.columns = text_slice_lookup[fp.stem]

        durns_index = set([col.split('_')[0] for col in df.columns if col.split('_')[0] in urns_index])
        durns_index = random.sample(durns_index, min(len(durns_index), 10))

        urn_cols = [col for col in df.columns if col.split('_')[0] in durns_index]
        df = df[urn_cols]

        df['char'] = chars
        df['dynaspan'] = str(fp.stem).replace('chunked_freq_', '')
        df_long = pd.melt(df, id_vars=['dynaspan', 'char'], var_name='urn_textslice', value_name='raw_freq')
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
    skip_first_step = input('skip creating chunked frequencies?\n')
    if skip_first_step == 'y':
        pass
    else:
        for corpus in corpus_lst():
            print(corpus.dynaspan)
            corpus.read_corpus()
            od_corpus = sorted(corpus.corpus, key=lambda x: Author(x.author).rep_year)
            if corpus.dynaspan == '清':
                for i in range(0, len(od_corpus), 300):
                    print(f'Processing split corpus at {i}, {i+300}')
                    calc_chunked_freq(od_corpus[i:i+300], f'chunked_freq_{corpus.dynaspan}_{str(i).zfill(4)}_{str(i+300).zfill(4)}.csv')
            # time.sleep(300)
            else:
                calc_chunked_freq(od_corpus, f'chunked_freq_{corpus.dynaspan}.csv')

    next_step = input('continue data transformation?\n')
    if next_step == 'y':
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