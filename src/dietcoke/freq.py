from pathlib import Path
import json
import pandas as pd
from tqdm.auto import tqdm
from scipy import sparse
from .utils import dynaspan_lst

def get_k_freq_mat(filter_year=True):
    FOLDER = Path('../data/chunked_freq')

    with open(FOLDER / 'text_slice_lookup.json', 'r', encoding='utf-8') as f:
        text_slice_lookup = json.load(f)

    if filter_year:
        PATH_AUTHOR_PROFILE = '../data/author_time/author_profile.csv'
        author_profile_lookup = pd.read_csv(PATH_AUTHOR_PROFILE, usecols=['urn', 'mid_year']) \
            .dropna() \
            .set_index('urn')['mid_year'].to_dict()

    mat = None
    cols = []
    for dynaspan in tqdm(dynaspan_lst):
        dyna_mat = sparse.load_npz(FOLDER / f'chunked_freq_{dynaspan}.npz')

        if filter_year:
            dyna_cols = [j for j, n in enumerate(text_slice_lookup[f'chunked_freq_{dynaspan}']) \
                if n.split('_')[0] in author_profile_lookup]
            
            dyna_mat = dyna_mat.todense()[:,dyna_cols]
            dyna_mat = sparse.csr_matrix(dyna_mat)
            cols += [text_slice_lookup[f'chunked_freq_{dynaspan}'][i] for i in dyna_cols]
        else:
            cols += text_slice_lookup[f'chunked_freq_{dynaspan}']

        print(dynaspan, dyna_mat.shape)

        if mat is None:
            mat = dyna_mat
        else:
            mat = sparse.hstack([mat, dyna_mat])

    print(mat.shape)
    return [mat, cols]