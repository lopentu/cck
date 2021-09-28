from pathlib import Path
import numpy as np
from scipy import sparse
from tqdm.auto import tqdm
import itertools
import collections
import time
import pickle
from get_dictionary import Text

##### 動態更新詞彙共現矩陣 #####
# sol 1: 矩陣相加
# sol 2: 找出原矩陣非零欄列位置及資料（row and column index & data），新增新資料欄列位置，建一新矩陣
# http://seanlaw.github.io/2019/02/27/set-values-in-sparse-matrix/

# reference: https://github.com/scipy/scipy/issues/11600
def create_full_mat(half_mat):
    start_time = time.time()

    t_mat = half_mat.copy().transpose()
    nonzero, = t_mat.diagonal().nonzero()
    t_mat[nonzero, nonzero] = 0
    full_mat = half_mat + t_mat

    print('Time elapsed for creating full co-occur matrix:', time.time() - start_time)
    return full_mat

def init_term_index():
    f = open('dictionary.txt', 'r', encoding='utf-8')
    dictionary = f.readlines()
    dictionary = ['UNK', 'PAD_START', 'PAD_END'] + [line.strip() for line in dictionary]

    vocabSize = len(dictionary)
    term2id = dict(zip(dictionary, range(vocabSize)))

    return [term2id, vocabSize]

# source: https://napsterinblue.github.io/notes/python/internals/itertools_create_concords/
# reference: https://coady.github.io/posts/split-an-iterable/
# reference: https://www.py4u.net/discuss/143692
# itertools.tee: https://medium.com/@jasonrigden/a-guide-to-python-itertools-82e5a306cdf8
def create_concords(iterable):
    iterable = [term2id.setdefault(term, 0) for term in iterable]

    for _ in range(windowSize[0]):
        iterable.insert(0, term2id.setdefault('PAD_START', 0))

    for _ in range(windowSize[2]):
        iterable.append(term2id.setdefault('PAD_END', 0))

    iterables = itertools.tee(iterable, contextSize)
    for iterable, num_skipped in zip(iterables, itertools.count()):
        for _ in range(num_skipped):
            next(iterable, None)

    return zip(*iterables)

def sum_mat_idx(data, row, col):
    mat = sparse.csr_matrix((data, (row, col)), shape=(vocabSize, vocabSize))
    stacked = np.vstack((mat.data, mat.nonzero()))
    return stacked

def create_line_mat_idx(concords):
    try:
        col = np.array([np.fromiter(concord, dtype=int) for concord in concords])
        row = np.repeat(col[:,2], contextSize)
        col = col.reshape(-1)
        data = np.repeat(1, len(col))

        line_mat_idx = sum_mat_idx(data, row, col)
        return line_mat_idx
    except:
        print('?')

def apply_sauce(line):
    try:
        texts = Text(line).texts
        concords = (create_concords(text) for text in texts)
        line_mat_idx = [create_line_mat_idx(concord) for concord in concords]
        line_mat_idx = np.concatenate(line_mat_idx, axis=1)

        # line_mat_idx = sum_mat_idx(line_mat_idx[0,:], line_mat_idx[1,:], line_mat_idx[2,:])
        line_mat = sparse.csr_matrix(
            (line_mat_idx[0,:], (line_mat_idx[1,:], line_mat_idx[2,:])),
            shape=(vocabSize, vocabSize), dtype=int)
    except:
        print('??')
        line_mat = sparse.csr_matrix((vocabSize, vocabSize), dtype=int)
    return line_mat

windowSize = (20, 1, 20) # (pre-context, keyword, post-context)
contextSize = sum(windowSize)

term2id, vocabSize = init_term_index()

def main():

    for fp in Path('tiers/dynasty_split/').glob('*.jsonl'):
        if ('清' in str(fp)):# \
            # and (not fp.exists()):
            print('-----', fp.name, '-----')
            f = open(fp, 'r', encoding='utf-8')
            corpus = f.readlines()

            half_mat = sparse.csr_matrix((vocabSize, vocabSize), dtype=int)
            for line in tqdm(corpus):
                line_mat = apply_sauce(line)
                half_mat += line_mat
            mat = create_full_mat(half_mat)
            print(mat.count_nonzero())

            start_time = time.time()
            sparse.save_npz(f'sparse_matrices/sparse_mat_{fp.stem}_contextSize{contextSize}.npz', mat)
            print('Time elapsed for saving matrix:', time.time() - start_time)

            time.sleep(300)

if __name__ == '__main__':
    main()
    # spr_mat = sparse.load_npz('sparse_mat.npz')