from scipy import sparse

mat = sparse.load_npz('sparse_matrices/sparse_mat_tier1.npz')
print(mat.todense())

def init_term_index():
    f = open('dictionary.txt', 'r', encoding='utf-8')
    dictionary = f.readlines()
    dictionary = ['UNK', 'PAD_START', 'PAD_END'] + [line.strip() for line in dictionary]

    vocabSize = len(dictionary)
    term2id = dict(zip(dictionary, range(vocabSize)))

    return [term2id, vocabSize]

term2id, vocabSize = init_term_index()
print(term2id['之'])
print(term2id['不'])
# print(mat.todok()[5,8])
indices = mat.nonzero()
result = sorted(enumerate(indices), key=lambda x: -mat.data[x[0]])
print(result[:5])
# print(sorted(-mat.data)[:5])

# top 100~500 在八個朝代都有出現 的 row 算 std, var

# 之 的
# 們

# entropy