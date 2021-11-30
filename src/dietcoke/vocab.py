PATH_func_chars = '../notes/functions.txt'
with open(PATH_func_chars, 'r', encoding='utf-8') as f:
    func_chars = set()
    for line in f.readlines():
        line = line.strip().split(',')[0]
        if line.startswith('#') or line == '': continue
        func_chars |= set(line)
class Vocabulary:
    def __init__(self, dict_path):
        f = open(dict_path, 'r', encoding='utf-8')
        dictionary = f.readlines()
        dictionary = ['UNK', 'PAD_START', 'PAD_END'] + [line.strip() for line in dictionary]
        
        self.dictionary = dictionary
        self.rev_idx = None
        self.term2id = dict(zip(dictionary, range(len(dictionary))))

    def __len__(self):
        return len(self.dictionary)
    
    def __getitem__(self, term):
        return self.encode(term)

    def encode(self, term):
        return self.term2id.get(term, 0)

    def decode(self, idx):
        return self.dictionary[idx]

    @property
    def get_func_chars(self):
        if not hasattr(self, 'func_char'):
            self.func_chars = sorted([(char, self.encode(char)) for char in func_chars], key=lambda x: x[1])
        return self.func_chars