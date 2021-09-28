
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
