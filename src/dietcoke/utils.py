from pathlib import Path
import json
from itertools import chain
import re
import regex

BASEDIR_CORPUS = Path(__file__).parents[3] / '../dynasty_split/'

PAT_CLEANTEXT_RE = re.compile(r'[^\u4e00-\u9fd5]')
PAT_CLEANTEXT_REGEX = regex.compile(r'[^\p{Han}]')

dynaspan_lst = '先秦,漢,魏晉南北,唐五代十國,宋元,明,清,民國'.split(',')

def corpus_lst(dynaspan_lst=dynaspan_lst):
    corpus_lst = [Corpus(dynaspan) for dynaspan in dynaspan_lst]
    return corpus_lst

class Text:
    def __init__(self, line):
        self.obj = json.loads(line)

        for key, value in self.obj.items():
            if key == 'text':
                text = []
                for n in value:
                    if isinstance(n['c'], str):
                        text.append(n['c'])
                    else:
                        flatten = list(chain.from_iterable(n['c']))
                        text.append(flatten)
                self.__raw_text = text
            else:
                setattr(self, key, value)

    @property
    def raw_text(self):
        return self.__raw_text

    @property
    def text(self):
        if not hasattr(self, 'clean_text'):
            self.clean()
        return self.clean_text

    def clean(self):
        self.clean_text = [regex.sub(PAT_CLEANTEXT_REGEX, '', text) for text in self.__raw_text]

class Corpus:
    def __init__(self, dynaspan=None, fp=None):
        if dynaspan:
            self.dynaspan = dynaspan
            self.fp = BASEDIR_CORPUS / f'{dynaspan}.jsonl'
        elif fp:
            if isinstance(fp, str): fp = Path(fp)
            self.fp = fp
            self.dynaspan = fp.stem
        else:
            raise ValueError('Either <dynaspan> or <fp> needs to be given')

    def read_corpus(self):
        corpus = []
        f = open(self.fp, 'r', encoding='utf-8')
        line_id = 0
        while True:
            line = f.readline()
            if not line:
                break
            try:
                corpus.append(Text(line))
            except:
                print(f'Unable to read line {line_id}')
            line_id += 1
        self.corpus = corpus

SPMAT_BASEDIR = Path("../data/cooccur_mat/win3")
def select_spmat(dynspan, winsize):
    sp_path = SPMAT_BASEDIR / f"sparse_mat_{dynspan}_contextSize{winsize}.npz"
    if sp_path.exists():
        return sp_path
    else:
        raise FileNotFoundError()

def save_file(data, fp_out):
    import pickle, json

    if isinstance(fp_out, str): fp_out = Path(fp_out)
    ext = fp_out.suffix

    if ext in ['.pickle', '.pkl']:
        with open(fp_out, 'wb') as f_out:
            pickle.dump(data, f_out)
    elif ext == '.json':
        with open(fp_out, 'w', encoding='utf-8') as f_out:
            json.dump(data, f_out, ensure_ascii=False, indent=2)
    else:
        raise TypeError

    print('File saved:', fp_out)

def read_file(fp_in):
    import pickle, json
    from scipy import sparse

    if isinstance(fp_in, str): fp_in = Path(fp_in)
    ext = fp_in.suffix

    if ext in ['.pickle', '.pkl']:
        with open(fp_in, 'rb') as f_in:
            data = pickle.load(f_in)
    elif ext == '.json':
        with open(fp_in, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)
    else:
        raise TypeError

    print('File read:', fp_in)
    return data

# cross_dynasty_urns = ['zhan-guo-ce', 'duduan', 'shan-hai-jing', 'er-ya', 'huangdi-neijing', 'kongcongzi', 'wenzi', 'guanzi', 'renwuzhi']

# dynasties = pd.read_csv('../dynasties.csv')