from pathlib import Path
import json
from itertools import chain
from tqdm.auto import tqdm
import re
import regex

BASEDIR_CORPUS = Path(__file__).parent / '../../data/dynasty_split/'

PAT_CLEANTEXT_RE = re.compile(r'[^\u4e00-\u9fd5]')
PAT_CLEANTEXT_REGEX = regex.compile(r'[^\p{Han}]')

dynaspan_lst = '先秦,漢,魏晉南北,唐五代十國,宋元,明,清,民國'.split(',')

def corpus_lst(dynaspan_lst=dynaspan_lst):
    corpus_lst = [Corpus(dynaspan) for dynaspan in tqdm(dynaspan_lst)]
    return corpus_lst

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
        with open(self.fp, 'r', encoding='utf-8') as fin:
            self.corpus = fin.readlines()

class Text:
    def __init__(self, line):
        self.obj = json.loads(line)
        self.urn = self.obj['urn']
        self.author = self.obj['author']

        # self.texts = [n['c'] for n in obj['text']]
        texts = []
        for n in self.obj['text']:
            if isinstance(n['c'], str):
                texts.append(n['c'])
            else:
                flatten = list(chain.from_iterable(n['c']))
                texts.append(flatten)
        self.__raw_texts = texts

    @property
    def raw_text(self):
        return self.__raw_texts

    @property
    def text(self):
        if not hasattr(self, "clean_texts"):
            self.clean()
        return self.clean_texts

    def clean(self):
        self.clean_texts = [regex.sub(PAT_CLEANTEXT_REGEX, '', text) for text in self.__raw_texts]

SPMAT_BASEDIR = Path("../data/cooccur_mat/win3")
def select_spmat(dynspan, winsize):
    sp_path = SPMAT_BASEDIR / f"sparse_mat_{dynspan}_contextSize{winsize}.npz"
    if sp_path.exists():
        return sp_path
    else:
        raise FileNotFoundError()

# cross_dynasty_urns = ['zhan-guo-ce', 'duduan', 'shan-hai-jing', 'er-ya', 'huangdi-neijing', 'kongcongzi', 'wenzi', 'guanzi', 'renwuzhi']

# dynasties = pd.read_csv('../dynasties.csv')