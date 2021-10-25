from pathlib import Path
import json
from itertools import chain
from collections import defaultdict
from tqdm.auto import tqdm
import re
import regex
from .author import Author

BASEDIR_CORPUS = Path(__file__).parents[3] / '../dynasty_split/'

PAT_CLEANTEXT_RE = re.compile(r'[^\u4e00-\u9fd5]')
PAT_CLEANTEXT_REGEX = regex.compile(r'[^\p{Han}]')

dynaspan_lst = '先秦,漢,魏晉南北,唐五代十國,宋元,明,清,民國'.split(',')

def corpus_lst(dynaspan_lst=dynaspan_lst):
    corpus_lst = [Corpus(dynaspan) for dynaspan in tqdm(dynaspan_lst)]
    return corpus_lst

class Text:
    def __init__(self, line):
        self.obj = json.loads(line)
        self.urn = self.obj['urn']
        self.author = self.obj['author']

        # self.texts = [n['c'] for n in obj['text']]
        text = []
        for n in self.obj['text']:
            if isinstance(n['c'], str):
                text.append(n['c'])
            else:
                flatten = list(chain.from_iterable(n['c']))
                text.append(flatten)
        self.__raw_text = text

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

    def get_author_time_lookup(self, save_lookup=False):
        lookup_fp = Path(f'../data/author_time/year_lineid_lookup_{self.dynaspan}.json')
        if not hasattr(self, 'author_time_lookup'):
            if lookup_fp.exists():
                print('Reading existing lookup file...')
                with open(lookup_fp, 'r', encoding='utf-8') as f:
                    self.author_time_lookup = json.load(f)
            else:
                authors = [line.author for line in self.corpus]
                rep_years = [(i, Author(author).rep_year) for i, author in enumerate(authors)]

                rep_years = sorted(rep_years, key=lambda x: x[1])
                rep_dic = defaultdict(list)
                for i, rep_year in rep_years:
                    rep_dic[rep_year].append(i)
                self.author_time_lookup = dict(rep_dic)

        if save_lookup:
            with open(lookup_fp, 'w', encoding='utf-8') as f:
                json.dump(self.author_time_lookup, f, ensure_ascii=False, indent=2)
            print('File saved:', lookup_fp)

        return self.author_time_lookup

    def get_author_time_corpus(self):
        if not hasattr(self, 'author_time_lookup'):
            self.get_author_time_lookup()

        author_time_corpus = {}
        for rep_year, i_lst in self.author_time_lookup.items():
            author_time_corpus[rep_year] = [self.corpus[i] for i in i_lst]
        self.author_time_corpus = author_time_corpus
        return self.author_time_corpus

    def check_author_time(self):
        error = None
        for rep_year, line_lst in self.author_time_corpus.items():
            for line in line_lst:
                if Author(line.author).rep_year != int(rep_year):
                    print(rep_year)
                    print('Not matched', (line.obj['title'], line.urn, line.author, Author(line.author).rep_year))
                    error = True
        
        if error is None:
            print('Done checking ...')

SPMAT_BASEDIR = Path("../data/cooccur_mat/win3")
def select_spmat(dynspan, winsize):
    sp_path = SPMAT_BASEDIR / f"sparse_mat_{dynspan}_contextSize{winsize}.npz"
    if sp_path.exists():
        return sp_path
    else:
        raise FileNotFoundError()

# cross_dynasty_urns = ['zhan-guo-ce', 'duduan', 'shan-hai-jing', 'er-ya', 'huangdi-neijing', 'kongcongzi', 'wenzi', 'guanzi', 'renwuzhi']

# dynasties = pd.read_csv('../dynasties.csv')