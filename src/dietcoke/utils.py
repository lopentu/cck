from pathlib import Path
import json
from itertools import chain
from tqdm.auto import tqdm

fp_lst = list(Path('../tiers/dynasty_split/').glob('*.jsonl'))

def corpus_lst():
    corpus_lst = [open(fp, 'r', encoding='utf-8').readlines() for fp in tqdm(fp_lst)]
    return corpus_lst

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
        self.texts = texts

class Author():
    def __init__(self):
        # self.roles = '輯;撰;傳;述;奉敕譯;註;注;編;著;輯註;箋;章句;疏'.split(';')
        # self.names_excluded = '李心[傳];劉銘[傳];[傳]恒;[傳]遜;孔[傳]金;畢弘[述];陳文[述];崔[述];莊[述]祖'.split(';')
        self.PAT_ROLES = '輯;撰;(?<!李心|劉銘)傳(?!恒|遜|金);(?<!畢弘|陳文)述(?!祖);奉敕譯;註;注;編;著;輯註;箋;章句;疏'.split(';')
        # alt: 注註
        # 《大學》、《中庸》中的註釋稱為「章句」，《論語》、《孟子》中的註釋集合了眾人說法，稱為「集注」。後人合稱其為「四書章句集注」，簡稱「四書集注」。
        self.PAT_PREFIX = '原題( *);舊題( *);題( *)'.split(';')

# cross_dynasty_urns = ['zhan-guo-ce', 'duduan', 'shan-hai-jing', 'er-ya', 'huangdi-neijing', 'kongcongzi', 'wenzi', 'guanzi', 'renwuzhi']

# dynasties = pd.read_csv('../dynasties.csv')