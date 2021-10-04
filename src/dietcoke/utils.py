from pathlib import Path
import json
from itertools import chain

fp_lst = [fp for fp in Path('../tiers/dynasty_split/').glob('*.jsonl')]

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

roles = '輯;撰;傳;述;奉敕譯;註;編;著;輯註;箋;章句;疏'
# alt: 注註
# 《大學》、《中庸》中的註釋稱為「章句」，《論語》、《孟子》中的註釋集合了眾人說法，稱為「集注」。後人合稱其為「四書章句集注」，簡稱「四書集注」。
prefix = '原題;舊題'
unknown = '[佚名];佚名'
others = '乾隆十三年;乾隆十二年'