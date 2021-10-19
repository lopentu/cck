from pathlib import Path
import pickle
import re
from itertools import chain

# roles = '輯;撰;傳;述;奉敕譯;註;注;編;著;輯註;箋;章句;疏'.split(';')
# names_excluded = '李心[傳];劉銘[傳];[傳]恒;[傳]遜;孔[傳]金;畢弘[述];陳文[述];崔[述];莊[述]祖'.split(';')
PAT_ROLES = '輯;撰;(?<!李心|劉銘)傳(?!恒|遜|金);(?<!畢弘|陳文)述(?!祖);奉敕譯;註;注;編;著;輯註;箋;章句;疏'.split(';')
# alt: 注註
# 《大學》、《中庸》中的註釋稱為「章句」，《論語》、《孟子》中的註釋集合了眾人說法，稱為「集注」。後人合稱其為「四書章句集注」，簡稱「四書集注」。
PAT_PREFIX = '原題( *);舊題( *);題( *)'.split(';')
PAT_CLEANAUTHOR = '|'.join(PAT_ROLES + PAT_PREFIX)

def clean(name, check_mode=False):
    if name == '':
        name_cleaned = name
    else:
        name_split = re.split('、|，|；', name)
        name_replaced = [re.sub(PAT_CLEANAUTHOR, ';', n).split(';') for n in name_split]
        name_replaced = chain.from_iterable(name_replaced)
        name_cleaned = [n for n in name_replaced if n != '']

    if check_mode:
        print(name, '->', name_cleaned)
    return name_cleaned

name_lookup = {}
f = open('../notes/names_norm.txt', 'r', encoding='utf-8')
for line in f.readlines():
    line = line.strip()
    if line.startswith('#'):
        continue
    if line == '':
        break
    ori_names, norm_name = line.split(',')[:2]
    ori_names = ori_names.split('|')
    if norm_name == '':
        norm_name = ori_names[0]
    for name_ori in ori_names:
        name_lookup[name_ori] = norm_name

def normalize(name):
    if isinstance(name, str):
        name = [name]
    name_norm = [name_lookup.get(n, n) for n in name]
    return name_norm

def match2time(match):
    # match = ('前1世紀', '前', '1', '世紀')
    year = int(match[2])
    if match[3] in ['世紀', '世纪']:
        year = (year - 1) * 100 + 50
    if match[1] == '前':
        year *= -1
    return year

def author2time(author):
    return author_life.get(author, [-9999])

class Author():
    def __init__(self, name):
        self.name = name
        self.name_clean = clean(self.name)
        self.name_norm = normalize(self.name_clean)
        self.life = [author2time(n) for n in self.name_norm]

if __name__ == "__main__":
    PATH_AUTHOR_TIME = Path('../data/mapping_author_time.pkl')
    if PATH_AUTHOR_TIME.exists():
        with open(PATH_AUTHOR_TIME, 'rb') as f:
            author_life = dict(pickle.load(f))
    else:
        author_life = {}
        print('No mapping for author and time')