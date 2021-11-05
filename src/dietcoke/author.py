from pathlib import Path
import pickle
import re
from opencc import OpenCC
import numpy as np

cc = OpenCC('t2s')

names_excluded = '李心[傳];劉銘[傳];[傳]恒;[傳]遜;孔[傳]金;畢弘[述];陳文[述];崔[述];莊[述]祖'.split(';')
PAT_names_excluded = '|'.join([re.sub('\[|\]', '', n) for n in names_excluded])
PAT_names_excluded = '(' + PAT_names_excluded + ')'

roles = '輯;撰;傳;述;奉敕譯;註;注;編;著;輯註;箋;章句;疏'.split(';')
PAT_roles = '(' + '|'.join(roles) + ')'
# alt: 注註
# 《大學》、《中庸》中的註釋稱為「章句」，《論語》、《孟子》中的註釋集合了眾人說法，稱為「集注」。後人合稱其為「四書章句集注」，簡稱「四書集注」。

prefixes = '原題( *);舊題( *);題( *)'.split(';')
PAT_prefix = '|'.join(prefixes)

name_lookup = {}
f = open('../notes/names_norm.txt', 'r', encoding='utf-8')
for line in f.readlines():
    line = line.strip()
    if line.startswith('#') or line == '':
        continue
    if line == '-----':
        break
    ori_names, norm_name = line.split(',')[:2]
    ori_names = ori_names.split('|')
    if norm_name == '':
        norm_name = ori_names[0]
    for name_ori in ori_names:
        name_lookup[name_ori] = norm_name

PATH_AUTHOR_TIME = Path('../data/author_time/wiki_author_time.pkl')
if PATH_AUTHOR_TIME.exists():
    with open(PATH_AUTHOR_TIME, 'rb') as f:
        author_life = dict(pickle.load(f))
        author_life = {cc.convert(k): v for k,v in author_life.items()}
else:
    author_life = {}
    print('No mapping for author and time')

class Author():
    def __init__(self, name):
        # <name> is a string that can contain several author names
        self.__name = name
        self.name_clean = self.clean()
        self.name_norm = self.normalize()

        self.author_time = self.get_author2time()
        self.rep_year = self.get_rep_year(select_year='mid_year')

    @property
    def name(self):
        return self.__name

    def clean(self, check_mode=False):
        if self.__name == '':
            name_cleaned = self.__name
        else:
            names_ori = re.findall(PAT_names_excluded, self.__name)
            names_protect = (re.sub(PAT_roles, r'<<\1>>', n) for n in names_ori)

            name = self.__name
            for name_ori, name_protect in zip(names_ori, names_protect):
                name = re.sub(name_ori, name_protect, self.__name)
            
            name_replaced = re.sub('([^<])'+PAT_roles, r'\1;', name).split(';')
            name_replaced = [re.sub(PAT_prefix, '', n) for n in name_replaced]
            name_split = [re.split('、|，|；', n) for n in name_replaced]
            name_unprotect = [re.sub('<|>', '', n) for m in name_split for n in m]
            name_cleaned = [n for n in name_unprotect if n != '']

        if check_mode:
            print(self.__name, '->', name_cleaned)
        return name_cleaned

    def normalize(self, in_simplified=True):
        name_norm = [name_lookup.get(n, n) for n in self.name_clean]
        
        if in_simplified:
            name_norm = [cc.convert(n) for n in name_norm]
        return name_norm

    def get_author2time(self, fill_value=-9999, keep_fill=False):
        author_time = [author_life.get(name_norm, [fill_value]) for name_norm in self.name_norm]

        if keep_fill == False:
            author_time = [[tp for tp in tp_lst if tp != -9999] for tp_lst in author_time]
        self.author_time = author_time
        return self.author_time

    def get_author_profile(self, order_by='mid_year'):
        if not hasattr(self, 'author_profile'):
            lst = []
            for name, tp_lst in zip(self.name_norm, self.author_time):
                if len(tp_lst) == 0:
                    continue
                
                assert len(tp_lst) <= 2
                birth_year, mort_year = int(np.min(tp_lst)), int(np.max(tp_lst))
                # birth_year, *_, mort_year = sorted(tp_lst)
                age = round(mort_year - birth_year)
                mid_year = birth_year + round((age/2))

                author_profile = {
                    'name': name,
                    'birth_year': birth_year,
                    'mort_year': mort_year,
                    'mid_year': mid_year,
                    'age': age
                }
                lst.append(author_profile)
            self.author_profile = sorted(lst, key=lambda item: item[order_by])
        return self.author_profile

    def get_rep_year(self, select_year='mid_year', fill_value=-9999):
        try:
            # rep_year = np.max([n[select_year] for n in self.get_author_profile()])
            self.rep_author = sorted(self.get_author_profile(), key=lambda item: item[select_year], reverse=True)[0]
            rep_year = self.rep_author[select_year]
        except:
            rep_year = fill_value
        self.rep_year = rep_year
        return self.rep_year