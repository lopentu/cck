from pathlib import Path
import re
from opencc import OpenCC
import numpy as np
from .utils import read_file

PATH_WIKI_AUTHOR_TIME = Path('../data/author_time/wiki_author_time.json')
PATH_NAMES_NORM = Path('../notes/names_norm.txt')

cc = OpenCC('t2s')

names_excluded = '李心[傳];劉銘[傳];[傳]恒;[傳]遜;孔[傳]金;畢弘[述];陳文[述];崔[述];莊[述]祖'
names_excluded = (names_excluded + cc.convert(names_excluded)).split(';')
PAT_names_excluded = '|'.join([re.sub('\[|\]', '', n) for n in names_excluded])
PAT_names_excluded = '(' + PAT_names_excluded + ')'

roles = '輯;撰;傳;述;奉敕譯;註;注;編;著;輯註;箋;章句;疏'
roles = (roles + cc.convert(roles)).split(';')
PAT_roles = '(' + '|'.join(roles) + ')'
# alt: 注註
# 《大學》、《中庸》中的註釋稱為「章句」，《論語》、《孟子》中的註釋集合了眾人說法，稱為「集注」。後人合稱其為「四書章句集注」，簡稱「四書集注」。

prefixes = '原題( *);舊題( *);題( *)'
prefixes = (prefixes + cc.convert(prefixes)).split(';')
PAT_prefix = '|'.join(prefixes)

name_lookup = {}
if PATH_NAMES_NORM.exists():
    f = open(PATH_NAMES_NORM, 'r', encoding='utf-8')
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

def get_author_time_map(merge_data=False):
    wiki_data = read_file(PATH_WIKI_AUTHOR_TIME)
    if merge_data:
        result = {}
        for fp in ['author_time_map.json', '../notes/post_edit.json']:
            result = {**result, **read_file(fp)}
        result = {**result, **wiki_data}
    else:
        result = wiki_data
    return {cc.convert(k): v for k,v in result.items()}

if PATH_WIKI_AUTHOR_TIME.exists():
    print('Mapping author time ...')
    author_time_map = get_author_time_map()
else:
    author_time_map = None
    print('No author time map ...')

class Name:
    def __init__(self, name):
        # <name> is a string that can contain several author names
        self.__name = name
        self.name_clean = self.clean()
        self.name_norm = self.normalize()

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

class Author(Name):
    def __init__(self, name):
        super().__init__(name)
        self.rep_year = self.get_rep_year

    @property
    def get_author_profile(self, fill_value=-9999, order_by='mid_year'):
        if not hasattr(self, 'author_profile'):
            author_profile = []
            for name_norm in self.name_norm:
                life_timepoints = author_time_map.get(name_norm, None)
                if life_timepoints is not None:
                    birth_year, death_year = np.min(life_timepoints), np.max(life_timepoints)

                    if birth_year == death_year:
                        age = fill_value
                        mid_year = birth_year
                    else:
                        age = round(death_year - birth_year)
                        mid_year = birth_year + round((age/2))

                    author_profile.append({
                        'name': name_norm,
                        'birth_year': birth_year,
                        'death_year': death_year,
                        'mid_year': mid_year,
                        'age': age
                    })
            self.author_profile = sorted(author_profile, key=lambda item: item[order_by])
        return self.author_profile

    @property
    def get_rep_year(self, select_year='mid_year', fill_value=-9999):
        rep_author_profile = sorted(self.get_author_profile, key=lambda item: item[select_year], reverse=True)
        if len(rep_author_profile) > 0:
            rep_year = rep_author_profile[0].get(select_year, fill_value)
        else:
            rep_year = fill_value
        return rep_year