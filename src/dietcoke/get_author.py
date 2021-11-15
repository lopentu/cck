from tqdm.auto import tqdm
from itertools import chain, groupby
import os
import pickle
import re
from .utils import corpus_lst, dynaspan_lst, save_file, read_file
from .author import Name

PAT_ANONYM = '^$|\[*佚名\]*'
PAT_TIMEPOINT = '((前*)(\d{1,4})(年|世紀|世纪))'
# {{bd|1609年|6月21日|1672年|1月23日|catIdx=W吴}}
PAT_WIKITEXT_LIFE = '\{\{' + '(bd|BD)([^}]+)' + '(\}\})'
PAT_INFOBOX_YEAR = '.*(（|\()(\d+)年.*(）|\)).*'

PATH_NAMES_CLEAN = '../data/author_time/names_clean.txt'
PATH_WIKI_DATA = '../data/author_time/wiki_retrieved_data.pkl'
PATH_WIKI_AUTHOR_TIME = '../data/author_time/wiki_author_time.json'

def get_all_authors(save_names_clean=False):
    authors_tier1, authors_tier12 = [], []
    for corpus in tqdm(corpus_lst(dynaspan_lst + ['tier1'])):
        corpus.read_corpus()
        authors = [line.author for line in corpus.corpus]

        if corpus.dynaspan == 'tier1': #
            authors_tier1 = authors
        else:
            authors_tier12 += authors

    authors_uni = [Name(author).name_clean for author in set(authors_tier12)]
    authors_uni = sorted(list(chain.from_iterable(authors_uni)))

    if save_names_clean:
        fp_out = PATH_NAMES_CLEAN
        with open(fp_out, 'w', encoding='utf-8') as f:
            for author in authors_uni:
                f.write(author + '\n')
        print('File saved:', fp_out)

    return [authors_tier1, authors_tier12, authors_uni]

def get_wiki_data(save_retrieved_data=False):
    # !pip install wptools
    # !pip install wikipedia
    # !pip install wordcloud

    import os
    from tqdm.auto import tqdm
    import wptools

    if not os.path.exists(PATH_WIKI_DATA):
        f = open(PATH_NAMES_CLEAN)

        retrieved_data = {}
        for line in tqdm(f.readlines()):
            author = line.strip()
            try:
                page = wptools.page(author, lang='zh')
                page.get_parse()

                data = page.data
                infobox, wikitext = None, None
                if 'infobox' in data:
                    infobox = data['infobox']
                if 'wikitext' in data:
                    wikitext = data['wikitext']

                retrieved_data[author] = {
                    'data': data,
                    'infobox': infobox,
                    'wikitext': wikitext
                }
            except Exception as e:
                print(e)

    if save_retrieved_data:
        save_file(retrieved_data, PATH_WIKI_DATA)

def get_wiki_author_time(save_wiki_author_time=True):
    retrieved_data = read_file(PATH_WIKI_DATA)

    author_life = []
    for author, data in retrieved_data.items():
        result = None
        if data['wikitext'] is not None:
            try:
                result = wikitext2life(data['wikitext'])
            except Exception as e:
                print(e)

        if (data['infobox'] is not None) \
            and (result is None):
                result = infobox2life(data['infobox'])

        if result is not None:
            author_life.append([Name(author).normalize(in_simplified=False)[0], result])

    author_life = dict(sorted(author_life, key=lambda x: x[1][0]))

    if save_wiki_author_time:
        save_file(author_life, PATH_WIKI_AUTHOR_TIME)

    print('Count of retrieved data:', len(retrieved_data))
    print('Count of author time info:', len(author_life))

def wikitext2life(wikitext):
    result = None
    match = re.search(PAT_WIKITEXT_LIFE, wikitext)
    if match:
        life = []
        for wikitext_frag in match.group(0).split('|'):
            wikitext_timepoints = re.findall(PAT_TIMEPOINT, wikitext_frag)
            if len(wikitext_timepoints) > 0:
                life_timepoint = match2year(wikitext_timepoints[0]) #
                life.append(life_timepoint)

        life = sorted(list(set(life)))
        if len(life) > 2:
            print('Need to check time:', match.group(0), '->', life)
        elif len(life) > 0:
            result = life
    return result

def infobox2life(infobox):
    result = None
    life = []
    for key in ['birth_date', 'death_date']:
        if key in infobox:
            life_timepoint = int(re.sub(PAT_INFOBOX_YEAR, r'\2', infobox[key]))
            life.append(life_timepoint)
    if len(life) > 0:
        result = life
    return result

def match2year(match):
    # match = ('前1世紀', '前', '1', '世紀')
    year = int(match[2])
    if match[3] in ['世紀', '世纪']:
        year = (year - 1) * 100 + 50
    if match[1] == '前':
        year *= -1
    return year

# URL_CHINESE_AUTHOR = 'https://zh.m.wikisource.org/w/api.php?action=query&prop=info&titles=Portal:中国作者&format=json'