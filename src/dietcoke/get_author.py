from tqdm.auto import tqdm
from itertools import chain, groupby
import os
import pickle
import re
from .utils import corpus_lst, dynaspan_lst
from .author import Author

PAT_ANONYM = '^$|\[*佚名\]*'

PATH_NAMES_CLEAN = '../data/author_time/names_clean.txt'
PATH_WIKI_DATA = '../data/author_time/wiki_retrieved_data.pkl'
PATH_WIKI_AUTHOR_TIME = '../data/author_time/wiki_author_time.pkl'

def read_wiki_author_time():
    with open(PATH_WIKI_AUTHOR_TIME, 'rb') as f:
        author_life = pickle.load(f)
    return author_life

def get_all_authors(save_names_clean=False):
    authors_tier1, authors_tier12 = [], []
    for corpus in tqdm(corpus_lst(dynaspan_lst + ['tier1'])):
        corpus.read_corpus()
        authors = [line.author for line in corpus.corpus]
        if corpus.dynaspan == 'tier1':
            authors_tier1 = authors
        else:
            authors_tier12 += authors

    authors_uni = [Author(author).name_clean for author in set(authors_tier12)]
    authors_uni = sorted(list(chain.from_iterable(authors_uni)))

    if save_names_clean:
        fp_out = PATH_NAMES_CLEAN
        with open(PATH_NAMES_CLEAN, 'w', encoding='utf-8') as f:
            for author in authors_uni:
                f.write(author + '\n')
        print('File saved:', fp_out)

    return [authors_tier1, authors_tier12, authors_uni]

def get_wiki_data(save_retrieved_data=False):
    # !pip install wptools
    # !pip install wikipedia
    # !pip install wordcloud

    import os
    import pickle
    from tqdm.auto import tqdm
    import wptools

    fp_out = PATH_WIKI_DATA

    if not os.path.exists(fp_out):
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
        with open(fp_out, 'wb') as f_out:
            pickle.dump(retrieved_data, f_out)
        print('File saved:', fp_out)

def get_wiki_author_time(save_wiki_author_time=True):
    # {{bd|1609年|6月21日|1672年|1月23日|catIdx=W吴}}
    PAT_WIKITEXT_LIFE = '\{\{' + '(bd|BD)([^}]+)' + '(\}\})'
    PAT_INFOBOX_YEAR = '.*(（|\()(\d+)年.*(）|\)).*'
    PAT_TIMEPOINT = '((前*)(\d{1,4})(年|世紀|世纪))'

    with open(PATH_WIKI_DATA, 'rb') as f_in:
        retrieved_data = pickle.load(f_in)

    author_life = []
    for author, data in retrieved_data.items():
        match = None
        result = None
        if data['wikitext'] is not None:
            try:
                match = re.search(PAT_WIKITEXT_LIFE, data['wikitext'])
                if match:
                    life = []
                    for n in match.group(0).split('|'):
                        life_timepoints = re.findall(PAT_TIMEPOINT, n)
                        if len(life_timepoints) > 0:
                            life.append(match2time(life_timepoints[0]))

                    life = list(set(life))
                    if len(life) > 2:
                        print('Need to check time:', match.group(0), '->', life)
                    elif len(life) > 0:
                        result = life
                        author_life.append([Author(author).name_norm[0], life])
            except Exception as e:
                print(e)

        if match == None:
            if data['infobox'] is not None:
                life_year = [None, None]
                for idx, key in enumerate(['birth_date', 'death_date']):
                    if key in data['infobox']:
                        life_year[idx] = int(re.sub(PAT_INFOBOX_YEAR, r'\2', data['infobox'][key]))
                life = [n for n in life_year if n != None]
                if len(life) > 0:
                    result = life

        if result is not None:
            author_life.append([Author(author).name_norm[0], result])

    author_life = sorted(author_life)
    author_life = list(k for k,_ in groupby(author_life))
    author_life = sorted(author_life, key=lambda x: x[1][0])

    if save_wiki_author_time:
        with open(PATH_WIKI_AUTHOR_TIME, 'wb') as f:
            pickle.dump(author_life, f)

    print('Count of retrieved data:', len(retrieved_data))
    print('Count of author time info:', len(author_life))

def match2time(match):
    # match = ('前1世紀', '前', '1', '世紀')
    year = int(match[2])
    if match[3] in ['世紀', '世纪']:
        year = (year - 1) * 100 + 50
    if match[1] == '前':
        year *= -1
    return year

# URL_CHINESE_AUTHOR = 'https://zh.m.wikisource.org/w/api.php?action=query&prop=info&titles=Portal:中国作者&format=json'