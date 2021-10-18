import sys, os
from pathlib import Path
if "../src" not in sys.path:
    sys.path.append("../src")

import re
import pandas as pd
import json
import pickle
import itertools
from itertools import chain
from collections import defaultdict
import requests
from dietcoke import dynaspan_lst, corpus_lst, Text, Author, match2time

PAT_ANONYM = '^$|\[*佚名\]*'

authors_tier1 = []
authors_tier12 = []
for corpus in corpus_lst(dynaspan_lst + ['tier1']):
    corpus.read_corpus()
    authors = [Text(line).author for line in corpus.corpus]
    if corpus.dynaspan == 'tier1':
        authors_tier1 = authors
    else:
        authors_tier12 += authors

## print
cnt_tier1 = sum([not re.match(PAT_ANONYM, n) for n in authors_tier1])
cnt_tier12 = sum([not re.match(PAT_ANONYM, n) for n in authors_tier12])

print('---\nCount of tier1:', cnt_tier1, '/', len(authors_tier1))
print('Coverage of tier1:', round(cnt_tier1 / len(authors_tier1), 2))

print('---\nCount of tier12:', cnt_tier12, '/', len(authors_tier12))
print('Coverage of tier12:', round(cnt_tier12 / len(authors_tier12), 2))