#%%
import sys
import json
from pathlib import Path
from tqdm.auto import tqdm
if "../src" not in sys.path: sys.path.append("../src")
from dietcoke.author import Author

OUTPATH = 'author_time_map.json'
CTEXT = Path('../data/dynasty_split/')
AUTHOR_MAP = {}
# a = Author("孔子").author_time


def main():
    for fp in tqdm(list(CTEXT.glob("*.jsonl"))):
        for text in load_jsonl(fp):
            get_author_time(text)
    
    with open(OUTPATH, "w", encoding="utf-8") as f:
        json.dump(AUTHOR_MAP, f, ensure_ascii=False)


def load_jsonl(fp):
    with open(fp, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def get_author_time(text):
    global AUTHOR_MAP
    authors = text.get('author', '').strip()
    for a in authors.split('、'):
        if a == '': continue
        t = Author(a).author_time[0]
        if len(t) == 0: continue
        if a not in AUTHOR_MAP:
            AUTHOR_MAP[a] = t



if __name__ == '__main__':
    main()