from pathlib import Path
import json
from itertools import chain
from collections import Counter
from tqdm.auto import tqdm
import multiprocessing as mp
import time

class Text: # 將資料存在 class 裡，在運算上比較有效率嗎？還是能夠節省記憶體？
    def __init__(self, line):
        self.obj = json.loads(line)
        self.urn = self.obj['urn']

        # self.texts = [n['c'] for n in obj['text']]
        texts = []
        for n in self.obj['text']:
            if isinstance(n['c'], str):
                texts.append(n['c'])
            else:
                flatten = list(chain.from_iterable(n['c']))
                texts.append(flatten)
        self.texts = texts

    def get_ch_freq(self):
        ch_freq = Counter(chain.from_iterable(self.texts))
        return ch_freq

def get_ch_freq_by_line(line):
    data = Text(line)
    ch_freq_line = data.get_ch_freq()
    return ch_freq_line

def get_ch_freq_by_dyna(fp): # 以一個jsonl檔案的字頻返回值
    f = open(fp, 'r', encoding='utf-8')
    corpus = f.readlines()

    pool = mp.Pool(processes=mp.cpu_count())
    ch_freq_lines = pool.map(get_ch_freq_by_line, corpus)
    pool.close()

    ch_freq = Counter()
    for n in tqdm(ch_freq_lines):
        ch_freq += n

    return ch_freq

def write_dictionary(ch_freq):
    with open('dictionary.txt', 'w') as out_f:
        for n in tqdm(ch_freq.most_common()):
            if len(n[0]) != 1:
                print(n[0])
                continue
            out_f.write(n[0])
            out_f.write('\n')

def main():

    start_time = time.time()

    ch_freq = Counter()
    for fp in Path('../../dynasty_split/').glob('*.jsonl'):
        print(fp)
        ch_freq += get_ch_freq_by_dyna(fp)

    print(ch_freq.most_common(10))
    write_dictionary(ch_freq)

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()