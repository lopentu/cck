from itertools import zip_longest
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.font_manager.fontManager.addfont('../../TaipeiSansTCBeta-Light.ttf')
matplotlib.rc('font', family='Taipei Sans TC Beta')

class Growth():
    def __init__(self, texts, chunk_size):
        self.texts = texts
        self.chunk_size = chunk_size

        self._get_text_slices()
        self._calc_char_freq()
        self.N = len(self.texts)

        self.group_freq = None
        self.char_freq_by_text_slice = None

    def _get_text_slices(self):
        # source: https://docs.python.org/3/library/itertools.html
        iterable = self.texts
        n = self.chunk_size

        args = [iter(iterable)] * n
        text_slices = zip_longest(*args, fillvalue='')
        text_slices = [''.join(char_lst) for char_lst in text_slices]

        self.text_slices = text_slices[1:]
        self.nchunks = len(self.text_slices)
        self.ntokens = [i*self.chunk_size for i in range(self.nchunks)]
        return self.text_slices

    def _calc_char_freq(self):
        self.char_freq = Counter(self.texts)
        self.V = len(self.char_freq)

    def _calc_group_freq(self):
        if self.group_freq is None:
            group_freq = defaultdict(list)
            for char, freq in self.char_freq.items():
                group_freq[freq].append(char)

            self.group_freq = group_freq
            self.freq_freq = {freq: len(char_lst) for freq, char_lst in self.group_freq.items()}
        return self.group_freq

    def calc_expected_V(self, text_slice_idx, return_all=True):
        # formula (1)
        self._calc_group_freq()

        text_slice = ''.join(self.text_slices[:text_slice_idx+1])
        N = len(text_slice)

        freq_freq_sum = sum([freq*((1-(N/self.N))**freq_group) for freq_group, freq in self.freq_freq.items()])
        expected_V = self.V - freq_freq_sum

        M = N-self.chunk_size
        V = len(Counter(text_slice))

        if return_all:
            return [M, N, V, expected_V, freq_freq_sum]
        else:
            return expected_V

    def get_vgc(self):
        data = []
        for i in range(self.nchunks):
            try:
                data.append(self.calc_expected_V(i))
            except:
                print(i)

        df = pd.DataFrame(data, columns=['M', 'N', 'V', 'expected_V', 'freq_freq_sum'])
        df['diff_V'] = df['expected_V'] - df['V']
        self.vgc_df = df
        return self.vgc_df

    def plot_vgc_curve(self, vgc_df, color_expected_V='red'):
        plt.plot(vgc_df['N'], vgc_df['expected_V'], c=color_expected_V)
        plt.plot(vgc_df['N'], vgc_df['V'], linestyle='dotted')
        plt.xlabel('N')
        plt.ylabel('V(N): dotted, E[V(N)]: solid')

    def plot_vgc_residuals(self, vgc_df):
        plt.plot(vgc_df['N'], vgc_df['diff_V'], linestyle='dotted', marker='o', markersize=2.5)
        plt.axhline(y=0, linestyle='-', color='gray')
        plt.xlabel('N')
        plt.ylabel('E[V(N)] - V(N)')

    def plot_vgc(self, vgc_df, figsize=(10, 5)):
        plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)
        self.plot_vgc_curve(vgc_df)

        plt.subplot(1, 2, 2)
        self.plot_vgc_residuals(vgc_df)

    def calc_char_freq_by_text_slice(self, char, moving_window=3):
        if self.char_freq_by_text_slice is None:
            self.char_freq_by_text_slice = [Counter(text_slice) for text_slice in self.text_slices]

        freq_lst = [n[char] for n in self.char_freq_by_text_slice]
        df = pd.DataFrame({'text_slice': self.ntokens, 'freq': freq_lst})
        df['running_median'] = df['freq'].rolling(moving_window, min_periods=1).median()
        return df

    def plot_char_freq_by_text_slice(self, char, moving_window=3, s=5):
        df = self.calc_char_freq_by_text_slice(char, moving_window)
        plt.scatter(df['text_slice'], df['freq'], s=s)
        plt.plot(df['text_slice'], df['running_median'], linestyle='dotted')
        plt.xlabel('text slice')
        plt.ylabel('frequency')
        plt.title(f'Word usage of {char}\n(TS smoother using running medians)')

    def get_prog_err_scores(self):
        if not hasattr(self, 'vgc_df'):
            raise ValueError('<vgc_df> does not exists.')
        df = self.vgc_df
        df['expected_V_prev'] = df['expected_V'].shift(1, fill_value=0)
        df['V_prev'] = df['V'].shift(1, fill_value=0)
        df['prog_err_score'] = (df['expected_V'] - df['expected_V_prev']) - (df['V'] - df['V_prev'])

        est = smf.ols(formula='prog_err_score ~ M', data=df).fit()
        df['yhat_ls'] = est.predict(df[['M']])
        self.prog_err_scores_df = df
        return self.prog_err_scores_df
        # est.params['M'], est.pvalues['M']

    def plot_prog_err_scores(self):
        plt.axhline(y=0, color='gray')
        plt.plot('M', 'yhat_ls', data=self.prog_err_scores_df, linestyle='dotted')
        sns.regplot('M', 'prog_err_score', data=self.prog_err_scores_df, lowess=True, line_kws={'ls': 'dashed'}, scatter_kws={'s': 5})
        plt.xlabel('k: text slice')
        plt.ylabel('D(k):\nProgressive difference error scores')
        plt.title('Error scores for the influx of new types')