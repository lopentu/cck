from itertools import zip_longest
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from tqdm.auto import tqdm

# matplotlib.font_manager.fontManager.addfont('../../TaipeiSansTCBeta-Light.ttf')
# matplotlib.rc('font', family='Taipei Sans TC Beta')

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
        fig = plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)
        self.plot_vgc_curve(vgc_df)

        plt.subplot(1, 2, 2)
        self.plot_vgc_residuals(vgc_df)

        fig.tight_layout(pad=3.0)

    def calc_char_freq_by_text_slice(self, char, moving_window=3, return_df=True):
        if self.char_freq_by_text_slice is None:
            self.char_freq_by_text_slice = [Counter(text_slice) for text_slice in self.text_slices]

        freq_lst = [n.get(char, 0) for n in self.char_freq_by_text_slice]
        if return_df:
            df = pd.DataFrame({'text_slice': self.ntokens, 'freq': freq_lst})
            df['running_median'] = df['freq'].rolling(moving_window, min_periods=1).median()
            return df
        else:
            return freq_lst

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

    def get_underdisperse_df(self):
        d_lst = []
        char_lst = []
        for char in self.char_freq:
            freq_lst = self.calc_char_freq_by_text_slice(char, return_df=False)
            d = sum([1 for freq in freq_lst if freq > 0])
            d_lst.append(d)
            char_lst.append(char)
            
        zscore_lst = self.__get_zscore_lst(d_lst)
        underdisperse_df = pd.DataFrame(
            {'char': char_lst, 'd': d_lst, 'zscore': zscore_lst})
        underdisperse_df['underdispersed'] = underdisperse_df['zscore'].apply(lambda x: abs(x) >= 3)
        self.underdisperse_df = underdisperse_df

        self.underdisperse_chars = self.underdisperse_df[self.underdisperse_df['underdispersed']]['char'].values
        self.d = dict(zip(char_lst, d_lst))
        return self.underdisperse_df

    def __get_zscore_lst(self, lst):
        zscore_lst = np.array(lst)
        zscore_lst = zscore_lst.astype('float')
        zscore_lst[zscore_lst == 0] = np.nan
        zscore_lst = stats.zscore(zscore_lst, nan_policy='omit')
        return zscore_lst

    def calc_d_k_threshold(self):
        d_k_threshold = {}
        for char, freq in self.char_freq.items():
            d = self.d.get(char, 0)
            if d > 0:
                d_k_threshold[char] = freq / d
        self.d_k_threshold = d_k_threshold
        return d_k_threshold

    def is_d_k(self, char, d_k_threshold, f_k):
        result = 0
        if char in self.underdisperse_chars:
            if d_k_threshold >= f_k:
                result = 1
        return result

    def get_VU(self):
        df_lst = []
        for char in tqdm(self.underdisperse_chars):
            df = self.calc_char_freq_by_text_slice(char)
            df['VU_k'] = df.apply(lambda row: self.is_d_k(char, self.d_k_threshold.get(char, 0), row['freq']), axis=1) #
            df['NU_k'] = df.apply(lambda row: row['VU_k'] * row['freq'], axis=1)
            df_lst.append(df)

        VU_df = pd.concat(df_lst)
        VU_df = VU_df.drop(['freq', 'running_median'], axis=1) \
                .rename({'text_slice': 'k'}, axis=1) \
                .groupby('k').sum().reset_index()

        VU_df['running_median_VU_k'] = VU_df['VU_k'].rolling(5, min_periods=1).median()
        VU_df['running_median_NU_k'] = VU_df['NU_k'].rolling(5, min_periods=1).median()

        VU_df['yhat_ls_VU_k'] = smf.ols(formula='VU_k ~ k', data=VU_df).fit().predict(VU_df[['k']])
        VU_df['yhat_ls_NU_k'] = smf.ols(formula='NU_k ~ k', data=VU_df).fit().predict(VU_df[['k']])

        self.VU_df = VU_df
        return self.VU_df

    def plot_U(self, U_var='VU'):
        U_col = U_var + '_k'
        plt.scatter(self.VU_df['k'], self.VU_df[U_col], s=5)
        plt.plot(self.VU_df['k'], self.VU_df[f'running_median_{U_col}'])
        plt.plot(self.VU_df['k'], self.VU_df[f'yhat_ls_{U_col}'], linestyle='dotted')
        plt.xlabel('k: text slice')

        if U_var == 'VU':
            plt.ylabel(f'VU(k):\nnumber of underdispersed types')
        elif U_var == 'NU':
            plt.ylabel(f'NU(k):\nnumber of underdispersed tokens')
        else:
            raise

    def plot_VU(self):
        self.plot_U()

    def plot_NU(self):
        self.plot_U(U_var='NU')

    def plot_U_acf(self, U_var='VU'):
        U_col = U_var + '_k'

        sm.graphics.tsa.plot_acf(self.VU_df[U_col].values)
        plt.xlabel('Lag')
        plt.ylabel('ACF')
        plt.title(f'Series: {U_var}')

    def get_Pr_type(self):
        VU_df = self.VU_df

        VU_df['VU_k_prev'] = VU_df['VU_k'].shift(1)
        VU_df['VU_k_new'] = VU_df['VU_k'] - VU_df['VU_k_prev']

        VU_df = self.vgc_df[['M', 'V']] \
                .rename({'M': 'k', 'V': 'V_k'}, axis=1) \
                .merge(VU_df)
        VU_df['V_k_prev'] = VU_df['V_k'].shift(1)
        VU_df['V_k_new'] = VU_df['V_k'] - VU_df['V_k_prev']

        VU_df['Pr_k_type'] = VU_df['VU_k_new'] / VU_df['V_k_new']
        self.VU_df = VU_df
        return self.VU_df

    def plot_Pr_type(self):
        plt.scatter(self.VU_df['k'], self.VU_df['Pr_k_type'], s=5)
        plt.xlabel('k: text slice')
        plt.ylabel('Pr(U,type):\nproportion of new underdispersed types')