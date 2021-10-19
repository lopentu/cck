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

matplotlib.font_manager.fontManager.addfont('../../TaipeiSansTCBeta-Light.ttf')
matplotlib.rc('font', family='Taipei Sans TC Beta')

class Growth():
    def __init__(self, texts, chunk_size):
        self.texts = texts
        self.chunk_size = chunk_size

        self._calc_char_freq()
        self._get_text_slices()

        self.group_freq = None
        self.char_freq_by_text_slice = None

    def _get_text_slices(self):
        # source: https://docs.python.org/3/library/itertools.html
        iterable = self.texts
        n = self.chunk_size

        args = [iter(iterable)] * n
        text_slices = zip_longest(*args, fillvalue='')
        text_slices = [''.join(char_lst) for char_lst in text_slices]

        self.text_slices = text_slices#[1:]
        self.nchunks = len(self.text_slices)
        self.ntokens_start = [i*self.chunk_size for i in range(self.nchunks)]
        self.ntokens_end = [(i+1)*self.chunk_size for i in range(self.nchunks)]
        return self.text_slices

    def _calc_char_freq(self):
        self.char_freq = Counter(self.texts)
        self.V = len(self.char_freq)
        self.N = len(self.texts)

    def _calc_group_freq(self):
        if self.group_freq is None:
            group_freq = defaultdict(list)
            for char, freq in self.char_freq.items():
                group_freq[freq].append(char)

            self.group_freq = group_freq
            self.freq_freq = {freq: len(char_lst) for freq, char_lst in self.group_freq.items()}
        return self.group_freq

    def _calc_expected_V(self, text_slice_idx, return_all=True):
        # formula (1)
        self._calc_group_freq()

        text_slice_cum = ''.join(self.text_slices[:text_slice_idx+1])
        N = len(text_slice_cum)

        freq_freq_sum = sum([freq*((1-(N/self.N))**freq_group) for freq_group, freq in self.freq_freq.items()])
        expected_V = self.V - freq_freq_sum

        M = N-self.chunk_size
        V = len(Counter(text_slice_cum))

        if return_all:
            return [M, N, V, expected_V, freq_freq_sum]
        else:
            return expected_V

    def get_vgc(self):
        data = []
        for i in range(self.nchunks):
            try:
                data.append(self._calc_expected_V(i))
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

    def calc_char_freq_by_text_slice(self, char, moving_window=5, return_df=True):
        if self.char_freq_by_text_slice is None:
            self.char_freq_by_text_slice = [Counter(text_slice) for text_slice in self.text_slices]

        freq_lst = [cnt.get(char, 0) for cnt in self.char_freq_by_text_slice]
        if return_df:
            df = pd.DataFrame({'text_slice': self.ntokens_end, 'freq': freq_lst})
            return df
        else:
            return freq_lst

    def plot_char_freq_by_text_slice(self, char, moving_window=5, s=5):
        df = self.calc_char_freq_by_text_slice(char, moving_window)
        plt.scatter(df['text_slice'], df['freq'], s=s)
        self._plot_w_ref_line(df, 'text_slice', 'freq', ['running_median'])
        plt.xlabel('text slice')
        plt.ylabel('frequency')
        plt.title(f'Word usage of {char}\n(TS smoother using running medians)')

    def _add_running_median(self, df, x_var, y_var, moving_window=5):
        df[f'running_median_{y_var}'] = df[y_var].rolling(moving_window, min_periods=1).median()
        plt.plot(df[x_var], df[f'running_median_{y_var}'], linestyle='dotted')

    def _add_ls_yhat(self, df, x_var, y_var):
        est = smf.ols(formula=f'{y_var} ~ {x_var}', data=df).fit()
        df[f'yhat_ls_{y_var}'] = est.predict(df[[x_var]])
        return df

    def _add_ls_line(self, df, x_var, y_var):
        df = self._add_ls_yhat(df, x_var, y_var)
        plt.plot(x_var, f'yhat_ls_{y_var}', data=df, linestyle='dotted')

    def _add_scatterplot_smoother(self, df, x_var, y_var):
        sns.regplot(x_var, y_var, data=df, lowess=True,
        line_kws={'ls': 'dashed'}, scatter_kws={'s': 5})

    def _plot_w_ref_line(self, df, x_var, y_var, ref):
        if 'ls' in ref:
            self._add_ls_line(df, x_var, y_var)
        if 'scatterplot_smoother' in ref:
            self._add_scatterplot_smoother(df, x_var, y_var)
        if 'running_median' in ref:
            self._add_running_median(df, x_var, y_var)

    def get_prog_err_scores_df(self):
        if not hasattr(self, 'vgc_df'):
            raise ValueError('<vgc_df> does not exists.')

        df = self.vgc_df
        df['V_prev'] = df['V'].shift(1, fill_value=0)
        df['expected_V_prev'] = df['expected_V'].shift(1, fill_value=0)
        df['prog_err_score'] = (df['expected_V'] - df['expected_V_prev']) - (df['V'] - df['V_prev'])

        self.prog_err_scores_df = df
        return self.prog_err_scores_df
        # est.params['M'], est.pvalues['M']

    def plot_prog_err_scores(self, prog_err_scores_df):
        plt.axhline(y=0, color='gray')
        self._plot_w_ref_line(prog_err_scores_df, 'N', 'prog_err_score', ref=['ls', 'scatterplot_smoother'])
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
        underdisperse_df['is_underdispersed'] = underdisperse_df['zscore'].apply(lambda x: abs(x) >= 2.57)
        self.underdisperse_df = underdisperse_df

        self.underdisperse_chars = self.underdisperse_df[self.underdisperse_df['is_underdispersed']]['char'].values
        self.d = dict(zip(char_lst, d_lst))
        return self.underdisperse_df

    def __get_zscore_lst(self, lst):
        zscore_lst = np.array(lst)
        zscore_lst = zscore_lst.astype('float')
        zscore_lst[zscore_lst == 0] = np.nan
        zscore_lst = stats.zscore(zscore_lst, nan_policy='omit')
        return zscore_lst

    def get_d_f_threshold(self):
        d_f_threshold = {}
        for char, freq in self.char_freq.items():
            d = self.d.get(char, 0)
            if d > 0:
                d_f_threshold[char] = freq / d
            else:
                print(f'{char} has dispersion of 0!?')
        self.d_f_threshold = d_f_threshold
        return d_f_threshold

    def calc_d_k_indicator(self, char, f_k):
        if not hasattr(self, 'd_f_threshold'):
            raise ValueError('<d_f_threshold> does not exists.')

        result = 0
        if (char in self.underdisperse_chars) and \
            (char in self.d_f_threshold):
            if self.d_f_threshold[char] >= f_k:
                result = 1
        return result

    def get_U_df(self):
        df_lst = []
        for char in tqdm(self.underdisperse_chars): #
            df = self.calc_char_freq_by_text_slice(char)
            df['VU_k'] = df.apply(lambda row: self.calc_d_k_indicator(char, row['freq']), axis=1)
            df['NU_k'] = df.apply(lambda row: row['VU_k'] * row['freq'], axis=1)
            df['char'] = char
            df_lst.append(df)
        U_df = pd.concat(df_lst)

        U_df = U_df.drop('freq', axis=1) \
                .rename({'text_slice': 'k'}, axis=1) \
                .groupby('k').sum().reset_index()

        self.U_df = U_df
        return self.U_df

    def _plot_U(self, U_df, U_var='VU'):
        U_col = U_var + '_k'
        plt.scatter(U_df['k'], U_df[U_col], s=5)
        self._plot_w_ref_line(U_df, 'k', U_col, ['running_median', 'ls'])
        plt.xlabel('k: text slice')

        if U_var == 'VU':
            plt.ylabel(f'VU(k):\nnumber of underdispersed types')
        elif U_var == 'NU':
            plt.ylabel(f'NU(k):\nnumber of underdispersed tokens')
        else:
            raise

    def plot_VU(self, U_df):
        self._plot_U(U_df)

    def plot_NU(self, U_df):
        self._plot_U(U_df, U_var='NU')

    def _plot_U_acf(self, U_df, U_var='VU'):
        U_col = U_var + '_k'

        sm.graphics.tsa.plot_acf(U_df[U_col].values)
        plt.xlabel('Lag')
        plt.ylabel('ACF')
        plt.title(f'Series: {U_var}')

    def plot_VU_acf(self, U_df):
        self._plot_U_acf(U_df)

    def plot_NU_acf(self, U_df):
        self._plot_U_acf(U_df, U_var='NU')

    def get_Pr_df(self):
        df_lst = []
        for char in tqdm(self.char_freq): #
            df = self.calc_char_freq_by_text_slice(char)
            df['d_k_indicator'] = df.apply(lambda row: self.calc_d_k_indicator(char, row['freq']), axis=1)
            df['d_k_f'] = df.apply(lambda row: row['d_k_indicator'] * row['freq'], axis=1)
            df['char'] = char
            df_lst.append(df)
        Pr_df = pd.concat(df_lst)

        # count
        Pr_df = Pr_df[Pr_df['freq'] > 0]
        Pr_df = Pr_df.groupby(['char', 'd_k_indicator']) \
            .first('freq').reset_index().assign(first=True) \
            .groupby(['text_slice', 'd_k_indicator']) \
            .sum(['freq', 'first']) \
            .drop('d_k_f', axis=1) \
            .rename({'first': 'type', 'freq': 'token'}, axis=1)
        # proportion
        Pr_df = Pr_df.groupby(level=0) \
            .apply(lambda x: x / x.sum()).reset_index() \
            .rename({'type': 'Pr_type', 'token': 'Pr_token'}, axis=1) \
            .merge(Pr_df.reset_index()) \
            .rename({'text_slice': 'k'}, axis=1)
        Pr_abridf = Pr_df[Pr_df['d_k_indicator'] == 1]
        self.Pr_df = Pr_df
        self.Pr_abridf = Pr_abridf
        return self.Pr_abridf

    def _plot_Pr(self, Pr_abridf, var='type'):
        col = f'Pr_{var}'
        plt.scatter(Pr_abridf['k'], Pr_abridf[col], s=5)
        plt.xlabel('k: text slice')
        plt.ylabel(f'Pr(U,{var}):\nproportion of new underdispersed {var}s')

    def plot_Pr_type(self, Pr_abridf):
        self._plot_Pr(Pr_abridf)

    def plot_Pr_token(self, Pr_abridf):
        self._plot_Pr(Pr_abridf, var='token')