from itertools import zip_longest
from collections import Counter, defaultdict
import re
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
    def __init__(self, text, chunk_size):
        self.moving_window = 5

        self.__text = text
        self.__chunk_size = chunk_size

        self._calc_char_freq()

    def change_moving_window(self, new_moving_window):
        self.moving_window = new_moving_window

    @property
    def text(self):
        return self.__text

    @property
    def chunk_size(self):
        return self.__chunk_size

    def _calc_char_freq(self):
        self.char_freq = Counter(self.__text)
        self.V = len(set(self.__text))
        self.N = len(self.__text)

    def _calc_group_freq(self):
        group_freq = defaultdict(list)
        for char, freq in self.char_freq.items():
            group_freq[freq].append(char)
        self.group_freq = group_freq
        return self.group_freq

    @property
    def get_group_freq(self):
        if not hasattr(self, 'group_freq'):
            self._calc_group_freq()
        return self.group_freq

    @property
    def get_freq_freq(self):
        if not hasattr(self, 'freq_freq'):
            self._calc_group_freq()
            self.freq_freq = {freq: len(char_lst) for freq, char_lst in self.group_freq.items()}
        return self.freq_freq

    def _add_running_median(self, df, x_var, y_var):
        df[f'running_median_{y_var}'] = df[y_var].rolling(self.moving_window, min_periods=1).median()

        plt.plot(df[x_var], df[f'running_median_{y_var}'], linestyle='dotted')

    def _add_ls_line(self, df, x_var, y_var):
        est = smf.ols(formula=f'{y_var} ~ {x_var}', data=df).fit()
        df[f'yhat_ls_{y_var}'] = est.predict(df[[x_var]])

        plt.plot(x_var, f'yhat_ls_{y_var}', data=df, linestyle='dotted')
        print(est.summary())

    def _add_scatterplot_smoother(self, df, x_var, y_var, hide_scatter=False):
        if hide_scatter:
            sns.regplot(x_var, y_var, data=df, lowess=True,
        line_kws={'ls': 'dashed'}, scatter_kws={'s': 5, 'alpha': 0})
        else:
            sns.regplot(x_var, y_var, data=df, lowess=True,
        line_kws={'ls': 'dashed'}, scatter_kws={'s': 5})

    def _plot_w_ref_line(self, df, x_var, y_var, ref, hide_scatter=False):
        if 'ls' in ref:
            self._add_ls_line(df, x_var, y_var)
        if 'scatterplot_smoother' in ref:
            self._add_scatterplot_smoother(df, x_var, y_var, hide_scatter=hide_scatter)
        if 'running_median' in ref:
            self._add_running_median(df, x_var, y_var)

    def get_text_slices(self, iterable=None, drop_last=False):
        # source: https://docs.python.org/3/library/itertools.html
        if iterable is None:
            iterable = self.__text
        n = self.__chunk_size

        args = [iter(iterable)] * n
        text_slices = zip_longest(*args, fillvalue='')

        for k, text_slice in enumerate(text_slices):
            if drop_last and text_slice[-1] == '':
                break
            yield k, text_slice

    def get_char_occ(self, text_slice, char):
        occs = []
        offset = -1
        while True:
            try:
                offset = text_slice.index(char, offset+1)
                occs.append(offset)
            except ValueError:
                break
        return occs

    def calc_char_freq_by_text_slice(self, char, return_df=True):
        k_freq = [(k, np.log1p(len(self.get_char_occ(text_slice, char)))) for k, text_slice in self.get_text_slices()]

        if return_df:
            df = pd.DataFrame(k_freq)
            df.columns = ('k', 'freq')

            df['freq_prev'] = df['freq'].shift(1, fill_value=np.log1p(0))
            df['freq_diff'] = df['freq'] - df['freq_prev']
            
            return df
        else:
            return k_freq

    def plot_char_freq_by_text_slice(self, char, show_change=True, s=5):
        df = self.calc_char_freq_by_text_slice(char)

        if show_change:
            df.set_index('k')['freq_diff'].plot()
            plt.ylabel('frequency change')
        else:
            df.set_index('k')['freq'].plot()
            plt.ylabel('frequency')

        plt.xlabel('text slice')
        plt.title(f'Word usage of {char}')

    def _calc_expected_V(self):
        expected_V_lst = []
        for k, _ in self.get_text_slices():
            N = (k+1) * self.__chunk_size
            freq_freq_sum = sum([freq*((1-(N/self.N))**freq_group) for freq_group, freq in self.get_freq_freq.items()])
            expected_V = self.V - freq_freq_sum
            expected_V_lst.append(expected_V)
        self.expected_V = expected_V_lst
        return self.expected_V

    @property
    def get_expected_V(self):
        if not hasattr(self, 'expected_V'):
            self._calc_expected_V()
        return self.expected_V

    def calc_chunked_V(self, iterable=None, prevV=set()):
        if iterable is None:
            iterable = self.__text
            is_ori_text = True
        else:
            is_ori_text = False

        Vset = set() | prevV
        xs = []
        ys = []
        for i, chunk in self.get_text_slices(iterable=iterable):
            xs.append(i)
            Vset |= set(chunk)
            ys.append(len(Vset))

        if is_ori_text:
            self.chunked_V = ys

        return (xs, ys, Vset)

    @property
    def get_chunked_V(self):
        if not hasattr(self, 'chunked_V'):
            self.calc_chunked_V()
        return self.chunked_V

    def get_vgc(self):
        if not hasattr(self, 'vgc_df'):
            BASE_DF = pd.DataFrame({
                'N': [k*self.__chunk_size for k,_ in self.get_text_slices()],
                'expected_V': self.get_expected_V,
                'chunked_V': self.get_chunked_V,
            })
            BASE_DF['V_diff'] = BASE_DF['expected_V'] - BASE_DF['chunked_V']
            self.vgc_df = BASE_DF
        return self.vgc_df

    def plot_vgc_curve(self, vgc_df):
        vgc_df.set_index('N')[['expected_V', 'chunked_V']].plot()

        plt.xlabel('N')
        plt.ylabel('V(N), E[V(N)]')

    def plot_vgc_residuals(self, vgc_df):
        plt.axhline(y=0, linestyle='-', color='gray')
        vgc_df.set_index('N')['V_diff'].plot(style='.-')

        plt.xlabel('N')
        plt.ylabel('E[V(N)] - V(N)')

    def plot_vgc(self, vgc_df, figsize=(10, 5)):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        vgc_df.set_index('N')[['expected_V', 'chunked_V']].plot(ax=axes[0])
        vgc_df.set_index('N')['V_diff'].plot(style='.-', ax=axes[1])
        plt.axhline(y=0, linestyle='-', color='gray')
        fig.tight_layout(pad=3.0)

    def get_prog_err_scores_df(self):
        if not hasattr(self, 'prog_err_scores_df'):
            df = self.get_vgc()
            df['chunked_V_prev'] = df['chunked_V'].shift(1, fill_value=0)
            df['expected_V_prev'] = df['expected_V'].shift(1, fill_value=0)
            df['prog_err_score'] = (df['expected_V'] - df['expected_V_prev']) - (df['chunked_V'] - df['chunked_V_prev'])

            min_val = df['prog_err_score'].min()
            df['prog_err_score_log'] = df['prog_err_score'].apply(lambda x: np.log1p(x-min_val))

            self.min_prog_err_score = min_val
            self.prog_err_scores_df = df
        return [self.min_prog_err_score, self.prog_err_scores_df]

    def plot_prog_err_scores(self, min_val, prog_err_scores_df):
        fig = plt.figure(figsize=(10, 5))

        ax1 = plt.subplot(1, 2, 1)
        plt.axhline(y=np.log1p(0-min_val), color='gray')
        plt.scatter(prog_err_scores_df['N'], prog_err_scores_df['prog_err_score_log'], s=5, alpha=0.5)
        plt.setp(ax1.get_yticklabels())

        ax2 = plt.subplot(1, 2, 2, sharey=ax1)
        plt.axhline(y=np.log1p(0-min_val), color='gray')
        self._plot_w_ref_line(prog_err_scores_df, 'N', 'prog_err_score_log', ref=['ls', 'scatterplot_smoother'], hide_scatter=True)

        fig.tight_layout(pad=3.0)

        plt.xlabel('k: text slice')
        plt.ylabel('D(k):\nProgressive difference error scores')
        plt.suptitle('Error scores for the influx of new types')

    def calc_dispersion(self):
        cnt = {}
        for k, text_slice in tqdm(self.get_text_slices()):
            cnt[k] = Counter(''.join(text_slice))

        df = pd.DataFrame(cnt).fillna(0)
        df['d'] = (df != 0).sum(axis=1)
        df = df.rename_axis('char').reset_index()

        df['d_zscore'] = self.__get_zscore_lst(df['d'])
        df['is_underdispersed'] = df['d_zscore'].apply(lambda x: 1 if abs(x) >= 2.57 else 0)
        df['is_overdispersed'] = df['d_zscore'].apply(lambda x: 1 if x >= 2.57 else 0)
        df['d_f_threshold'] = df.apply(lambda row: self.char_freq.get(row['char'], 0) / row['d'], axis=1)

        self.disperse_df = df
        self.disperse = dict(zip(df['char'], df['d']))
        self.d_f_threshold = dict(zip(df['char'], df['d_f_threshold']))
        self.underdisperse_chars = df[df['is_underdispersed'] == 1]['char'].to_list()
        self.overdisperse_chars = df[df['is_overdispersed'] == 1]['char'].to_list()

    @property
    def get_disperse_df(self):
        if not hasattr(self, 'disperse_df'):
            self.calc_dispersion()
        return self.disperse_df

    @property
    def get_disperse(self):
        if not hasattr(self, 'disperse'):
            self.calc_dispersion()
        return self.disperse

    @property
    def get_underdisperse_chars(self):
        if not hasattr(self, 'underdisperse_chars'):
            self.calc_dispersion()
        return self.underdisperse_chars

    @property
    def get_overdisperse_chars(self):
        if not hasattr(self, 'overdisperse_chars'):
            self.calc_dispersion()
        return self.overdisperse_chars

    @property
    def get_d_f_threshold(self):
        if not hasattr(self, 'd_f_threshold'):
            self.calc_dispersion()
        return self.d_f_threshold

    def __get_zscore_lst(self, lst):
        zscore_lst = np.array(lst)
        zscore_lst = zscore_lst.astype('float')
        zscore_lst[zscore_lst == 0] = np.nan
        zscore_lst = stats.zscore(zscore_lst, nan_policy='omit')
        return zscore_lst

    def get_U(self):
        if not hasattr(self, 'U'):
            k_cols = [col for col in self.get_disperse_df.columns if isinstance(col, int)]
            df = self.get_disperse_df[k_cols]
            df.index = self.get_disperse_df['char']
            freq_mat = np.array(df)
            underdisperse_mat = np.array(self.get_disperse_df['is_underdispersed'])

            self.freq_df = df
            self.freq_mat = freq_mat

            threshold_mat = np.array(self.get_disperse_df['d_f_threshold']).reshape(-1, 1)
            indicator_mat = (threshold_mat >= freq_mat) * underdisperse_mat.reshape(-1, 1) #
            freq_indicator_mat = freq_mat * indicator_mat

            self.VU = indicator_mat.sum(axis=0)
            self.NU = freq_indicator_mat.sum(axis=0, dtype=int)

            first_indices = np.argmax(freq_mat > 0, axis=1)
            first_freq = [freq_mat[i,j] for i,j in zip(range(first_indices.shape[0]), first_indices)]

            grouper = defaultdict(list)
            for k, freq, underdisperse in zip(first_indices, first_freq, underdisperse_mat):
                grouper[k].append([freq, underdisperse])

            Pr_token = np.zeros(freq_mat.shape[1])
            Pr_type = Pr_token.copy()
            for k, data in grouper.items():
                mat = np.array(data)
                Pr_token[k] = sum(mat[:,0] * mat[:,1]) / sum(mat[:,0])
                Pr_type[k] = sum((mat[:,0] > 0) * mat[:,1]) / sum(mat[:,0] > 0)

            self.Pr_token = Pr_token
            self.Pr_type = Pr_type

        return {
            'freq_df': self.freq_df, 'freq_mat': self.freq_mat,
            'VU': self.VU, 'NU': self.NU,
            'Pr_token': self.Pr_token, 'Pr_type': self.Pr_type}

    @property
    def get_freq_df(self):
        if not hasattr(self, 'freq_df'):
            self.get_U()
        return self.freq_df

    @property
    def get_freq_mat(self):
        if not hasattr(self, 'freq_mat'):
            self.get_U()
        return self.freq_mat

    @property
    def get_VU(self):
        if not hasattr(self, 'VU'):
            self.get_U()
        return self.VU

    @property
    def get_NU(self):
        if not hasattr(self, 'NU'):
            self.get_U()
        return self.NU

    def _plot_U(self, U_var='VU'):
        xs = [k for k, _ in self.get_text_slices()]
        ys = self.get_U()[U_var]
        plt.plot(xs, ys)

        plt.xlabel('k: text slice')
        plt.ylabel(f'{U_var}(k):\nnumber of underdispersed tokens')

    def plot_VU(self):
        self._plot_U()

    def plot_NU(self):
        self._plot_U(U_var='NU')

    def _plot_U_acf(self, U_var='VU', diff=None):
        vals = self.get_U()[U_var]

        if diff is not None:
            for _ in range(diff):
                vals = (vals - np.roll(vals, 1))[1:]

        fig = plt.figure(figsize=(12,8))

        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(vals, ax=ax1)

        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(vals, ax=ax2)

        plt.suptitle(f'Series: {U_var}')

    def plot_VU_acf(self, diff=None):
        self._plot_U_acf(diff=diff)

    def plot_NU_acf(self, diff=None):
        self._plot_U_acf(U_var='NU', diff=diff)

    @property
    def get_Pr_type(self):
        if not hasattr(self, 'Pr_type'):
            self._get_Pr()
        return self.Pr_type

    @property
    def get_Pr_token(self):
        if not hasattr(self, 'Pr_token'):
            self._get_Pr()
        return self.Pr_token

    def _plot_Pr(self, var='type'):
        plt.plot([k for k in range(self.get_U()[f'Pr_{var}'].shape[0])], self.get_U()[f'Pr_{var}'])

        plt.xlabel('k: text slice')
        plt.ylabel(f'Pr(U,{var}):\nproportion of new underdispersed {var}s')

    def plot_Pr_type(self):
        self._plot_Pr()

    def plot_Pr_token(self):
        self._plot_Pr(var='token')