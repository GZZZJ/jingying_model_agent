import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
try:
    import toad
    _HAS_TOAD = True
except ImportError:
    _HAS_TOAD = False
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid



def quantile_bin(x, bin_num=10):
    quantile = [100 / bin_num * i for i in range(1, bin_num)]
    x_no_na = pd.Series(x).dropna()
    cut = np.percentile(x_no_na.values, quantile).tolist()
    cut.insert(0, -np.inf)
    cut.insert(bin_num, np.inf)
    cut = sorted(list(pd.Series(cut).drop_duplicates().dropna()))

    return cut


def quantile_bin_wgt(x, wgt, bin_num=10):
    tot_wgt = wgt.sum()
    tmp_df = pd.DataFrame({'x': x, 'wgt': wgt}).sort_values(by=['x', 'wgt'])
    tmp_df['cumsum_wgt'] = tmp_df['wgt'].cumsum()
    tmp_df['cumsum_wgt_pct'] = tmp_df['cumsum_wgt'] / tot_wgt
    wgt_distance_cut = distance_bin(x=tmp_df['cumsum_wgt_pct'], bin_num=bin_num)
    tmp_df['wgt_grp'] = pd.cut(tmp_df['cumsum_wgt_pct'], wgt_distance_cut, labels=list(range(1, bin_num + 1)))
    # cut = list(tmp_df.groupby('wgt_grp').x.max().values)
    cut = list(tmp_df.groupby('wgt_grp').x.agg(['max'])['max'].values)
    cut.insert(0, -np.inf)
    cut[-1] = np.inf

    return sorted(set(cut))

def distance_bin(x, bin_num=10):
    max_prob = x.max()
    min_prob = x.min()
    bin_diff = (max_prob - min_prob) / bin_num

    cut = [min_prob + bin_diff * i for i in range((bin_num + 1))]
    cut[0] = -np.inf
    cut[-1] = np.inf

    return sorted(set(cut))


def cat_bin(x, bin_num=None):
    # val_cnts = x.value_counts(normalize=True,dropna=False).to_frame().reset_index().rename(columns={'index': 'vals'})
    val_cnts = x.value_counts(normalize=True,dropna=False).to_frame().reset_index()
    val_cnts.columns = ['index', 'proportion']
    val_cnts = val_cnts.rename(columns={'index': 'vals'})
    
    val_cnts['vals'] = val_cnts['vals'].fillna('missing')

    if not bin_num:

        return {i: i for i in val_cnts.vals}
    elif len(val_cnts) > bin_num:
        less_vals = tuple(val_cnts.iloc[bin_num - len(val_cnts) - 1:, :].vals)

        return {
            i: i if i not in less_vals else str(less_vals)
            for i in val_cnts.vals
        }
    else:
        return {i: i for i in val_cnts.vals}


def format_num_bin(cut_list):
    # idx_plus = lambda x, n: str(x / 10 ** (n - 1)).replace('.', '') if x != 0 else '000'
    idx_plus = lambda x, n: '0' * (n - len(str(x))) + str(x)
    bin_str = map(lambda x: '{}:({:.7f}, {:.7f}]'.format(idx_plus(x[0] + 1, 3), x[1][0], x[1][1]) if x[1][1] != np.inf else '{}:({:.7f}, {:.7f}]'.format(idx_plus(x[0] + 1, 3), x[1][0], x[1][1]), enumerate(zip(cut_list[:-1], cut_list[1:])))

    return list(bin_str)


def bin_apply(x, cut, precision=7, x_type='NUM', na_bin='000:NULL', na_label=-999, else_bin='999:ELSE', else_label=999):
    # apply cut on x, return bins and labels series
    if x_type == 'NUM':
        bin_str_list = format_num_bin(cut)
        bins = pd.cut(x=x, bins=cut, precision=precision, labels=bin_str_list)
        labels = pd.cut(x=x, bins=cut, precision=precision, labels=range(len(cut) - 1))

        # pd.cut对缺失值返回的是null
        bins = pd.Series(bins).astype(object).fillna(na_bin)
        labels = pd.Series(labels).astype(object) + 1
        labels = labels.fillna(na_label)

        return bins, labels
    elif x_type == 'CHAR':
        bins = x.fillna('missing').apply(lambda x: cut.get(x) if cut.get(x) else else_bin)
        bins = bins.replace('missing', na_bin)
        # 按照cut的顺序给定序号
        # bin_label_map = list(map(lambda x: (x[0] + 1, x[1]), enumerate(map(lambda x: x[1], sorted(set(enumerate(cut.values())))))))
        cut_set = list()
        for i in cut.values():
            if i not in cut_set:
                cut_set.append(i)

        bin_label_map = list(map(lambda x: (x[0] + 1, x[1]), enumerate(cut_set)))
        bin_label_map = [(na_label, na_bin)] + bin_label_map + [(else_label, else_bin)]
        bin_label_map = {v: k for k, v in bin_label_map}

        labels = bins.apply(lambda x: bin_label_map.get(x))

        return bins, labels
    else:
        raise ValueError("unsopprted x_type: {}".format(x_type))


def toad_bin(df, varname, target_var, method='chi', min_samples=None, n_bins=None, empty_separate=False):
    if is_numeric_dtype(df[varname]):
        print("fit NUM {} ...".format(varname))
    else:
        print("fit CHAR {} ...".format(varname))

    comb = toad.transform.Combiner()

    if method == 'chi':
        comb.fit(df.loc[:, [varname, target_var]], y=target_var, method=method, min_samples=min_samples, empty_separate=empty_separate)
    elif method == 'quantile':
        comb.fit(df.loc[:, [varname, target_var]], y=target_var, method=method, n_bins=n_bins, empty_separate=empty_separate)

    cut_info = comb.export()[varname]
    cut_info = [i for i in cut_info if i == i]

    return cut_info


def call_iv(event_pct, no_event_pct, woe_cap=None):
    eps = 1.0e-38
    woe = np.log((event_pct + eps) / (no_event_pct + eps))
    # 修正woe极端值, [-woe_cap, woe_cap]
    if woe_cap is not None:
        woe = np.clip(woe, -woe_cap, woe_cap)

    iv = (event_pct - no_event_pct) * woe

    return woe, iv


def call_metric(df, varname, tgt_col, tot_col=None, weight_col=None, var_type='NUM', inverse=False, use_bin=True, woe_cap=None):
    varname_bin = varname + '_bin'

    if not use_bin:
        df[varname_bin] = df[varname].copy()

    df['n'] = 1 if weight_col is None else df[weight_col].copy()
    tgt_col = 'n' if tgt_col is None else tgt_col
    tot_col = 'n' if tot_col is None else tot_col

    all_tgt_sum = df[tgt_col].sum()
    all_tot_sum = df[tot_col].sum()
    all_sub_sum = all_tot_sum - all_tgt_sum
    all_tgt_rate = all_tgt_sum / all_tot_sum
    all_sub_rate = all_sub_sum / all_tot_sum

    # var stt
    if var_type == 'NUM':
        stt_df = df.groupby(varname_bin)[varname].agg(['min', 'max', 'mean', 'sum']).reset_index()
        if use_bin:
            tmp_df = stt_df[varname_bin].str.extract(r'\((?P<left>.+?), (?P<right>.+?)\]', expand=True).astype(float)
        else:
            tmp_df = stt_df[[varname_bin]].copy()
            tmp_df['left'] = tmp_df[varname_bin].copy()
            tmp_df['right'] = tmp_df[varname_bin].copy()
            tmp_df.drop(columns=[varname_bin], inplace=True)

        stt_df = pd.concat([stt_df, tmp_df], axis=1)
        stt_df['mid'] = (stt_df.left.replace(-np.inf, stt_df['min'].min()) + stt_df.right.replace(np.inf, stt_df['max'].max())) / 2
        stt_df = stt_df.loc[:, [varname_bin, 'left', 'right', 'min', 'max', 'mean', 'mid', 'sum']].rename(columns={varname_bin: varname})
    else:
        stt_df = df.groupby(varname_bin)[varname_bin].agg(['count']).reset_index().rename(columns={varname_bin: varname})
        stt_df['left'] = '-'
        stt_df['right'] = '-'
        stt_df['min'] = '-'
        stt_df['max'] = '-'
        stt_df['mean'] = '-'
        stt_df['mid'] = '-'
        stt_df['sum'] = '-'
        stt_df.drop(columns='count', inplace=True)

    # bin metric
    # metric_df = df.groupby(varname_bin).sum()[[tgt_col, tot_col]].reset_index().rename(columns={varname_bin: varname})
    metric_df = df.groupby(varname_bin).agg({tgt_col: 'sum', tot_col: 'sum'}).reset_index().rename(columns={varname_bin: varname})
    metric_df = metric_df.rename(columns={tgt_col: 'tgt_sum', tot_col: 'tot_sum'})
    metric_df = pd.merge(left=stt_df, right=metric_df, on=varname, how='left')

    if inverse:
        metric_df = metric_df.iloc[::-1, :]

    metric_df['tgt_rate'] = metric_df['tgt_sum'] / metric_df['tot_sum']
    metric_df['tgt_lift'] = metric_df['tgt_rate'] / all_tgt_rate
    metric_df['tgt_pct'] = metric_df['tgt_sum'] / all_tgt_sum
    metric_df['tot_pct'] = metric_df['tot_sum'] / all_tot_sum
    metric_df['sub_sum'] = metric_df['tot_sum'] - metric_df['tgt_sum']
    metric_df['sub_rate'] = 1 - metric_df['tgt_rate']
    metric_df['sub_pct'] = metric_df['sub_sum'] / all_sub_sum
    metric_df['sub_lift'] = metric_df['sub_pct'] / all_tgt_rate

    metric_df['cumsum_tgt_sum'] = metric_df['tgt_sum'].cumsum()
    metric_df['cumsum_tot_sum'] = metric_df['tot_sum'].cumsum()
    metric_df['cumsum_tgt_rate(Precision)'] = metric_df['cumsum_tgt_sum'] / metric_df['cumsum_tot_sum']
    metric_df['cumsum_tgt_lift'] = metric_df['cumsum_tgt_rate(Precision)'] / all_tgt_rate
    metric_df['cumsum_tgt_pct(TPR)'] = metric_df['cumsum_tgt_sum'] / all_tgt_sum
    metric_df['cumsum_tot_pct'] = metric_df['cumsum_tot_sum'] / all_tot_sum

    metric_df['cumsum_sub_sum'] = metric_df['sub_sum'].cumsum()
    metric_df['cumsum_sub_rate'] = metric_df['cumsum_sub_sum'] / metric_df['cumsum_tot_sum']
    metric_df['cumsum_sub_lift'] = metric_df['cumsum_sub_rate'] / all_sub_rate
    metric_df['cumsum_sub_pct(FPR)'] = metric_df['cumsum_sub_sum'] / all_sub_sum

    metric_df['rest_tgt_sum'] = all_tgt_sum - metric_df['cumsum_tgt_sum']
    metric_df['rest_tot_sum'] = all_tot_sum - metric_df['cumsum_tot_sum']
    metric_df['rest_tgt_rate'] = metric_df['rest_tgt_sum'] / metric_df['rest_tot_sum']
    metric_df['rest_tgt_lift'] = metric_df['rest_tgt_rate'] / all_tgt_rate
    metric_df['rest_tgt_pct(FNR)'] = metric_df['rest_tgt_sum'] / all_tgt_sum
    metric_df['rest_tot_pct'] = metric_df['rest_tot_sum'] / all_tot_sum

    metric_df['rest_sub_sum'] = all_sub_sum - metric_df['cumsum_sub_sum']
    metric_df['rest_sub_rate'] = metric_df['rest_sub_sum'] / metric_df['rest_tot_sum']
    metric_df['rest_sub_lift'] = metric_df['rest_sub_rate'] / all_tgt_rate
    metric_df['rest_sub_pct(TNR)'] = metric_df['rest_sub_sum'] / all_sub_sum

    metric_df['accuracy'] =  (metric_df['cumsum_tgt_sum'] + metric_df['rest_sub_sum']) / all_tot_sum

    metric_df['woe'], metric_df['iv'] = call_iv(event_pct=metric_df['tgt_pct'], no_event_pct=metric_df['sub_pct'], woe_cap=woe_cap)
    metric_df['sum_iv'] = metric_df['iv'].sum()

    metric_df['ks_bin'] = metric_df['cumsum_tgt_pct(TPR)'] - metric_df['cumsum_sub_pct(FPR)']
    metric_df['ks_max'] = metric_df['ks_bin'].max()
    metric_df['auc_bin'] = metric_df['cumsum_tgt_pct(TPR)'] * metric_df['tot_sum'] / all_tot_sum
    # metric_df['auc_sum'] = (metric_df['cumsum_tgt_pct(TPR)'] * metric_df['tot_sum']  / all_tot_sum).sum()
    metric_df['auc_sum'] = trapezoid(metric_df['cumsum_tgt_pct(TPR)'], metric_df['cumsum_sub_pct(FPR)'])

    metric_df['all_tgt_sum'] = all_tgt_sum
    metric_df['all_tot_sum'] = all_tot_sum
    metric_df['all_tgt_rate'] = all_tgt_rate
    metric_df['all_sub_rate'] = all_sub_rate

    # cnt_df = df.groupby(varname_bin).sum()['n'].reset_index().rename(columns={varname_bin: varname})
    cnt_df = df.groupby(varname_bin).agg({'n': 'sum'}).reset_index().rename(columns={varname_bin: varname})
    cnt_df['cnt_pct'] = cnt_df['n'] / df['n'].sum()
    metric_df = pd.merge(left=metric_df, right=cnt_df, on=varname, how='left')

    # 新增拒绝/通过样本均值列
    if var_type == 'NUM':
        metric_df['upper_mean'] = metric_df['sum'].cumsum() / metric_df['n'].cumsum()
        metric_df['lower_mean'] = (metric_df['sum'].sum() - metric_df['sum'].cumsum()) / (metric_df['n'].sum() - metric_df['n'].cumsum())
        metric_df['lower_mean'].fillna(0, inplace=True)
    else:
        metric_df['upper_mean'] = '-'
        metric_df['lower_mean'] = '-'

    metric_df.drop(columns=['n'], inplace=True)

    return metric_df


def distribution_plot(x_label, bar_values_1, bar_values_2=None, line_values=None, hline_value=None,
                    barlabel1="groupA", barlabel2="groupB", linelabel="比例",
                    xlabel="分组", ylabel="人数", title="分布图"):
    '''
    绘制分布图，包含堆叠的柱状图以及折线图，常见的如分组逾期图、按某个变量的分组的另一个变量的默认值分布图等
    -----

    params
    -----
    x_label: list, x轴标签，即分组;
    bar_values_1: list, 堆叠柱状图中全体类别的分布,如全体样本数量;
    bar_values_2: list, 堆叠柱状图中某一个类别的分布,如好样本数量;
    line_values: list, 折线图的数据,以比例为主,如逾期率等;
    hline_value: float, 水平线的值, 比如均值

    return
    -----
    fig: matplotlib的figure对象
    '''
    x_list = np.arange(len(x_label))

    fig, ax1 = plt.subplots()
    fig.set_size_inches((12, 6))
    ax1.bar(x_list, bar_values_1, width=0.6, align='center', color=(24/254, 192/254, 196/254), label=barlabel1)

    # 是否存在第二个堆叠柱状图系列,
    if bar_values_2 is not None:
        ax1.bar(x_list, bar_values_2, width=0.6, align='center', color=(246/254, 115/254, 109/254), label=barlabel2)

    ax1.set_xticks(x_list)
    ax1.set_xticklabels(x_label, rotation=60)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_ylim(0, max(bar_values_1) * 1.2)

    ax2 = ax1.twinx()
    ax2.plot(x_list, line_values, 'o-', label=linelabel)

    # 水平线
    if hline_value is not None:
        ax2.hlines(hline_value, min(x_list), max(x_list), colors='g', linestyles='dashed')

    ax2.set_ylim(min([min(line_values) * 1.2, 0]), max(line_values) * 1.2)
    ax2.set_ylabel("Metric")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(handles1+handles2, labels1+labels2, loc='upper right')
    plt.title(title)

    return fig


class Binner:
    def __init__(self):
        self.cut_info = dict()
        self.merge_round = 0
        self.var_type = None
        self.var_type_info = dict()

    def fit_toad(self, df, varname, target_var, method='chi', min_samples=None, n_bins=None):
        cut_rlt = toad_bin(df, varname, target_var, method=method, min_samples=min_samples, n_bins=n_bins, empty_separate=True)

        # transform numeric and category cut info
        if all([not isinstance(i, list) for i in cut_rlt]):
            cut_rlt = [-np.inf] + cut_rlt + [np.inf]
            self.var_type = 'NUM'
        else:
            cut_rlt = {j: i for i in cut_rlt for j in i}
            self.var_type = 'CHAR'

        return cut_rlt

    def fit_raw(self, df, varname, method='quantile', num_nbins=10, char_nbins=None, weight_col=None):
        # cut numeric or category x
        cut_method_map = {
            'quantile': quantile_bin,
            'distance': distance_bin,
            'quantile_wgt': quantile_bin_wgt,
        }

        if method not in cut_method_map.keys():
            raise Exception("method: {} not supported yet!".format(method))

        if is_numeric_dtype(df[varname]):
            print("fit NUM {} ...".format(varname))

            if method == 'quantile' and weight_col:
                method = 'quantile_wgt'
                cut_rlt = cut_method_map[method](x=df[varname], wgt=df[weight_col], bin_num=num_nbins)
            else:
                cut_rlt = cut_method_map[method](x=df[varname], bin_num=num_nbins)

            self.var_type = 'NUM'
        else:
            print("fit CHAR {} ...".format(varname))
            cut_rlt = cat_bin(x=df[varname], bin_num=char_nbins)
            self.var_type = 'CHAR'

        return cut_rlt

    def fit(self, df, exclude, method='quantile', num_nbins=10, char_nbins=None, toad_nbins=None, target_var=None, weight_col=None, min_samples=None, toad=False):
        cut_info = dict()
        varlist = [i for i in df.columns if i not in exclude and i != target_var]
        for varname in varlist:
            if toad:
                cut_rlt = self.fit_toad(df, varname, target_var, method, min_samples, toad_nbins)
            else:
                cut_rlt = self.fit_raw(df, varname, method, num_nbins, char_nbins, weight_col)

            cut_info[varname] = cut_rlt
            self.var_type_info[varname] = self.var_type

        self.cut_info = cut_info

    def adjust(self, varname, cut, verbose=True):
        '''
        adjust specific var bin result with cut list
        '''
        if self.cut_info is None:
            self.cut_info = dict()

        if verbose:
            print("adjust var: {}, cut_info: {}".format(varname, cut))

        self.cut_info[varname] = cut

    def get_cut_info(self, varname):
        '''
        get cut list of sepecific var
        '''
        if varname not in self.cut_info.keys():
            raise Exception("key: {} not found in cut_info".format(varname))

        return self.cut_info[varname]

    def transform(self, varname, x, precision=7):
        '''
        apply bin result on specific var

        :param varname: string, varname with x to transform
        :param x: Series, data of varname
        :param precision: int, num of precision for pd.cut
        '''
        # return bins and labels
        cut_info = self.cut_info.get(varname)
        var_type = self.var_type_info.get(varname)
        # var_type = 'NUM' if isinstance(cut_info, list) else 'CHAR'
        x_bin, x_label = bin_apply(x, cut_info, precision, var_type)

        return x_bin, x_label

    def export(self):
        return self.cut_info

    @property
    def label_bin_map(self):
        label_bin_map = dict()
        for var, var_cut in self.cut_info.items():
            if isinstance(var_cut, list):
                label_bin_map_tmp = dict(zip(range(1, len(var_cut) + 1), format_num_bin(var_cut)))
            else:
                cut_set = list()
                for i in var_cut.values():
                    if i not in cut_set:
                        cut_set.append(i)
                label_bin_map_tmp = {idx + 1: itm for idx, itm in enumerate(cut_set)}

            label_bin_map[var] = label_bin_map_tmp

        label_bin_map = {k: {-999: '000:NULL', **v} for k, v in label_bin_map.items()} # 添加缺失组

        return label_bin_map

    def stt(self, df, varname, tgt_col, tot_col=None, weight_col=None, inverse=False, use_bin=True, woe_cap=None):
        df_copy = df.copy()
        df_copy[varname + '_bin'] = self.transform(varname, x=df_copy[varname], precision=7)[0]
        stt_df = call_metric(df_copy, varname, tgt_col, tot_col, weight_col, self.var_type_info[varname], inverse, use_bin, woe_cap)

        return stt_df

    def plot_bin(self, df, varname, tgt_nam, tgt_col, tot_col=None, weight_col=None, tgt_type='tgt_rate', inverse=False, use_bin=True, woe_cap=None):
        stt_df = self.stt(df, varname, tgt_col, tot_col, weight_col, inverse, use_bin, woe_cap)
        fig = distribution_plot(x_label=list(stt_df[varname]),
                          bar_values_1=list(stt_df['cnt_pct']),
                          bar_values_2=None,
                          line_values=list(stt_df[tgt_type]),
                          hline_value=list(stt_df['all_tgt_rate']),
                          barlabel1="BIN % Total",
                          barlabel2="groupB",
                          linelabel=tgt_type,
                          xlabel="BIN",
                          ylabel="BIN % Total",
                          title=f"Stats {tgt_nam} {tgt_type} by {varname} bins")

        return fig

    def plot_roc(self, df, varname_list, tgt_col_list, tot_col_list=None, weight_col=None, inverse=False, use_bin=True, woe_cap=None):
        if use_bin:
            print("WARNING! This ROC curve is ploted using bin result, may not correct with few bins!")

        fig = plt.figure(figsize=(10, 10))
        for varname in varname_list:
            for tgt_col, tot_col in zip(tgt_col_list, tot_col_list):
                stt_df = self.stt(df, varname, tgt_col, tot_col, weight_col, inverse, use_bin, woe_cap)
                tpr = list(stt_df['cumsum_tgt_pct(TPR)'])
                fpr = list(stt_df['cumsum_sub_pct(FPR)'])
                ks = stt_df['ks_max'].iloc[0]
                auc = stt_df['auc_sum'].iloc[0]
                plt.plot([0.0] + fpr, [0.0] + tpr, label=f'var={varname}, agg={tgt_col}, KS={ks:.3f}, AUC={auc:.3f}')

        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xticks([0.1 * i for i in range(0, 11)], labels=[f'{0.1 * i:.0%}' for i in range(0, 11)])
        plt.xlabel('FPR(group%)')
        plt.xlim(-0.02, 1.02)

        plt.yticks([0.1 * i for i in range(0, 11)], labels=[f'{0.1 * i:.0%}' for i in range(0, 11)])
        plt.ylabel('TPR')
        plt.ylim(-0.02, 1.02)
        plt.title(label='ROC curve')
        plt.legend(loc='lower right')
        plt.grid(which='major', axis='both', linestyle='-.')
        plt.show()

        return fig

    def merge(self, df, varname, tgt_nam, tgt_col, tot_col=None, tgt_type='woe', weight_col=None, min_nbins=2, stepwise=True, verbose=False, plot=False, woe_cap=None, min_cnt_pct=0.01):
        print("start merge ...")
        if plot:
            self.plot_bin(df, varname, tgt_nam, tgt_col, tot_col, weight_col, tgt_type, woe_cap=woe_cap)

        stt_df = self.stt(df, varname, tgt_col, tot_col, weight_col=weight_col, woe_cap=woe_cap)
        cut = self.cut_info.get(varname)
        cut_metric_zip = list(zip(cut[1:], list(stt_df[stt_df[varname] != '000:NULL'][tgt_type])))

        # check direction
        direction = cut_metric_zip[-1][1] - cut_metric_zip[0][1] > 0 # True if last > first else False

        # find False cut
        dir_check = map(lambda x: (x[0][0], x[1][0], x[1][1] - x[0][1] > 0), zip(cut_metric_zip[:-1], cut_metric_zip[1:]))
        merge_candidate = list(filter(lambda x: x[2] != direction, dir_check))
        merge_candidate = sorted(set([j for i in merge_candidate for j in i[:-1] if j != np.inf]))

        if len(cut) - 1 <= min_nbins:
            print("Hit min_nbins limit, exit! merge round: {}".format(self.merge_round))
            self.merge_round = 0 # reset merge_round
            return None
        elif not merge_candidate:
            print("No candidate, exit! merge round: {}".format(self.merge_round))
            self.merge_round = 0 # reset merge_round
            return None
        else:
            # all merge result
            self.merge_round += 1
            print("=" * 20, "round {}".format(self.merge_round), "=" * 20)
            print("try drop all candidate cut ...")

            if stepwise:
                print("use stepwise strategy ...")
                merge_result = dict()
                for cand in merge_candidate:
                    self.adjust(varname, cut=[i for i in cut if i != cand], verbose=verbose)
                    stt_df = self.stt(df, varname, tgt_col, tot_col, weight_col=weight_col, woe_cap=woe_cap)
                    merge_iv = stt_df.sum_iv.iloc[0]
                    merge_result[cand] = merge_iv

                print("candidate drop info:", list(merge_result.items()))
                final_merge_cut = sorted(merge_result.items(), key=lambda x: x[1], reverse=True)[0] # 选择删除后IV最大的结果
            else:
                print("use min-to-max strategy ...")
                cand = merge_candidate[0]
                self.adjust(varname, cut=[i for i in cut if i != cand], verbose=verbose)
                stt_df = self.stt(df, varname, tgt_col, tot_col, weight_col=weight_col, woe_cap=woe_cap)
                merge_iv = stt_df.sum_iv.iloc[0]
                final_merge_cut = (cand, merge_iv)

            print("final drop cut: {}, iv after drop: {}".format(*final_merge_cut))
            final_cut = [i for i in cut if i != final_merge_cut[0]]
            self.adjust(varname, final_cut)
            print("\n")

        # recursive merge
        self.merge(df, varname, tgt_nam, tgt_col, tot_col, tgt_type, weight_col, min_nbins, stepwise, verbose, plot, woe_cap, min_cnt_pct)

        # 判断是否存在占比过小的分组
        print(f"check cnt_pct < {min_cnt_pct} condition ")
        stt_df = self.stt(df, varname, tgt_col, tot_col, weight_col=weight_col, woe_cap=woe_cap)
        min_cnt_grp_right = list(stt_df[stt_df.cnt_pct < min_cnt_pct].right)
        if np.inf in min_cnt_grp_right:
            min_cnt_grp_right = [i for i in min_cnt_grp_right if i != np.inf] + [stt_df.left.max()]

        if min_cnt_grp_right:
            cut = self.cut_info.get(varname)
            self.adjust(varname, cut=[i for i in cut if i not in min_cnt_grp_right], verbose=verbose)

        if plot:
            self.plot_bin(df, varname, tgt_nam, tgt_col, tot_col, weight_col, tgt_type, woe_cap=woe_cap)

    def report(self, df, exclude, target_all, output_dir, prefix):
        pass


def distribution_plot_common(x_label, bar_values, line_values, bar_labels, line_labels, hline_value=None, xlabel="Bin", ax1_ylabel="% Total", ax2_ylabel="Bad Rate", title="Stats", show=True):
    '''
    绘制分布图，包含堆叠的柱状图以及折线图，常见的如分组逾期图、按某个变量的分组的另一个变量的默认值分布图等

    :params x_label: list, x轴标签, 即分组;
    :params bar_values: iterable, 柱状图系列数据;
    :params line_values: iterable, 折线图系列数据;
    :params bar_labels: iterable, 柱状图系列标签;
    :params line_labels: iterable, 折线图系列标签;
    :params hline_value: float, 水平基准线的值;
    '''
    raw_x_list = np.arange(len(x_label))

    fig, ax1 = plt.subplots()
    fig.set_size_inches((12, 6))
    
    num_bars = len(bar_values)
    width = 1 / (num_bars + 2)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    if num_bars % 2 == 0:
        align='edge'
    else:
        align='center'
    
    for idx, (bar_label, bar_value) in enumerate(zip(bar_labels, bar_values)):
        x_list = [i + (idx - num_bars // 2) * width for i in raw_x_list]
        ax1.bar(x_list, bar_value, width=width, align=align, label=bar_label, edgecolor='black', color=colors[idx])

    ax1.set_xticks(x_list)
    ax1.set_xticklabels(x_label, rotation=60)
    ax1.set_xlabel(xlabel)
    ax1.set_yticks([i * 0.1 for i in range(11)])
    ax1.set_yticklabels([f'{i * 10}.0%' for i in range(11)])
    ax1.set_ylabel(ax1_ylabel)
#     ax1.set_ylim(0, max([max(i) for i in bar_values]) * 1.2)
    ax1.set_ylim(0, 1.1)

    ax2 = ax1.twinx()
    
    for idx, (line_label, line_value) in enumerate(zip(line_labels, line_values)):
        ax2.plot(x_list, line_value, 'o-', label=line_label, color=colors[idx])

    # 水平线
    if hline_value is not None:
        ax2.hlines(hline_value, min(x_list), max(x_list), colors='grey', linestyles='dashed')

    ax2.set_ylim(min([min([min(i) for i in line_values]) * 1.2, 0]), max([max(i) for i in line_values]) * 1.2)
    ax2.set_ylabel(ax2_ylabel)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    
#     plt.legend(handles1+handles2, labels1+labels2, loc='upper right')
    plt.legend(handles1+handles2, labels1+labels2, loc=2,  bbox_to_anchor=(1.05,1.0),borderaxespad=0.5)
    plt.title(title, pad=15)

    if not show:
        plt.close()

    return fig


def split_plot_feature(df, feature_list, tw_col, dev_tw, tgt_col, tot_col=None, tw_set=None, method='quantile', num_nbins=5, merge=False, stepwise=False, woe_cap=None, show=False):
    '''
    拆分数据集在同一张图中绘制woe, 返回分箱woe数据以及绘图对象fig, 可以使用fig.savefig(png或者svg文件路径, bbox_inches='tight')来保存图片

    :param df: pd.DataFrame, 样本数据
    :param feature_list: list, 要绘制的变量列表
    :param tw_col: string, 样本窗口字段
    :param dev_tw: string, DEV的样本窗口
    :param tgt_col: string, 统计指标的分子字段
    :param tot_col: string, 统计指标的分母字段
    :param tw_set: list, 指定的样本窗口取值, 会按照这个顺序来画图, 是tw_col所有取值的子集
    :param method: string, 分箱的方法
    :param num_nbins: int, 初始分箱数, 默认5, 太多会导致图片过于复杂
    :param merge: bool, 是否合并分箱, 默认False, 合并会增加耗时
    :param stepwise: bool, 是否使用stepwise合并分箱
    :param woe_cap: float, woe上限
    '''
    eps = 1.0e-38
    weighted_psi = lambda x, y, wgt: (x - y) * np.log((x + eps) / (y + eps)) * wgt

    dev_df = df[df[tw_col] == dev_tw]
    exclude = [i for i in df.columns if i not in feature_list]
    
    if tw_set is None:
        tw_set = sorted(list(df[tw_col].unique()))

    bin_obj = Binner()
    bin_obj.fit(df=dev_df, exclude=exclude, method=method, num_nbins=num_nbins)

    if merge:
        for fea in feature_list:
            bin_obj.merge(df=dev_df, varname=fea, tgt_nam='badrate', tgt_col=tgt_col, tot_col=tot_col, 
                        tgt_type='woe', weight_col=None, min_nbins=2, stepwise=stepwise, 
                        verbose=False, plot=False, woe_cap=woe_cap)
            
    all_fea_stt = dict()
    all_fea_fig = dict()
    for fea in feature_list:
        tw_tmp_stt = list()
        for tw in tw_set:
            tw_df = df[df[tw_col] == tw]
            tw_stt = bin_obj.stt(df=tw_df, varname=fea, tgt_col=tgt_col, tot_col=tot_col, weight_col=None, inverse=False, use_bin=True, woe_cap=woe_cap)
            tw_stt = tw_stt.rename(columns={fea: 'bin'})
            old_cols = list(tw_stt.columns)
            tw_stt['varname'] = fea
            tw_stt['time_window'] = tw
            tw_stt = tw_stt.loc[:, ['varname', 'time_window'] + old_cols]
            tw_tmp_stt.append(tw_stt)

            if tw == dev_tw:
                base_stt = tw_stt.loc[:, ['bin', 'tot_pct', 'tgt_pct', 'tgt_rate']].copy()
                base_stt.columns = ['bin', 'base_tot_pct', 'base_tgt_pct', 'base_tgt_rate']

        for tw_stt in tw_tmp_stt:
            tw_stt_tmp = tw_stt.loc[:, ['bin', 'tot_pct', 'tgt_pct', 'tgt_rate']].copy()
            tw_stt_tmp.columns = ['bin', 'exp_tot_pct', 'exp_tgt_pct', 'exp_tgt_rate']
            merge_stt = pd.merge(left=base_stt, right=tw_stt_tmp, on='bin', how='outer')

            normal_wgt = np.ones(len(merge_stt))
            sample_pct_wgt = merge_stt.base_tot_pct

            tw_stt['psi'] = weighted_psi(merge_stt.base_tot_pct, merge_stt.exp_tot_pct, normal_wgt)
            tw_stt['psi_badpct'] = weighted_psi(merge_stt.base_tgt_pct, merge_stt.exp_tgt_pct, normal_wgt)
            tw_stt['psi_badrate'] = weighted_psi(merge_stt.base_tgt_rate, merge_stt.exp_tgt_rate, normal_wgt)
            tw_stt['psi_badrate_wgt'] = weighted_psi(merge_stt.base_tgt_rate, merge_stt.exp_tgt_rate, sample_pct_wgt) # 样本占比少的分组权重降低
            
        all_bins = sorted(set([j for i in tw_tmp_stt for j in i['bin']])) # 全部分组
        bin_dataset = {i: 0 for i in all_bins}
        bar_data = [{**bin_dataset, **dict(zip(i['bin'], i['tot_pct']))} for i in tw_tmp_stt]        
        line_data = [{**bin_dataset, **dict(zip(i['bin'], i['tgt_rate']))} for i in tw_tmp_stt]
        
        x_label = all_bins
        bar_values = [list(i.values()) for i in bar_data]
        line_values = [list(i.values()) for i in line_data]
        bar_labels = tw_set
        line_labels = tw_set
        hline_value = tw_tmp_stt[0]['all_tgt_rate'][0]

        fig = distribution_plot_common(x_label, bar_values, line_values, bar_labels, line_labels, hline_value, xlabel="Bin", ax1_ylabel="% Total", ax2_ylabel="Bad Rate", title=f"Stats by {fea}", show=show)

        all_fea_stt[fea] = tw_tmp_stt
        all_fea_fig[fea] = fig

    return all_fea_stt, all_fea_fig


def split_plot_to_excel(output_dir, prefix, all_fea_fig, feature_comment_map):
    import os
    import xlsxwriter
    from io import BytesIO

    filename = os.path.join(output_dir, f'{prefix}.xlsx')

    wb = xlsxwriter.Workbook(filename)
    ws = wb.add_worksheet(name='summary')

    text_params = {
        'width': 64 * 7, # 单元格宽64 * 7
        'height': 20 * 2, # 单元格高20 * 2
        'x_scale': 1,
        'y_scale': 1,
        'x_offset': 64 * 2, # 水平右移1个单元格宽度
        'y_offset': 0,
        'font': {'bold': True},
        'align': {'vertical': 'middle','horizontal': 'center'},
    }

    image_params = {
        'x_offset': 0,
        'y_offset': 20 * 3, # 纵向下移3个单元格高度
        'x_scale': 0.6, # 缩小为0.6倍
        'y_scale': 0.6, # 缩小为0.6倍
        'object_position': 2, # 不随单元格调整改变大小
        'image_data': None,
        'url': None,
        'description': None,
        'decorative': False,
    }

    start_row = 1
    start_col = 1

    for idx, (fea, fig) in enumerate(all_fea_fig.items()):
        comment = feature_comment_map.get(fea, '')
        buf = BytesIO()
        fig.savefig(buf, format='PNG', bbox_inches='tight') # insert_image不能插入svg
        image_params.update({'image_data': buf})

        # 文本框 - 中文含义
        ws.insert_textbox(row=start_row, col=start_col, text=comment, options=text_params)
        ws.insert_image(row=start_row, col=start_col, filename='', options=image_params)

        # 每行3个图
        if (idx + 1) % 3 == 0:
            start_row += 30
            start_col = 1
        else:
            start_col += 12

    # 背景设置成灰色
    cell_format = {'bg_color': '#7F7F7F'}
    cell_format = wb.add_format(cell_format)
    for i in range(0, 1000):
        ws.set_row(i, None, cell_format)
    
    # 隐藏网格线
    ws.hide_gridlines()
    wb.close()