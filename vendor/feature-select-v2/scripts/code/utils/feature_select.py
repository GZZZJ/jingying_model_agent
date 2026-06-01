import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import spearmanr
try:
    import toad
    _HAS_TOAD = True
except ImportError:
    _HAS_TOAD = False
from binner import Binner
from metric import ks_auc



def _calculate_iv(feature_series, target_series, n_bins=10):
    '''
    计算单个特征的 IV 值，NaN 作为独立分箱参与计算

    :param feature_series: pd.Series, 特征列
    :param target_series: pd.Series, 目标列 (0/1)
    :param n_bins: int, 等频分箱数
    :return: float, IV 值
    '''
    eps = 1e-38

    # 构建分箱标签
    bins = pd.Series(index=feature_series.index, dtype='object')
    null_mask = feature_series.isna()

    if null_mask.all():
        # 全 NaN: 只有一个箱
        bins[:] = 'NaN_bin'
    else:
        non_null_mask = ~null_mask
        non_null_vals = feature_series[non_null_mask]

        # 等频分箱（duplicates='drop' 处理唯一值 < n_bins 的情况）
        try:
            binned = pd.qcut(non_null_vals, q=n_bins, duplicates='drop')
            bins[non_null_mask] = binned.astype(str)
        except ValueError:
            # 常量特征，所有非 NaN 值相同
            bins[non_null_mask] = 'const_bin'

        if null_mask.any():
            bins[null_mask] = 'NaN_bin'

    # 按分箱统计 event / non_event
    tmp = pd.DataFrame({'bin': bins, 'target': target_series})
    grouped = tmp.groupby('bin')['target'].agg(['sum', 'count'])
    grouped.columns = ['event', 'total']
    grouped['non_event'] = grouped['total'] - grouped['event']

    total_events = grouped['event'].sum()
    total_non_events = grouped['non_event'].sum()

    if total_events == 0 or total_non_events == 0:
        return 0.0

    # 计算各箱 WOE 和 IV
    event_pct = grouped['event'] / total_events
    non_event_pct = grouped['non_event'] / total_non_events

    woe = np.log((non_event_pct + eps) / (event_pct + eps))
    iv_bin = (non_event_pct - event_pct) * woe

    return float(iv_bin.sum())


def _empty_rate_filter(df, feature_list, threshold):
    '''
    返回缺失率超过阈值的特征列表

    :param df: DataFrame
    :param feature_list: list, 待检查特征
    :param threshold: float, 缺失率阈值 (如 0.9 表示缺失率 > 90% 则剔除)
    :return: list, 缺失率超标的特征列表
    '''
    n_rows = len(df)
    drop_list = []
    for fea in feature_list:
        miss_rate = df[fea].isna().sum() / n_rows
        if miss_rate > threshold:
            drop_list.append(fea)
    return drop_list


def _iv_filter(df, feature_list, target_col, threshold, n_bins=20):
    '''
    返回 IV 低于阈值的特征列表及所有特征的 IV 字典

    :param df: DataFrame
    :param feature_list: list, 待检查特征
    :param target_col: str, 目标列名
    :param threshold: float, IV 阈值 (如 0.02, IV < 阈值则剔除)
    :param n_bins: int, 分箱数
    :return: (iv_drop_list, iv_dict)
    '''
    iv_dict = {}
    iv_drop_list = []
    target = df[target_col]

    for fea in feature_list:
        iv_val = _calculate_iv(df[fea], target, n_bins=n_bins)
        iv_dict[fea] = iv_val
        if iv_val < threshold:
            iv_drop_list.append(fea)

    return iv_drop_list, iv_dict


def _corr_filter(df, feature_list, iv_dict, threshold):
    '''
    贪心法剔除高相关特征，保留 IV 更高的

    :param df: DataFrame
    :param feature_list: list, 待检查特征（已通过缺失率和 IV 筛选）
    :param iv_dict: dict, {特征名: IV值}
    :param threshold: float, 相关系数阈值 (如 0.7, 绝对值 > 阈值则剔除 IV 低的)
    :return: list, 被剔除的特征列表
    '''
    if len(feature_list) <= 1:
        return []

    # 计算皮尔逊相关矩阵
    corr_matrix = df[feature_list].corr().abs()

    # 按 IV 降序排列
    sorted_features = sorted(feature_list, key=lambda x: iv_dict.get(x, 0), reverse=True)

    keep_set = set()
    drop_set = set()

    for fea in sorted_features:
        if fea in drop_set:
            continue
        keep_set.add(fea)

        # 找出与当前特征高度相关、且尚未处理的特征
        if fea in corr_matrix.columns:
            corr_vals = corr_matrix[fea]
            for other_fea in sorted_features:
                if other_fea == fea or other_fea in keep_set or other_fea in drop_set:
                    continue
                if other_fea in corr_vals.index and corr_vals[other_fea] > threshold:
                    drop_set.add(other_fea)

    return list(drop_set)


def native_select(df, target_col, exclude, preselect_condition, exclude_var_list=[], forced_var_list=[], call_remain_iv=True):
    '''
    纯 numpy/pandas 实现的特征筛选，不依赖 toad 库
    签名和返回值与 toad_select 完全一致

    :param df: DataFrame
    :param target_col: str, 目标列名
    :param exclude: list, 排除列
    :param preselect_condition: dict, 筛选条件 {'empty': 0.9, 'iv': 0.02, 'corr': 0.7}
    :param exclude_var_list: list, 额外排除特征
    :param forced_var_list: list, 强制保留特征
    :param call_remain_iv: bool, 是否计算剩余特征的 IV
    :return: (toad_drop, all_drop, all_remain, remain_iv)
    '''
    empty_th = preselect_condition.get('empty', 0.9)
    iv_th = preselect_condition.get('iv', 0.02)
    corr_th = preselect_condition.get('corr', 0.7)

    print("native select by empty: {empty}, corr: {corr}, iv: {iv}".format(**preselect_condition))

    # 1. 提取特征列表
    all_fea_list = [i for i in df.columns if i not in exclude and i != target_col]
    exclude_var_list = list(set(all_fea_list) & set(exclude_var_list))

    feature_list = all_fea_list.copy()

    # 2. 缺失率筛选
    empty_drop = _empty_rate_filter(df, feature_list, empty_th)
    feature_list = [f for f in feature_list if f not in empty_drop]

    # 3. IV 筛选 (使用 n_bins=20，与 toad 内部一致)
    iv_drop, iv_dict = _iv_filter(df, feature_list, target_col, iv_th, n_bins=20)
    feature_list = [f for f in feature_list if f not in iv_drop]

    # 4. 相关性筛选
    corr_drop = _corr_filter(df, feature_list, iv_dict, corr_th)

    # 5. 组装结果（与 toad_select 格式一致）
    native_drop = {
        'empty': list(empty_drop),
        'iv': list(iv_drop),
        'corr': list(corr_drop),
    }

    all_drop = list(set([i for k, v in native_drop.items() for i in v]))
    native_drop['ex'] = [i for i in exclude_var_list if i not in all_drop]
    all_drop = all_drop + native_drop['ex']

    # 处理 forced_var_list
    native_drop = {k: [i for i in v if i not in forced_var_list] for k, v in native_drop.items()}
    all_drop = [i for i in all_drop if i not in forced_var_list]
    all_remain = [i for i in all_fea_list if i not in all_drop]
    print("total: {}, drop: {}, remain: {}".format(len(all_fea_list), len(all_drop), len(all_remain)))

    # 6. 计算剩余特征 IV (n_bins=10，与 toad_select 的 remain_iv 一致)
    if call_remain_iv:
        print("call remain feature iv")
        remain_iv = {fea: _calculate_iv(df[fea], df[target_col], n_bins=10) for fea in all_remain}
    else:
        remain_iv = dict()

    return native_drop, all_drop, all_remain, remain_iv


def toad_select(df, target_col, exclude, preselect_condition, exclude_var_list=[], forced_var_list=[], call_remain_iv=True):
    '''
    使用toad进行特征筛选
    '''
    print("toad select by empty: {empty}, corr: {corr}, iv: {iv}".format(**preselect_condition))
    all_fea_list = [i for i in df.columns if i not in exclude and i != target_col]
    exclude_var_list = list(set(all_fea_list) & set(exclude_var_list))
    dropped = toad.selection.select(df, target=target_col, return_drop=True, exclude=exclude, **preselect_condition)[1]
    toad_drop = {i: list(dropped[i]) for i in ['empty', 'corr', 'iv']}
    all_drop = list(set([i for k, v in toad_drop.items() for i in v]))
    toad_drop['ex'] = [i for i in exclude_var_list if i not in all_drop]
    all_drop = all_drop + toad_drop['ex']

    toad_drop = {k: [i for i in v if i not in forced_var_list] for k, v in toad_drop.items()}
    all_drop = [i for i in all_drop if i not in forced_var_list]
    all_remain = [i for i in all_fea_list if i not in all_drop]
    print("total: {}, drop: {}, remain: {}".format(len(all_fea_list), len(all_drop), len(all_remain)))

    if call_remain_iv: # 变量多的时候比较耗时
        print("call remain feature iv")
        remain_iv = {fea: toad.stats.IV(df[fea], target=df[target_col], n_bins=10, method='quantile') for fea in all_remain}
    else:
        remain_iv = dict()

    return toad_drop, all_drop, all_remain, remain_iv


def gen_data_iter(df, round_num=5, bagging_fraction=0.5):
    for _ in range(round_num):
        random_seed = np.random.randint(10000)
        sub_df = df.sample(frac=bagging_fraction, random_state=random_seed).reset_index(drop=True)
        print(sub_df.shape)

        yield sub_df


def d01_preselect_by_toad(df, target_col, feature_list, preselect_condition, round_num=500, use_native=None, max_round=None):
    '''
        使用toad进行初筛，支持自动降级到 native_select（不依赖 toad）

        use_native: None=自动检测(有toad用toad_select，无toad降级为native_select)
                    True=强制使用native_select
                    False=强制使用toad_select(无toad时报错)

        max_round: None=不限制迭代轮数(按原逻辑收敛即停), 正整数=最多迭代N轮
                   多轮迭代主要收敛相关性筛选(corr在分组内计算，跨组冗余需多轮)
                   iv和empty不受分组影响，第一轮即完成; 特征质量高时可设1

        递归迭代筛选, 每一轮按照round_num分成多组, 直到剩余变量数小于round_num或者没有变量没剔除
        如: 总共1200个特征
        round_0: [500, 500, 200], 剩余[100, 300, 150], 共550个， 大于round_num, 继续迭代
        round_1: [500, 50], 剩余[300, 50], 剩余350个, 小于round_num
        停止
    '''
    # 选择筛选函数
    if use_native is None:
        select_func = native_select if not _HAS_TOAD else toad_select
    elif use_native:
        select_func = native_select
    else:
        select_func = toad_select

    print(f"d01_preselect using: {select_func.__name__}")

    sub_feature_list = feature_list.copy()
    
    round_idx = 0
    round_select_rlt = dict()

    if len(sub_feature_list) <= round_num:
        round_num = len(sub_feature_list)
    
    # 剩余变量少于round_num 或 达到最大迭代轮数 则停止
    while len(sub_feature_list) >= round_num and (max_round is None or round_idx < max_round):
        print(f"=== round: {round_idx}")
        fea_set_list = [sub_feature_list[i:i+round_num] for i in range(0, len(sub_feature_list), round_num)]
        
        select_rlt_all = list()
        for idx, fea_set in enumerate(fea_set_list):
            select_rlt = select_func(df=df.loc[:, fea_set + [target_col]],
                                     target_col=target_col, 
                                     exclude=[], 
                                     preselect_condition=preselect_condition, 
                                     exclude_var_list=[], 
                                     forced_var_list=[], 
                                     call_remain_iv=False)
            select_rlt_all.append(select_rlt)
    
        all_remain_feature = [b for a in select_rlt_all for b in a[2]]
        round_select_rlt[round_idx] = select_rlt_all
        
        # 没有筛掉变量则停止(有可能是因为顺序分组的问题, 实际应该打乱后再重新分组尝试几次)
        if len(all_remain_feature) == len(sub_feature_list):
            print(len(all_remain_feature))
            break
            
        round_idx += 1
        sub_feature_list = all_remain_feature.copy() # 剩余变量

    return round_select_rlt


def batch_psi(data_iter, fea_cols, method='quantile', num_nbins=10):
    '''
    批量psi计算

    :param data_iter: iterator, 数据迭代器, 包括数据名称(data_name), 数据集(data), 如: iter([('train', train_df), ('eval', eval_df)]), base样本必须是第一个样本
    :param fea_cols: list, 需要计算psi的特征列表
    :param method: str, 分箱方法, 默认等频分箱
    :param num_nbins: int, 分箱数, 默认10
    '''
    
    eps = 1e-38
    psi_func = lambda x, y: (x - y) * np.log((x + eps) / (y + eps))
    
    fea_bin_info = dict()
    
    # 对照组必须是第一个
    data_name, base_data = next(data_iter)
    
    # 对照组分箱
    bin_info = dict()
    bin_obj = Binner()
    bin_obj.fit(df=base_data, exclude=[i for i in base_data.columns if i not in fea_cols], method=method, num_nbins=num_nbins)

    # 分箱映射统计
    for fea in fea_cols:
        bin_col = f'{fea}_bin'
        x_bin, x_label = bin_obj.transform(varname=fea, x=base_data[fea])
        bin_stt = x_bin.value_counts(dropna=False, normalize=True).sort_index()
        fea_bin_info.setdefault(fea, dict()).update({data_name: bin_stt})
    
    # 实验组分箱映射统计
    for data_name, exp_data in data_iter:
        for fea in fea_cols:
            x_bin, x_label = bin_obj.transform(varname=fea, x=exp_data[fea])
            bin_stt = x_bin.value_counts(dropna=False, normalize=True).sort_index()
            fea_bin_info.setdefault(fea, dict()).update({data_name: bin_stt})

    # 汇总分箱信息
    fea_bin_info = {fea: pd.concat(data_bin_info, axis=1).fillna(0).sort_index() for fea, data_bin_info in fea_bin_info.items()}
    fea_psi_info = {fea: pd.DataFrame(psi_func(bin_info.values, bin_info.iloc[:, 0].values.reshape(-1, 1)), columns=bin_info.columns, index=bin_info.index) for fea, bin_info in fea_bin_info.items()}
    fea_psi = {fea: psi_info.sum().to_dict() for fea, psi_info in fea_psi_info.items()}

    return fea_bin_info, fea_psi_info, fea_psi
            

def d02_psi_select(data_iter_list, psi_threshold=0.1, method='quantile', num_nbins=10):
    '''
    :param data_iter_list: list, 所有的data_iter, 通常是每个宽表中的剩余特征的data_iter
        如: [
            ('bigtable1', fea_list1, data_iter1),
            ('bigtable2', fea_list2, data_iter2),
        ]
    :param psi_threshold: float, psi阈值
    :param method: str, 分箱方法, 默认等频分箱
    :param num_nbins: int, 分箱数, 默认10
    '''
    all_psi_rlt = list()
    for bigtable, fea_list, data_iter in data_iter_list:
        print(f"psi select, table: {bigtable}, num_fea: {len(fea_list)}")
        psi_rlt = batch_psi(data_iter, fea_list, method, num_nbins)
        all_psi_rlt.append(psi_rlt)

    fea_max_psi = {fea: max(psi_info.values()) for psi_rlt in all_psi_rlt for fea, psi_info in psi_rlt[2].items()}
    psi_drop_fea = [fea for fea, psi in fea_max_psi.items() if psi > psi_threshold]

    return all_psi_rlt, fea_max_psi, psi_drop_fea


def select_by_importance(dataset, model_features, random_col, params_dict, thresholds=None, importance_type_list=['split', 'gain'], num_boost_round=1000, weight=1.0):
    '''
    根据随机数重要性和累计阈值筛选

    weight: 随机数重要性的权重, 剔除重要性小于 weight * 随机数重要性
    thresholds: 保留重要性top thresholds
    '''
    log_callback = lgb.log_evaluation(period=50, show_stdv=True)
    categorical_feature = 'auto'
    
    model = lgb.train(params={**params_dict, 'metric': ['auc']},
                    train_set=dataset, 
                    valid_sets=[dataset], 
                    valid_names=['INS'],
                    num_boost_round=num_boost_round, 
                    categorical_feature=categorical_feature, 
                    feature_name=model_features,
                    feval=None, 
                    callbacks=[log_callback])

    split_importance = model.feature_importance(importance_type='split')
    gain_importance = model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({'fea': model_features, 'split': split_importance, 'gain': gain_importance})

    # 筛选split大于随机数重要性的特征
    random_drop_features = list()
    zero_drop_features = list()
    for imp_typ in importance_type_list:
        random_imp = importance_df[importance_df.fea == random_col][imp_typ].iloc[0]
        random_drop_features += list(importance_df[(importance_df[imp_typ] < random_imp * weight) & (importance_df[imp_typ] > 0)].fea) # 小于随机数重要性 * weight 且 > 0
        zero_drop_features += list(importance_df[importance_df[imp_typ] == 0].fea) # 重要性为0

    random_drop_features = list(set(random_drop_features))
    zero_drop_features = list(set(zero_drop_features))

    importance_df = importance_df[~importance_df.fea.isin(random_drop_features + zero_drop_features)].reset_index(drop=True)

    thresholds_drop_features = list()
    if thresholds is not None:
        for imp_typ in importance_type_list:
            tmp_df = importance_df.sort_values(by=imp_typ, ascending=False)
            thresholds_drop_features += list(tmp_df[tmp_df[imp_typ].cumsum() / tmp_df[imp_typ].sum() > thresholds].fea)
            
            # fea_imp = dict(zip(importance_df['fea'], importance_df[imp_typ]))
            # th_gain_value = np.percentile(sorted(list(fea_imp.values())), [thresholds])
            # drop_features = [k for k, v in fea_imp.items() if v < th_gain_value]
            # thresholds_drop_features.extend(drop_features)

    thresholds_drop_features = list(set(thresholds_drop_features))
    all_drop_features = {'random': random_drop_features, 'zero': zero_drop_features, 'thresholds': thresholds_drop_features}
    print("drop info: ", {k: len(v) for k, v in all_drop_features.items()})

    return all_drop_features


def d03_random_importance_select(data_iter, model_features, target, random_col, params_dict, thresholds=None, importance_type_list=['split', 'gain'], num_boost_round=1000, weight=1.0, iter_round_num=1):
    '''
        多轮采样重要性筛选, 每轮迭代中根据随机数重要性和累计阈值筛选剔除特征, 继续使用剩余特征进行下一轮迭代, 最后取多轮迭代的剔除特征的并集
    '''
    round_select_rlt = list()
    for data in data_iter:
        model_features_new = model_features + [random_col]
        model_features_temp = model_features_new.copy()
        
        for round in range(iter_round_num):
            print(f"random-zero iter round: {round+1} / {iter_round_num}, num features: {len(model_features_temp)}")
            dataset = lgb.Dataset(data=data.loc[:, model_features_temp], label=data.loc[:, target].values, weight=None, categorical_feature='auto')
            print("start select ...")
            all_drop_features = select_by_importance(dataset, model_features_temp, random_col, params_dict, thresholds, importance_type_list, num_boost_round, weight)
            all_drop_features_list = all_drop_features['random'] + all_drop_features['zero'] + all_drop_features['thresholds']
            all_drop_features_list = list(set(all_drop_features_list))
            model_features_temp = all_drop_features_list.copy() # 使用剔除特征继续迭代
            model_features_temp = list(set(model_features_temp + [random_col]))
        
        round_select_rlt.append(all_drop_features) # 多轮迭代后的剔除特征

    # 取剔除特征的并集
    all_drop = list(set([i for itm in round_select_rlt for i in itm['random'] + itm['zero'] + itm['thresholds']]))

    return round_select_rlt, all_drop


def plot_importance_figure(feature, real_importance, null_importance, importance_type='split'):
    '''
    绘图展示null importance结果
    '''
    real_importance = [i[feature][importance_type] for i in real_importance]
    null_importance = [i[feature][importance_type] for i in null_importance]

    hist = np.histogram(null_importance)
    max_cnt = np.max(hist[0])

    fig = plt.figure(figsize=(5, 5))

    plt.hist(null_importance, label='Null importances', color='b')
    plt.vlines(x=np.mean(real_importance), ymin=0, ymax=max_cnt, color='r', linewidth=10, label='Real Target')
    
    plt.legend()
    plt.xlabel(f'Null Importance ({importance_type}) Distribution for {feature}')
    plt.title(f'{importance_type} importance of {feature}', fontweight='bold')

    return fig


class NullImportanceScore:
    '''
    多轮迭代计算null importance(lightgbm)
    '''
    def __init__(self, ins_df, model_features, target_col, params_dict, num_boost_round=1000, categorical_feature='auto'):
        self.ins_df = ins_df
        self.model_features = model_features
        self.target_col = target_col
        self.params_dict = params_dict
        self.categorical_feature = categorical_feature
        self.real_importance = None
        self.null_importance = None
        self.num_boost_round = num_boost_round
        
    def __fit_lgb_model(self, target):
        log_callback = lgb.log_evaluation(period=50, show_stdv=True)

        model_features = self.model_features.copy()
        
        ins_dataset = lgb.Dataset(data=self.ins_df.loc[:, model_features], label=target, weight=None, categorical_feature=self.categorical_feature)
        valid_sets = [ins_dataset]
        valid_names = ['INS']

        # 生成训练过程中的随机数, 降低随机性
        seed = np.random.randint(1, 1000, 1)
        
        model = lgb.train(params={**self.params_dict, 'metric': ['auc'], 'seed': seed},
                        train_set=ins_dataset, 
                        valid_sets=valid_sets, 
                        valid_names=valid_names,
                        num_boost_round=self.num_boost_round, 
                        categorical_feature=self.categorical_feature, 
                        feature_name=model_features,
                        feval=None, 
                        callbacks=[log_callback])
        
        return model

    def __get_importance(self, model):
        split_importance = model.feature_importance(importance_type='split')
        gain_importance = model.feature_importance(importance_type='gain')
        importance = {fea: {'split': split, 'gain': gain} for fea, split, gain in zip(self.model_features, split_importance, gain_importance)}

        return importance

    def __call_importance(self, round=5, shuffle=False):
        importance = list()
        for round_idx in range(round):
            print(f"=== fit model, shuffle {shuffle},  round {round_idx + 1} / {round}")
            target = self.ins_df[self.target_col].copy().values

            if shuffle:
                np.random.shuffle(target)
                
            model = self.__fit_lgb_model(target)
            round_importance = self.__get_importance(model)
            importance.append(round_importance)

        return importance

    def call_real_importance(self, round=5):
        self.real_importance = self.__call_importance(round, shuffle=False)
        
        return self.real_importance
        
    def call_null_importance(self, round=100):
        self.null_importance = self.__call_importance(round, shuffle=True)
        
        return self.null_importance

    def get_score(self, percent=75):
        score_info = dict()
        for fea in self.model_features:
            tmp_all_score = dict()
            for imp_typ in ['split', 'gain']:
                real_imp_list = [real_imp[fea][imp_typ] for real_imp in self.real_importance]
                null_imp_list = [null_imp[fea][imp_typ] for null_imp in self.null_importance]
                
                real_imp = np.mean(real_imp_list)
                imp_score = np.log(1e-10 + real_imp / (1 + np.percentile(null_imp_list, percent)))
                
                tmp_all_score[imp_typ] = imp_score

            score_info[fea] = tmp_all_score

        return score_info

    def plot_importance_figure(self, feature, importance_type='split'):
        fig = plot_importance_figure(feature, self.real_importance, self.null_importance, importance_type='split')
    
        return fig


def d04_select_by_null_importance(ins_df, oos_df, model_features, target, null_importance_score, thresholds_list, params_dict, categorical_feature='auto', num_boost_round=1000, gap=1e-3):
    '''
    根据null importance的结果进行特征筛选, 选择oos效果不下降的split/gain最优剔除特征组合(lightgbm)
    '''
    log_callback = lgb.log_evaluation(period=50, show_stdv=True)
    
    split_score_all = [(fea, null_importance_score[fea]['split']) for fea in model_features]
    gain_score_all = [(fea, null_importance_score[fea]['gain']) for fea in model_features]

    # split和gain的各个阈值的drop结果
    th_drop_info = dict()
    for threshold in thresholds_list:
        split_score_list = [i[1] for i in split_score_all]
        gain_score_list = [i[1] for i in split_score_all]

        th_split = np.percentile(split_score_list, threshold)
        th_gain = np.percentile(gain_score_list, threshold)

        split_drop = [fea for fea, imp in split_score_all if imp < th_split]
        gain_drop = [fea for fea, imp in gain_score_all if imp < th_gain]

        th_drop_info[threshold] = {'split': split_drop, 'gain': gain_drop}

    # 组合split和gain的结果, 找出oos效果不降的最优组合
    th_split_drop_all = [(k, v['split']) for k, v in th_drop_info.items()]
    th_gain_drop_all = [(k, v['gain']) for k, v in th_drop_info.items()]

    split_gain_set_auc = dict()
    for split_drop_set, gain_drop_set in product(th_split_drop_all, th_gain_drop_all):
        split_th = split_drop_set[0]
        gain_th = gain_drop_set[0]
        print(f"drop split top {split_th}%, gain top {gain_th}%")
        
        all_drop = list(set(split_drop_set[1] + gain_drop_set[1]))
        remain_features = [i for i in model_features if i not in all_drop]
        print("drop num: ", len(all_drop))

        ins_dataset = lgb.Dataset(data=ins_df.loc[:, remain_features], label=ins_df[target].values, weight=None, categorical_feature=categorical_feature)
        oos_dataset = lgb.Dataset(data=oos_df.loc[:, remain_features], label=oos_df[target].values, weight=None, categorical_feature=categorical_feature)
        valid_sets = [ins_dataset, oos_dataset]
        valid_names = ['INS', 'OOS']
        
        tmp_model = lgb.train(params={**params_dict, 'metric': ['auc']},
                        train_set=ins_dataset, 
                        valid_sets=valid_sets, 
                        valid_names=valid_names,
                        num_boost_round=num_boost_round, 
                        categorical_feature=categorical_feature, 
                        feature_name=remain_features,
                        feval=None, 
                        callbacks=[log_callback])
        
        tmp_oos_prb = tmp_model.predict(oos_df.loc[:, remain_features])
        tmp_oos_auc = ks_auc(tmp_oos_prb, oos_df[target])[1]

        split_gain_set_auc[(split_th, gain_th)] = tmp_oos_auc
        del tmp_model

    # base oos auc
    print("train base model with all features:")
    ins_dataset = lgb.Dataset(data=ins_df.loc[:, model_features], label=ins_df[target].values, weight=None, categorical_feature=categorical_feature)
    oos_dataset = lgb.Dataset(data=oos_df.loc[:, model_features], label=oos_df[target].values, weight=None, categorical_feature=categorical_feature)
    valid_sets = [ins_dataset, oos_dataset]
    valid_names = ['INS', 'OOS']
    
    base_model = lgb.train(params={**params_dict, 'metric': ['auc']},
                    train_set=ins_dataset, 
                    valid_sets=valid_sets, 
                    valid_names=valid_names,
                    num_boost_round=num_boost_round, 
                    categorical_feature=categorical_feature, 
                    feature_name=model_features,
                    feval=None, 
                    callbacks=[log_callback])
        
    base_prb = base_model.predict(oos_df.loc[:, model_features])
    base_oos_auc = ks_auc(base_prb, oos_df[target])[1]
    print('base oos auc: ', base_oos_auc)

    # 选择误差范围内的最优组合
    candidate_set = [(k, v) for k, v in split_gain_set_auc.items() if v >= (base_oos_auc - gap)]
    if not candidate_set:
        print("no candidate drop set")
        best_th_set = None
    else:
        best_th_set = sorted(candidate_set, key=lambda x: x[1], reverse=True)[0]
        print(best_th_set)
    
    return th_drop_info, split_gain_set_auc, best_th_set
    

def d05_select_by_top_importance(ins_df, oos_df, model_features, target, params_dict,
                                  thresholds_list=None, top_n_list=None,
                                  importance_type='gain', categorical_feature='auto',
                                  num_boost_round=1000, gap=1e-3):
    '''
    根据给定的重要性阈值或者最优的阈值进行截断, 遍历阈值区间, 保留模型效果下降幅度在一定范围内的top重要性特征

    支持两种截断模式:
    - thresholds_list: 按累计重要性占比截断, 如[0.5, 0.6, 0.7, 0.8, 0.9]表示保留累计重要性前50%/60%/.../90%的特征
    - top_n_list: 按特征个数截断, 如[20, 50, 100]表示保留重要性top 20/50/100的特征
    两者至少指定一个, 都指定时取并集

    :param ins_df: DataFrame, INS样本数据
    :param oos_df: DataFrame, OOS样本数据
    :param model_features: list, 模型特征列表
    :param target: str, 目标列名
    :param params_dict: dict, LightGBM参数
    :param thresholds_list: list, 累计重要性占比阈值列表, 如[0.5, 0.6, 0.7, 0.8, 0.9]
    :param top_n_list: list, top N特征个数列表, 如[20, 50, 100]
    :param importance_type: str, 重要性类型, 'gain'或'split'
    :param categorical_feature: str/list, 类别型特征
    :param num_boost_round: int, 训练轮数
    :param gap: float, OOS AUC容忍下降幅度, 默认1e-3
    '''
    log_callback = lgb.log_evaluation(period=50, show_stdv=True)

    if thresholds_list is None and top_n_list is None:
        raise ValueError("thresholds_list 和 top_n_list 至少指定一个")

    # 1. 训练基准模型, 获取特征重要性
    print("train base model to get feature importance ...")
    ins_dataset = lgb.Dataset(data=ins_df.loc[:, model_features], label=ins_df[target].values, weight=None, categorical_feature=categorical_feature)
    oos_dataset = lgb.Dataset(data=oos_df.loc[:, model_features], label=oos_df[target].values, weight=None, categorical_feature=categorical_feature)

    base_model = lgb.train(params={**params_dict, 'metric': ['auc']},
                    train_set=ins_dataset,
                    valid_sets=[ins_dataset, oos_dataset],
                    valid_names=['INS', 'OOS'],
                    num_boost_round=num_boost_round,
                    categorical_feature=categorical_feature,
                    feature_name=model_features,
                    feval=None,
                    callbacks=[log_callback])

    base_prb = base_model.predict(oos_df.loc[:, model_features])
    base_oos_ks, base_oos_auc, _ = ks_auc(base_prb, oos_df[target])
    print(f"base model: num_features={len(model_features)}, OOS AUC={base_oos_auc:.6f}, OOS KS={base_oos_ks:.6f}")

    # 2. 获取重要性并排序
    importance_values = base_model.feature_importance(importance_type=importance_type)
    importance_df = pd.DataFrame({'fea': model_features, importance_type: importance_values})
    importance_df = importance_df.sort_values(by=importance_type, ascending=False).reset_index(drop=True)
    importance_df['cumsum_pct'] = importance_df[importance_type].cumsum() / importance_df[importance_type].sum()
    importance_df['rank'] = range(1, len(importance_df) + 1)

    print(f"importance distribution ({importance_type}):")
    print(f"  mean={importance_df[importance_type].mean():.4f}, median={importance_df[importance_type].median():.4f}")
    print(f"  top10 cumsum_pct={importance_df.loc[9, 'cumsum_pct']:.4f}" if len(importance_df) > 10 else "")

    # 3. 构建候选截断方案
    candidate_sets = dict()  # {label: remain_features}

    if thresholds_list is not None:
        for th in sorted(thresholds_list):
            remain = list(importance_df[importance_df['cumsum_pct'] <= th]['fea'])
            # 至少保留累计重要性刚好超过阈值的那个特征
            if len(remain) < len(importance_df):
                next_fea = importance_df.iloc[len(remain)]['fea']
                if next_fea not in remain:
                    remain.append(next_fea)
            label = f"cumsum_{th}"
            candidate_sets[label] = remain
            print(f"  threshold cumsum {th}: keep {len(remain)} / {len(model_features)} features")

    if top_n_list is not None:
        for n in sorted(top_n_list):
            n = min(n, len(model_features))
            remain = list(importance_df.iloc[:n]['fea'])
            label = f"top_{n}"
            candidate_sets[label] = remain
            print(f"  top {n}: keep {len(remain)} / {len(model_features)} features")

    # 4. 对每个候选方案训练模型并评估OOS效果
    candidate_auc = dict()
    for label, remain_features in candidate_sets.items():
        if len(remain_features) == 0:
            print(f"  {label}: no features, skip")
            continue
        if len(remain_features) == len(model_features):
            print(f"  {label}: same as base, skip")
            candidate_auc[label] = base_oos_auc
            continue

        print(f"evaluate {label}: {len(remain_features)} features ...")
        tmp_ins_dataset = lgb.Dataset(data=ins_df.loc[:, remain_features], label=ins_df[target].values, weight=None, categorical_feature=categorical_feature)
        tmp_oos_dataset = lgb.Dataset(data=oos_df.loc[:, remain_features], label=oos_df[target].values, weight=None, categorical_feature=categorical_feature)

        tmp_model = lgb.train(params={**params_dict, 'metric': ['auc']},
                        train_set=tmp_ins_dataset,
                        valid_sets=[tmp_ins_dataset, tmp_oos_dataset],
                        valid_names=['INS', 'OOS'],
                        num_boost_round=num_boost_round,
                        categorical_feature=categorical_feature,
                        feature_name=remain_features,
                        feval=None,
                        callbacks=[log_callback])

        tmp_oos_prb = tmp_model.predict(oos_df.loc[:, remain_features])
        tmp_oos_auc = ks_auc(tmp_oos_prb, oos_df[target])[1]
        candidate_auc[label] = tmp_oos_auc
        print(f"  {label}: OOS AUC={tmp_oos_auc:.6f}, diff={tmp_oos_auc - base_oos_auc:.6f}")

    # 5. 选择OOS效果不降的最优方案(特征最少的)
    valid_candidates = [(label, auc, len(candidate_sets[label]))
                        for label, auc in candidate_auc.items()
                        if auc >= (base_oos_auc - gap)]

    if not valid_candidates:
        print("no valid candidate within gap tolerance")
        best_set = None
        best_remain = model_features
    else:
        # 选择特征数最少的候选方案
        best_set = sorted(valid_candidates, key=lambda x: x[2])[0]
        best_remain = candidate_sets[best_set[0]]
        print(f"best: {best_set[0]}, num_features={best_set[2]}, OOS AUC={best_set[1]:.6f}")

    drop_features = [f for f in model_features if f not in best_remain]

    return importance_df, candidate_auc, best_set, drop_features


def d06_select_by_shap(ins_df, oos_df, model_features, target, params_dict,
                        thresholds_list=None, top_n_list=None,
                        categorical_feature='auto', num_boost_round=1000, gap=1e-3):
    '''
    基于SHAP值进行特征筛选, 使用TreeExplainer计算每个特征的mean(|SHAP|)作为全局重要性,
    遍历候选阈值, 保留OOS效果不下降的最精简特征子集

    支持两种截断模式:
    - thresholds_list: 按累计SHAP重要性占比截断, 如[0.5, 0.6, 0.7, 0.8, 0.9]
    - top_n_list: 按特征个数截断, 如[20, 50, 100]
    两者至少指定一个, 都指定时取并集

    :param ins_df: DataFrame, INS样本数据
    :param oos_df: DataFrame, OOS样本数据
    :param model_features: list, 模型特征列表
    :param target: str, 目标列名
    :param params_dict: dict, LightGBM参数
    :param thresholds_list: list, 累计SHAP重要性占比阈值列表, 如[0.5, 0.6, 0.7, 0.8, 0.9]
    :param top_n_list: list, top N特征个数列表, 如[20, 50, 100]
    :param categorical_feature: str/list, 类别型特征
    :param num_boost_round: int, 训练轮数
    :param gap: float, OOS AUC容忍下降幅度, 默认1e-3

    :return: (shap_importance_df, candidate_auc, best_set, drop_features)
    - shap_importance_df: DataFrame, 特征SHAP重要性排名表, 包含列: fea, mean_abs_shap, cumsum_pct, rank
    - candidate_auc: dict, 各候选方案的OOS AUC, {label: auc}
    - best_set: tuple/None, 最优方案(label, auc, num_features), 无满足条件时为None
    - drop_features: list, 被剔除的特征列表
    '''
    import shap

    log_callback = lgb.log_evaluation(period=50, show_stdv=True)

    if thresholds_list is None and top_n_list is None:
        raise ValueError("thresholds_list 和 top_n_list 至少指定一个")

    # 1. 训练基准模型
    print("train base model ...")
    ins_dataset = lgb.Dataset(data=ins_df.loc[:, model_features], label=ins_df[target].values, weight=None, categorical_feature=categorical_feature)
    oos_dataset = lgb.Dataset(data=oos_df.loc[:, model_features], label=oos_df[target].values, weight=None, categorical_feature=categorical_feature)

    base_model = lgb.train(params={**params_dict, 'metric': ['auc']},
                    train_set=ins_dataset,
                    valid_sets=[ins_dataset, oos_dataset],
                    valid_names=['INS', 'OOS'],
                    num_boost_round=num_boost_round,
                    categorical_feature=categorical_feature,
                    feature_name=model_features,
                    feval=None,
                    callbacks=[log_callback])

    base_prb = base_model.predict(oos_df.loc[:, model_features])
    base_oos_ks, base_oos_auc, _ = ks_auc(base_prb, oos_df[target])
    print(f"base model: num_features={len(model_features)}, OOS AUC={base_oos_auc:.6f}, OOS KS={base_oos_ks:.6f}")

    # 2. 计算SHAP值
    print("computing SHAP values on INS data ...")
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(ins_df.loc[:, model_features])

    # 对于二分类, shap_values可能返回list[array, array], 取正类
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # 计算每个特征的mean(|SHAP|)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    del shap_values, explainer
    import gc; gc.collect()
    shap_importance_df = pd.DataFrame({'fea': model_features, 'mean_abs_shap': mean_abs_shap})
    shap_importance_df = shap_importance_df.sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
    shap_importance_df['cumsum_pct'] = shap_importance_df['mean_abs_shap'].cumsum() / shap_importance_df['mean_abs_shap'].sum()
    shap_importance_df['rank'] = range(1, len(shap_importance_df) + 1)

    print(f"SHAP importance distribution:")
    print(f"  mean={shap_importance_df['mean_abs_shap'].mean():.6f}, median={shap_importance_df['mean_abs_shap'].median():.6f}")
    if len(shap_importance_df) > 10:
        print(f"  top10 cumsum_pct={shap_importance_df.loc[9, 'cumsum_pct']:.4f}")

    # 3. 构建候选截断方案
    candidate_sets = dict()

    if thresholds_list is not None:
        for th in sorted(thresholds_list):
            remain = list(shap_importance_df[shap_importance_df['cumsum_pct'] <= th]['fea'])
            if len(remain) < len(shap_importance_df):
                next_fea = shap_importance_df.iloc[len(remain)]['fea']
                if next_fea not in remain:
                    remain.append(next_fea)
            label = f"shap_cumsum_{th}"
            candidate_sets[label] = remain
            print(f"  threshold cumsum {th}: keep {len(remain)} / {len(model_features)} features")

    if top_n_list is not None:
        for n in sorted(top_n_list):
            n = min(n, len(model_features))
            remain = list(shap_importance_df.iloc[:n]['fea'])
            label = f"shap_top_{n}"
            candidate_sets[label] = remain
            print(f"  top {n}: keep {len(remain)} / {len(model_features)} features")

    # 4. 对每个候选方案训练模型并评估OOS效果
    candidate_auc = dict()
    for label, remain_features in candidate_sets.items():
        if len(remain_features) == 0:
            print(f"  {label}: no features, skip")
            continue
        if len(remain_features) == len(model_features):
            print(f"  {label}: same as base, skip")
            candidate_auc[label] = base_oos_auc
            continue

        print(f"evaluate {label}: {len(remain_features)} features ...")
        tmp_ins_dataset = lgb.Dataset(data=ins_df.loc[:, remain_features], label=ins_df[target].values, weight=None, categorical_feature=categorical_feature)
        tmp_oos_dataset = lgb.Dataset(data=oos_df.loc[:, remain_features], label=oos_df[target].values, weight=None, categorical_feature=categorical_feature)

        tmp_model = lgb.train(params={**params_dict, 'metric': ['auc']},
                        train_set=tmp_ins_dataset,
                        valid_sets=[tmp_ins_dataset, tmp_oos_dataset],
                        valid_names=['INS', 'OOS'],
                        num_boost_round=num_boost_round,
                        categorical_feature=categorical_feature,
                        feature_name=remain_features,
                        feval=None,
                        callbacks=[log_callback])

        tmp_oos_prb = tmp_model.predict(oos_df.loc[:, remain_features])
        tmp_oos_auc = ks_auc(tmp_oos_prb, oos_df[target])[1]
        candidate_auc[label] = tmp_oos_auc
        print(f"  {label}: OOS AUC={tmp_oos_auc:.6f}, diff={tmp_oos_auc - base_oos_auc:.6f}")

    # 5. 选择OOS效果不降的最优方案(特征最少的)
    valid_candidates = [(label, auc, len(candidate_sets[label]))
                        for label, auc in candidate_auc.items()
                        if auc >= (base_oos_auc - gap)]

    if not valid_candidates:
        print("no valid candidate within gap tolerance")
        best_set = None
        best_remain = model_features
    else:
        best_set = sorted(valid_candidates, key=lambda x: x[2])[0]
        best_remain = candidate_sets[best_set[0]]
        print(f"best: {best_set[0]}, num_features={best_set[2]}, OOS AUC={best_set[1]:.6f}")

    drop_features = [f for f in model_features if f not in best_remain]

    return shap_importance_df, candidate_auc, best_set, drop_features


def d07_select_by_woe_trend(all_fea_stt, compare_tw_list, dev_tw='DEV_INS',
                             min_cnt_pct=0.05, metric='woe',
                             concordance_threshold=0.6,
                             range_ratio_threshold=0.3,
                             spearman_threshold=0.4,
                             require_all_windows=True,
                             forced_var_list=None):
    '''
    根据WOE趋势稳定性筛选特征, 排除在OOT时间窗口上趋势倒挂或区分力消失的特征

    使用三个互补指标进行复合判断:
    1. concordance(一致性比率): 检测趋势反转, 基于所有bin对的WOE相对顺序一致性
    2. range_ratio(WOE范围比): 检测区分力丧失, OOT的WOE范围/DEV的WOE范围
    3. spearman(Spearman秩相关): 综合秩相关补充指标

    :param all_fea_stt: dict, {feature_name: [DataFrame, ...]}, 来自split_plot_feature的输出
    :param compare_tw_list: list[str], OOT时间窗口名称列表, 如['OOT1', 'OOT2']
    :param dev_tw: str, DEV时间窗口名称, 默认'DEV_INS'
    :param min_cnt_pct: float, 最小分组占比阈值, 低于此值的分组不参与计算
    :param metric: str, 趋势比较的指标, 'woe'或'tgt_rate'
    :param concordance_threshold: float, 一致性比率阈值, 低于此值判定为趋势反转, 默认0.6
    :param range_ratio_threshold: float, WOE范围比阈值, 低于此值判定为区分力丧失, 默认0.3
    :param spearman_threshold: float, Spearman相关系数阈值, 默认0.4
    :param require_all_windows: bool, True=要求在所有OOT窗口都满足阈值(严格), False=所有窗口都不满足才剔除(宽松)
    :param forced_var_list: list, 强制保留的特征列表

    :return: (stability_detail, drop_reasons, drop_features)
    - stability_detail: DataFrame, 所有特征在每个OOT窗口的三项指标明细
    - drop_reasons: dict, {feature_name: [reason_str, ...]}, 被剔除特征及原因
    - drop_features: list, 被剔除的特征列表
    '''
    if forced_var_list is None:
        forced_var_list = []

    # 1. 计算所有特征的稳定性指标
    print(f"computing WOE trend stability: {len(all_fea_stt)} features x {len(compare_tw_list)} OOT windows ...")
    stability_scores = cal_woe_stability(all_fea_stt, compare_tw_list, dev_tw, min_cnt_pct, metric)

    # 2. 构建明细表并判断通过/不通过
    detail_rows = []
    drop_reasons = dict()
    fea_fail_windows = dict()  # {fea: [是否在该窗口失败]}

    for fea, tw_scores in stability_scores.items():
        fea_any_fail = False
        fea_all_fail = True

        for tw, scores in tw_scores.items():
            conc = scores['concordance']
            rr = scores['range_ratio']
            sp = scores['spearman']
            n_bins = scores['n_bins']

            # nan视为通过(数据不足不惩罚)
            pass_conc = True if np.isnan(conc) else (conc >= concordance_threshold)
            pass_rr = True if np.isnan(rr) else (rr >= range_ratio_threshold)
            pass_sp = True if np.isnan(sp) else (sp >= spearman_threshold)
            pass_all = pass_conc and pass_rr and pass_sp

            detail_rows.append({
                'feature': fea,
                'compare_tw': tw,
                'concordance': conc,
                'range_ratio': rr,
                'spearman': sp,
                'n_bins': n_bins,
                'dev_range': scores['dev_range'],
                'oot_range': scores['oot_range'],
                'pass_concordance': pass_conc,
                'pass_range_ratio': pass_rr,
                'pass_spearman': pass_sp,
                'pass_all': pass_all,
            })

            if not pass_all:
                fea_any_fail = True
                reasons = drop_reasons.setdefault(fea, [])
                if not pass_conc:
                    reasons.append(f"{tw}: concordance={conc:.3f}<{concordance_threshold} (趋势反转)")
                if not pass_rr:
                    reasons.append(f"{tw}: range_ratio={rr:.3f}<{range_ratio_threshold} (区分力丧失)")
                if not pass_sp:
                    reasons.append(f"{tw}: spearman={sp:.3f}<{spearman_threshold} (秩相关不足)")
            else:
                fea_all_fail = False

        fea_fail_windows[fea] = (fea_any_fail, fea_all_fail)

    stability_detail = pd.DataFrame(detail_rows)

    # 3. 根据策略确定剔除特征
    drop_features = []
    for fea, (any_fail, all_fail) in fea_fail_windows.items():
        if fea in forced_var_list:
            continue
        if require_all_windows and any_fail:
            drop_features.append(fea)
        elif not require_all_windows and all_fail:
            drop_features.append(fea)

    # 移除未实际剔除的特征的 drop_reasons
    drop_reasons = {fea: reasons for fea, reasons in drop_reasons.items() if fea in drop_features}

    remain_count = len(all_fea_stt) - len(drop_features)
    print(f"d07 WOE trend: total={len(all_fea_stt)}, drop={len(drop_features)}, remain={remain_count}")
    for fea, reasons in drop_reasons.items():
        print(f"  drop {fea}: {'; '.join(reasons)}")

    return stability_detail, drop_reasons, drop_features


def d08_select_by_woe_explain(all_fea_stt, feature_comment_map, dev_tw='DEV', compare_tw_list=None):
    '''
    为剩余特征生成 WOE 趋势摘要，供 AI 结合业务知识生成解释性评估。

    :param all_fea_stt: dict, {feature_name: [DataFrame, ...]}, 来自 split_plot_feature 的输出
    :param feature_comment_map: dict, {feature_name: 中文名/描述}
    :param dev_tw: str, DEV 窗口名称
    :param compare_tw_list: list, OOT 窗口名称列表，None 则自动从数据中提取非 DEV 窗口
    :return: (woe_summary_list, woe_summary_text)
        - woe_summary_list: list of dict, 每个特征的结构化 WOE 摘要
        - woe_summary_text: str, 格式化文本摘要，供 AI 生成解释
    '''
    woe_summary_list = []
    text_blocks = []

    for fea, tw_stt_list in all_fea_stt.items():
        comment = feature_comment_map.get(fea, '')

        # 按窗口整理 WOE 数据
        woe_by_window = {}
        for tw_df in tw_stt_list:
            if tw_df is None or len(tw_df) == 0:
                continue
            tw_name = tw_df['time_window'].iloc[0] if 'time_window' in tw_df.columns else 'unknown'
            bins = tw_df['bin'].tolist() if 'bin' in tw_df.columns else []
            woes = tw_df['woe'].tolist() if 'woe' in tw_df.columns else []
            tgt_rates = tw_df['tgt_rate'].tolist() if 'tgt_rate' in tw_df.columns else []
            cnt_pcts = tw_df['cnt_pct'].tolist() if 'cnt_pct' in tw_df.columns else []
            woe_by_window[tw_name] = {
                'bins': bins, 'woes': woes,
                'tgt_rates': tgt_rates, 'cnt_pcts': cnt_pcts
            }

        # 自动提取 OOT 窗口
        all_windows = list(woe_by_window.keys())
        if compare_tw_list is None:
            oot_windows = [tw for tw in all_windows if tw != dev_tw]
        else:
            oot_windows = [tw for tw in compare_tw_list if tw in woe_by_window]

        # 计算 DEV 窗口 WOE 单调性
        monotonicity = np.nan
        trend_direction = '未知'
        dev_woes = []
        if dev_tw in woe_by_window:
            dev_data = woe_by_window[dev_tw]
            dev_woes = [w for w in dev_data['woes'] if not (isinstance(w, float) and np.isnan(w))]
            if len(dev_woes) >= 2:
                diffs = [dev_woes[i+1] - dev_woes[i] for i in range(len(dev_woes) - 1)]
                pos = sum(1 for d in diffs if d > 0)
                neg = sum(1 for d in diffs if d < 0)
                total_diffs = len(diffs)
                monotonicity = max(pos, neg) / total_diffs if total_diffs > 0 else np.nan
                if pos > neg:
                    trend_direction = '递增'
                elif neg > pos:
                    trend_direction = '递减'
                else:
                    trend_direction = '非单调'

        # 计算各 OOT 窗口与 DEV 的一致性
        window_consistency = {}
        for oot_tw in oot_windows:
            if oot_tw in woe_by_window and len(dev_woes) >= 2:
                oot_data = woe_by_window[oot_tw]
                oot_woes = [w for w in oot_data['woes'] if not (isinstance(w, float) and np.isnan(w))]
                min_len = min(len(dev_woes), len(oot_woes))
                if min_len >= 2:
                    concordance = _concordance_rate(
                        np.array(dev_woes[:min_len]), np.array(oot_woes[:min_len]))
                    window_consistency[oot_tw] = round(concordance, 4)

        summary = {
            'feature_name': fea,
            'comment': comment,
            'monotonicity': round(monotonicity, 4) if not np.isnan(monotonicity) else None,
            'trend_direction': trend_direction,
            'window_consistency': window_consistency,
            'woe_by_window': woe_by_window,
        }
        woe_summary_list.append(summary)

        # 格式化文本块
        block = f"### 特征: {fea}\n"
        block += f"中文名: {comment}\n"
        block += f"单调性: {monotonicity:.2f} ({trend_direction})\n" if not np.isnan(monotonicity) else f"单调性: 无法计算\n"
        if window_consistency:
            consistency_str = ', '.join([f"{tw}={v:.2f}" for tw, v in window_consistency.items()])
            block += f"窗口一致性: {consistency_str}\n"
        block += "\n"
        for tw_name, tw_data in woe_by_window.items():
            block += f"[{tw_name}]\n"
            block += f"{'bin':<30} {'woe':>8} {'tgt_rate':>10} {'cnt_pct':>10}\n"
            for i in range(len(tw_data['bins'])):
                b = str(tw_data['bins'][i])[:28] if i < len(tw_data['bins']) else ''
                w = f"{tw_data['woes'][i]:.4f}" if i < len(tw_data['woes']) else ''
                t = f"{tw_data['tgt_rates'][i]:.4f}" if i < len(tw_data['tgt_rates']) else ''
                c = f"{tw_data['cnt_pcts'][i]:.4f}" if i < len(tw_data['cnt_pcts']) else ''
                block += f"{b:<30} {w:>8} {t:>10} {c:>10}\n"
            block += "\n"
        text_blocks.append(block)

    woe_summary_text = '\n'.join(text_blocks)

    print(f"[d08] 共生成 {len(woe_summary_list)} 个特征的 WOE 摘要")
    return woe_summary_list, woe_summary_text


def _concordance_rate(woe_dev, woe_oot):
    '''
    WOE序对一致性比率, 本质是Kendall's tau归一化到[0, 1]
    对所有C(n,2)个bin对, 检查WOE差值方向是否一致
    concordant: 方向相同; discordant: 方向相反; tied: 某侧差值为0
    返回 (concordant + 0.5 * tied) / total, 范围[0, 1]
    '''
    from itertools import combinations
    n = len(woe_dev)
    if n < 2:
        return np.nan
    concordant = 0
    discordant = 0
    tied = 0
    for i, j in combinations(range(n), 2):
        diff_dev = woe_dev[i] - woe_dev[j]
        diff_oot = woe_oot[i] - woe_oot[j]
        product = diff_dev * diff_oot
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1
        else:
            tied += 1
    total = concordant + discordant + tied
    return (concordant + 0.5 * tied) / total


def _range_ratio(woe_dev, woe_oot, eps=1e-10):
    '''WOE范围比 = OOT范围 / DEV范围, 接近0说明OOT区分力丧失'''
    dev_range = np.max(woe_dev) - np.min(woe_dev)
    oot_range = np.max(woe_oot) - np.min(woe_oot)
    return oot_range / (dev_range + eps)


def _spearman_corr(woe_dev, woe_oot):
    '''Spearman秩相关, 比Pearson更鲁棒; 样本<3或方差为0时返回nan'''
    if len(woe_dev) < 3:
        return np.nan
    if np.std(woe_dev) < 1e-10 or np.std(woe_oot) < 1e-10:
        return np.nan
    corr, _ = spearmanr(woe_dev, woe_oot)
    return corr


def cal_woe_stability(all_fea_stt, compare_tw_list, dev_tw='DEV_INS', min_cnt_pct=0.05, metric='woe'):
    '''
    计算每个特征在DEV与多个OOT时间窗口之间的WOE趋势稳定性指标(三指标)

    :param all_fea_stt: dict, {feature_name: [DataFrame, ...]}, 来自split_plot_feature的输出
    :param compare_tw_list: list[str], OOT时间窗口名称列表
    :param dev_tw: str, DEV时间窗口名称
    :param min_cnt_pct: float, 最小分组占比阈值, 占比低于此值的分组不参与计算
    :param metric: str, 使用的指标列名, 默认'woe', 也可以是'tgt_rate'

    :return: dict, {fea: {compare_tw: {concordance, range_ratio, spearman, n_bins, dev_range, oot_range}}}
    '''
    if isinstance(compare_tw_list, str):
        compare_tw_list = [compare_tw_list]

    woe_stability = dict()
    for fea, all_tw_stt_list in all_fea_stt.items():
        # 获取DEV stt
        dev_stt_list = [i for i in all_tw_stt_list if i.time_window.iloc[0] == dev_tw]
        if not dev_stt_list:
            print(f"  warning: {fea} has no DEV window '{dev_tw}', skip")
            continue
        dev_stt = dev_stt_list[0]

        # 过滤: 排除NULL bin, 排除cnt_pct过小的bin
        dev_filtered = dev_stt[(dev_stt['bin'] != '000:NULL') & (dev_stt['cnt_pct'] >= min_cnt_pct)]
        dev_filtered = dev_filtered.loc[:, ['bin', 'cnt_pct', metric]].rename(columns={metric: f'{metric}_dev'})

        fea_scores = dict()
        for compare_tw in compare_tw_list:
            compare_stt_list = [i for i in all_tw_stt_list if i.time_window.iloc[0] == compare_tw]
            if not compare_stt_list:
                print(f"  warning: {fea} has no window '{compare_tw}', skip")
                fea_scores[compare_tw] = {'concordance': np.nan, 'range_ratio': np.nan, 'spearman': np.nan,
                                           'n_bins': 0, 'dev_range': np.nan, 'oot_range': np.nan}
                continue

            compare_stt = compare_stt_list[0]
            compare_filtered = compare_stt[compare_stt['bin'] != '000:NULL']
            compare_filtered = compare_filtered.loc[:, ['bin', metric]].rename(columns={metric: f'{metric}_oot'})

            # 内连接: 仅对比两边都有的bin
            merged = pd.merge(left=dev_filtered, right=compare_filtered, on='bin', how='inner')

            if len(merged) < 2:
                print(f"  warning: {fea} @ {compare_tw}: only {len(merged)} valid bins after filtering, skip")
                fea_scores[compare_tw] = {'concordance': np.nan, 'range_ratio': np.nan, 'spearman': np.nan,
                                           'n_bins': len(merged), 'dev_range': np.nan, 'oot_range': np.nan}
                continue

            woe_dev = merged[f'{metric}_dev'].values
            woe_oot = merged[f'{metric}_oot'].values

            dev_range = float(np.max(woe_dev) - np.min(woe_dev))
            oot_range = float(np.max(woe_oot) - np.min(woe_oot))

            fea_scores[compare_tw] = {
                'concordance': _concordance_rate(woe_dev, woe_oot),
                'range_ratio': _range_ratio(woe_dev, woe_oot),
                'spearman': _spearman_corr(woe_dev, woe_oot),
                'n_bins': len(merged),
                'dev_range': dev_range,
                'oot_range': oot_range,
            }

        woe_stability[fea] = fea_scores

    return woe_stability


def replace_special_values(df, feature_list, categorical_features=None):
    '''
    替换特征中的特殊值为 np.nan, 在筛选流程开始前执行

    :param df: DataFrame, 包含特征的数据集
    :param feature_list: list, 需要处理的特征列表
    :param categorical_features: list/None, 类别型特征列表, None时自动识别object/category类型列
    :return: DataFrame, 处理后的数据集(就地修改)
    '''
    # 数值型特殊值
    numeric_special = {-999.0, -998.0, np.inf, -np.inf}
    # 类别型特殊值
    categorical_special = {"", "D#999", "None", "none", "NONE", "NULL", "Null", "null",
                           "nan", "NAN", "Nan", " ", "NA"}

    if categorical_features is None:
        categorical_features = [col for col in feature_list if col in df.columns and df[col].dtype in ('object', 'category')]

    categorical_set = set(categorical_features)
    numeric_features = [col for col in feature_list if col in df.columns and col not in categorical_set]

    # 替换数值型特殊值
    replaced_num = 0
    for col in numeric_features:
        mask = df[col].isin(numeric_special)
        cnt = mask.sum()
        if cnt > 0:
            df.loc[mask, col] = np.nan
            replaced_num += cnt

    # 替换类别型特殊值
    replaced_cat = 0
    for col in categorical_features:
        if col in df.columns:
            mask = df[col].isin(categorical_special)
            cnt = mask.sum()
            if cnt > 0:
                df.loc[mask, col] = np.nan
                replaced_cat += cnt

    print(f"replace_special_values: {len(numeric_features)} numeric cols ({replaced_num} values), "
          f"{len(categorical_features)} categorical cols ({replaced_cat} values)")

    return df