# 汇总所有筛选流程的结果并生成报告
import os
import gc
import time
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from tmlpatch.database import TMLSQLClient
from procs.base_proc import BaseProc
from utils.data_utility import str_format, safe_sql_execute, safe_pickle_dump, cons_join_sql
from utils.log_config import get_main_logger
from utils.metric import ks_auc, plot_roc_multi
from utils.binner import split_plot_feature
from utils.decorators import capture_print
from utils.remain_resolver import resolve_flat_remain_features, resolve_table_remain_fea
from utils.param_manager import resolve_params



class Proc03Summary(BaseProc):
    '''
    汇总所有筛选步骤的结果, 训练基线模型, 生成固化排版的markdown报告
    '''

    PROC_CACHE_NAME = 'Proc03Summary'

    def __init__(self, config):
        super().__init__(config)

        # 校验 metadata 是否存在
        metadata_save_path = os.path.join(config['project_path'], 'data', 'metadata.pkl')
        if not os.path.exists(metadata_save_path):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_save_path}，请先执行 Proc01Prepare")

        self.steps = config.get('steps') or self.metadata['steps']
        self.train_baseline_model = config.get('train_baseline_model', True)

        self.project_name = config['project_name']
        self.id_col = config['sample']['id_col'] if config.get('sample', {}).get('id_col') else self.metadata['id_col']
        self.sample_table = config['sample']['table']
        self.target_col = config['sample']['target_col']
        self.tw_col = config['sample']['tw_col']
        self.ins_oos_col = config['sample']['ins_oos_col']
        self.tw_period_map = self.metadata['tw_period_map']
        self.dev_tw = self.metadata['dev_tw']
        self.sample_partition = self.metadata['sample_table_partition_type']
        self.bigtable_partition = list(self.metadata['bigtable_partition_type'].values())[0]
        self.bigtable_ds_range = config.get('bigtable_ds_range')

        # 构建特征→宽表的反向映射
        self.fea_table_map = {}
        for table, fea_list in self.table_fea_map.items():
            for fea in fea_list:
                self.fea_table_map[fea] = table

        # 加载特征字典
        feature_dict_path = os.path.join(self.metadata['data_path'], 'feature_dict.feather')
        feature_dict_df = pd.read_feather(feature_dict_path)
        self.feature_comment_map = dict(zip(feature_dict_df['feature_name'], feature_dict_df['feature_comment']))
        self.feature_category_map = dict(zip(feature_dict_df['feature_name'], feature_dict_df['category_name']))

        # 加载merge表信息(如果有)
        d01_merge_save_path = os.path.join(self.metadata['result_path'], 'Proc02SelectD01Merge_merge_info.pkl')
        if os.path.exists(d01_merge_save_path):
            with open(d01_merge_save_path, 'rb') as f:
                d01_merge_info = pickle.load(f)
            self.merge_table_fea_map = d01_merge_info['merge_table_fea_map']
        else:
            self.merge_table_fea_map = None

        # LightGBM参数(支持用户覆盖和Claude动态调整)
        default_lgb_params = resolve_params('DEFAULT_LGB_PARAMS', config)
        self.lgb_params = default_lgb_params.copy()

        # Summary参数
        summary_params = resolve_params('SUMMARY_PARAMS', config)
        self.num_boost_round = summary_params['num_boost_round']
        self.score_num_nbins = summary_params['score_num_nbins']

        # 抽样参数
        self.random_num = summary_params['random_num']
        self.random_seed = summary_params['random_seed'] or np.random.randint(1, 10000)


    def _collect_step_results(self):
        """
        收集 d01-d07 各步骤的筛选结果, 仅收集 self.steps 中包含的步骤
        返回: feature_drop_map {fea: {'step', 'reason', 'detail'}}, remain_features list
        """
        main_logger = get_main_logger()
        main_logger.info("=== 收集各步骤筛选结果 ===")

        feature_drop_map = {}
        cache_path = self.metadata['cache_path']
        result_path = self.metadata['result_path']

        # --- 1. 收集 d01 结果 ---
        if 'd01' in self.steps:
            d01_cache_dir = os.path.join(cache_path, 'Proc02SelectD01')
            if os.path.exists(d01_cache_dir):
                for table in self.table_fea_map.keys():
                    table_rename = '__dot__'.join(table.split('.'))
                    d01_file = os.path.join(d01_cache_dir, f'{table_rename}.pkl')
                    if os.path.exists(d01_file):
                        with open(d01_file, 'rb') as f:
                            round_select_rlt = pickle.load(f)

                        for round_idx, select_rlt_list in round_select_rlt.items():
                            for (toad_drop, all_drop_list, all_remain, remain_iv) in select_rlt_list:
                                for fea in toad_drop.get('empty', []):
                                    if fea not in feature_drop_map:
                                        feature_drop_map[fea] = {
                                            'step': 'd01_toad', 'reason': 'empty',
                                            'detail': '缺失率过高'
                                        }
                                for fea in toad_drop.get('iv', []):
                                    if fea not in feature_drop_map:
                                        feature_drop_map[fea] = {
                                            'step': 'd01_toad', 'reason': 'iv',
                                            'detail': 'IV过低'
                                        }
                                for fea in toad_drop.get('corr', []):
                                    if fea not in feature_drop_map:
                                        feature_drop_map[fea] = {
                                            'step': 'd01_toad', 'reason': 'corr',
                                            'detail': '相关性过高'
                                        }

        # --- 2. 收集 d02 PSI 结果 ---
        if 'd02' in self.steps:
            d02_psi_file = os.path.join(result_path, 'Proc02SelectD02PSI_psi_info.pkl')
            if os.path.exists(d02_psi_file):
                with open(d02_psi_file, 'rb') as f:
                    d02_result = pickle.load(f)

                fea_max_psi = d02_result.get('fea_max_psi', {})
                psi_drop_fea = d02_result.get('psi_drop_fea', [])
                for fea in psi_drop_fea:
                    max_psi = fea_max_psi.get(fea, 0)
                    feature_drop_map[fea] = {
                        'step': 'd02_psi', 'reason': 'psi',
                        'detail': f'最大PSI {max_psi:.4f}'
                    }

        # --- 3. 收集 d03 结果 ---
        d03d06_cache_dir = os.path.join(cache_path, 'Proc02SelectD03D06')
        if 'd03' in self.steps:
            d03_file = os.path.join(d03d06_cache_dir, 'd03_result.pkl')
            if os.path.exists(d03_file):
                with open(d03_file, 'rb') as f:
                    d03_result = pickle.load(f)
                for fea in d03_result.get('all_drop', []):
                    feature_drop_map[fea] = {
                        'step': 'd03_random', 'reason': 'random_importance',
                        'detail': 'split重要性低于随机数'
                    }

        # --- 4. 收集 d04 结果 ---
        if 'd04' in self.steps:
            d04_file = os.path.join(d03d06_cache_dir, 'd04_result.pkl')
            if os.path.exists(d04_file):
                with open(d04_file, 'rb') as f:
                    d04_result = pickle.load(f)
                if d04_result.get('best_th_set') is not None:
                    best_key = d04_result['best_th_set'][0]
                    split_th, gain_th = best_key
                    split_drop = d04_result['th_drop_info'][split_th]['split']
                    gain_drop = d04_result['th_drop_info'][gain_th]['gain']
                    d04_drop = list(set(split_drop + gain_drop))
                    for fea in d04_drop:
                        feature_drop_map[fea] = {
                            'step': 'd04_null_importance', 'reason': 'null_importance',
                            'detail': 'null importance低于阈值'
                        }

        # --- 5. 收集 d05 结果 ---
        if 'd05' in self.steps:
            d05_file = os.path.join(d03d06_cache_dir, 'd05_result.pkl')
            if os.path.exists(d05_file):
                with open(d05_file, 'rb') as f:
                    d05_result = pickle.load(f)
                for fea in d05_result.get('drop_features', []):
                    feature_drop_map[fea] = {
                        'step': 'd05_top_importance', 'reason': 'top_importance',
                        'detail': '累计重要性截断'
                    }

        # --- 6. 收集 d06 结果 ---
        if 'd06' in self.steps:
            d06_file = os.path.join(d03d06_cache_dir, 'd06_result.pkl')
            if os.path.exists(d06_file):
                with open(d06_file, 'rb') as f:
                    d06_result = pickle.load(f)
                for fea in d06_result.get('drop_features', []):
                    feature_drop_map[fea] = {
                        'step': 'd06_shap', 'reason': 'shap',
                        'detail': 'SHAP累计重要性截断'
                    }

        # --- 7. 收集 d07 结果 ---
        if 'd07' in self.steps:
            d07_file = os.path.join(result_path, 'Proc02SelectD07D08_d07_detail.pkl')
            if os.path.exists(d07_file):
                with open(d07_file, 'rb') as f:
                    d07_result = pickle.load(f)

                drop_reasons = d07_result.get('drop_reasons', {})
                for fea, reason in drop_reasons.items():
                    feature_drop_map[fea] = {
                        'step': 'd07_woe_trend', 'reason': 'woe_trend',
                        'detail': reason
                    }

        # --- 8. 获取最终剩余特征: 使用 resolver 动态查找 ---
        # 构造一个虚拟的 "summary" 步骤, 它在所有步骤之后
        # 先尝试 d07-d08 输出, 再 d03-d06 输出, 再表级输出
        remain_features = None

        # 检查 d07-d08 输出
        if {'d07', 'd08'} & set(self.steps):
            remain_file = os.path.join(result_path, 'Proc02SelectD07D08_remain_features.pkl')
            if os.path.exists(remain_file):
                with open(remain_file, 'rb') as f:
                    remain_features = pickle.load(f)

        # 检查 d03-d06 输出
        if remain_features is None and {'d03', 'd04', 'd05', 'd06'} & set(self.steps):
            d06_remain_file = os.path.join(result_path, 'Proc02SelectD03D06_remain_features.pkl')
            if os.path.exists(d06_remain_file):
                with open(d06_remain_file, 'rb') as f:
                    remain_features = pickle.load(f)

        # fallback: 表级输出展平
        if remain_features is None:
            table_remain = resolve_table_remain_fea(
                self.config['project_path'], self.steps, 'summary'
            )
            remain_features = [f for flist in table_remain.values() for f in
                              (list(flist) if not isinstance(flist, list) else flist)]

        main_logger.info(f"收集完成: 剔除特征 {len(feature_drop_map)} 个, 剩余特征 {len(remain_features)} 个")
        return feature_drop_map, remain_features

    def _build_bigtable_stats(self, feature_drop_map, remain_features):
        """
        按宽表维度统计每步的剔除数量, 仅展示 self.steps 中实际执行步骤的列
        """
        # 步骤到列名的映射
        step_to_cols = {
            'd01': ['d01_toad_empty', 'd01_toad_iv', 'd01_toad_corr'],
            'd02': ['d02_psi'],
            'd03': ['d03_random'],
            'd04': ['d04_null_imp'],
            'd05': ['d05_top_imp'],
            'd06': ['d06_shap'],
            'd07': ['d07_woe_trend'],
        }
        # 只保留 self.steps 中对应的列
        step_cols = [col for s in self.steps if s in step_to_cols for col in step_to_cols[s]]

        # 步骤reason到列名的映射
        reason_to_col = {
            ('d01_toad', 'empty'): 'd01_toad_empty',
            ('d01_toad', 'iv'): 'd01_toad_iv',
            ('d01_toad', 'corr'): 'd01_toad_corr',
            ('d02_psi', 'psi'): 'd02_psi',
            ('d03_random', 'random_importance'): 'd03_random',
            ('d04_null_importance', 'null_importance'): 'd04_null_imp',
            ('d05_top_importance', 'top_importance'): 'd05_top_imp',
            ('d06_shap', 'shap'): 'd06_shap',
            ('d07_woe_trend', 'woe_trend'): 'd07_woe_trend',
        }

        remain_set = set(remain_features)
        rows = []
        for table, fea_list in self.table_fea_map.items():
            total = len(fea_list)
            fea_set = set(fea_list)
            remain_cnt = len(fea_set & remain_set)
            drop_cnt = total - remain_cnt

            # 统计各步骤剔除数
            step_counts = {col: 0 for col in step_cols}
            for fea in fea_list:
                if fea in feature_drop_map:
                    info = feature_drop_map[fea]
                    col_key = (info['step'], info['reason'])
                    col_name = reason_to_col.get(col_key)
                    if col_name:
                        step_counts[col_name] += 1

            row = {
                '来源宽表': table,
                '特征总数': total,
                '剔除总数': drop_cnt,
                '剩余总数': remain_cnt,
                '剔除比例': f'{drop_cnt / total * 100:.2f}%' if total > 0 else '0%',
            }
            row.update(step_counts)
            rows.append(row)

        stats_df = pd.DataFrame(rows)
        return stats_df

    def _build_drop_detail(self, feature_drop_map):
        """
        构建特征剔除明细表: 特征名, 来源宽表, 特征中文名, 剔除步骤, 剔除原因, 剔除原因明细
        """
        rows = []
        for fea, info in feature_drop_map.items():
            rows.append({
                '特征名': fea,
                '来源宽表': self.fea_table_map.get(fea, ''),
                '特征中文名': self.feature_comment_map.get(fea, ''),
                '剔除步骤': info['step'],
                '剔除原因': info['reason'],
                '剔除原因明细': info['detail'],
            })

        drop_df = pd.DataFrame(rows)
        # 按步骤排序
        step_order = ['d01_toad', 'd02_psi', 'd03_random', 'd04_null_importance',
                       'd05_top_importance', 'd06_shap', 'd07_woe_trend']
        drop_df['_sort'] = drop_df['剔除步骤'].apply(lambda x: step_order.index(x) if x in step_order else 99)
        drop_df = drop_df.sort_values('_sort').drop(columns=['_sort']).reset_index(drop=True)
        return drop_df

    def _build_drop_funnel(self, feature_drop_map, remain_features, total_features):
        """
        构建筛选步骤漏斗统计表
        """
        # 步骤顺序（按执行顺序）
        step_order = ['d01_toad', 'd02_psi', 'd03_random', 'd04_null_importance',
                      'd05_top_importance', 'd06_shap', 'd07_woe_trend']

        # 统计各步骤剔除的特征数
        step_drop_count = {}
        for fea, info in feature_drop_map.items():
            step = info['step']
            step_drop_count[step] = step_drop_count.get(step, 0) + 1

        # 构建漏斗数据
        rows = []
        cumulative_drop = 0
        remaining = total_features

        # 第一行：原始特征
        rows.append({
            '筛选步骤': '原始特征',
            '步骤后剩余数': total_features,
            '本步剔除数': 0,
            '本步剔除比例': '0.00%',
            '累计剔除数': 0,
            '累计剔除比例': '0.00%',
        })

        # 各步骤
        for step in step_order:
            if step not in self.steps:
                continue
            drop_count = step_drop_count.get(step, 0)
            cumulative_drop += drop_count
            remaining = total_features - cumulative_drop

            rows.append({
                '筛选步骤': step,
                '步骤后剩余数': remaining,
                '本步剔除数': drop_count,
                '本步剔除比例': f'{drop_count / total_features * 100:.2f}%',
                '累计剔除数': cumulative_drop,
                '累计剔除比例': f'{cumulative_drop / total_features * 100:.2f}%',
            })

        # 最后一行：最终剩余
        rows.append({
            '筛选步骤': '最终剩余',
            '步骤后剩余数': len(remain_features),
            '本步剔除数': 0,
            '本步剔除比例': '0.00%',
            '累计剔除数': cumulative_drop,
            '累计剔除比例': f'{cumulative_drop / total_features * 100:.2f}%',
        })

        return pd.DataFrame(rows)

    def _build_step_top_tables(self, feature_drop_map, top_n=5):
        """
        构建各步骤中剔除占比最高的 TOP N 宽表
        """
        step_order = ['d01_toad', 'd02_psi', 'd03_random', 'd04_null_importance',
                      'd05_top_importance', 'd06_shap', 'd07_woe_trend']

        result = {}

        for step in step_order:
            if step not in self.steps:
                continue

            # 统计该步骤中各宽表的剔除数
            table_drop_count = {}
            for fea, info in feature_drop_map.items():
                if info['step'] == step:
                    table = self.fea_table_map.get(fea, '')
                    if table:
                        table_drop_count[table] = table_drop_count.get(table, 0) + 1

            # 构建 DataFrame
            rows = []
            for table, drop_count in table_drop_count.items():
                total_count = len(self.table_fea_map.get(table, []))
                rows.append({
                    '宽表名': table,
                    '剔除数量': drop_count,
                    '宽表总特征数': total_count,
                    '剔除占比': f'{drop_count / total_count * 100:.2f}%' if total_count > 0 else '0.00%',
                })

            # 计算剔除占比（数值），用于排序
            if rows:
                step_df = pd.DataFrame(rows)
                step_df['_sort_ratio'] = step_df.apply(
                    lambda r: r['剔除数量'] / r['宽表总特征数'] if r['宽表总特征数'] > 0 else 0,
                    axis=1
                )
                # 按剔除占比降序排序，取 TOP N
                step_df = step_df.sort_values('_sort_ratio', ascending=False).head(top_n).reset_index(drop=True)
                step_df = step_df.drop(columns=['_sort_ratio'])  # 删除临时排序列
                result[step] = step_df

        return result

    def _build_remain_eval(self, remain_features):
        """
        构建剩余特征评估表: 特征名, 来源宽表, 特征中文名, max_psi, importance, shap, woe解释性
        数据来源: d02(psi), d05(importance), d06(shap), d08(woe解释性)
        """
        result_path = self.metadata['result_path']
        cache_path = self.metadata['cache_path']

        # 加载d05 importance
        importance_map = {}
        d05_file = os.path.join(cache_path, 'Proc02SelectD03D06', 'd05_result.pkl')
        if os.path.exists(d05_file):
            with open(d05_file, 'rb') as f:
                d05_result = pickle.load(f)
            imp_df = d05_result.get('importance_df')
            if imp_df is not None and 'feature' in imp_df.columns:
                importance_map = dict(zip(imp_df['feature'], imp_df.get('importance', imp_df.iloc[:, 1])))

        # 加载d06 shap importance
        shap_map = {}
        d06_file = os.path.join(cache_path, 'Proc02SelectD03D06', 'd06_result.pkl')
        if os.path.exists(d06_file):
            with open(d06_file, 'rb') as f:
                d06_result = pickle.load(f)
            shap_df = d06_result.get('shap_importance_df')
            if shap_df is not None and 'feature' in shap_df.columns:
                shap_map = dict(zip(shap_df['feature'], shap_df.get('shap_importance', shap_df.iloc[:, 1])))

        # 加载d02 PSI
        psi_map = {}
        d02_file = os.path.join(result_path, 'Proc02SelectD02PSI_psi_info.pkl')
        if os.path.exists(d02_file):
            with open(d02_file, 'rb') as f:
                d02_result = pickle.load(f)
            psi_map = d02_result.get('fea_max_psi', {})

        # 加载d08 WOE解释性 (如果有)
        woe_summary_map = {}
        d08_file = os.path.join(result_path, 'Proc02SelectD07D08_d08_summary.pkl')
        if os.path.exists(d08_file):
            with open(d08_file, 'rb') as f:
                d08_result = pickle.load(f)
            woe_summary_list = d08_result.get('woe_summary_list', [])
            for item in woe_summary_list:
                if isinstance(item, dict) and 'feature_name' in item:
                    mono = item.get('monotonicity', '')
                    trend = item.get('trend_direction', '')
                    consistency = item.get('window_consistency', {})
                    consistency_str = ', '.join(f'{k}={v:.2f}' for k, v in consistency.items()) if consistency else ''
                    summary_text = f'单调性:{mono}, 趋势:{trend}'
                    if consistency_str:
                        summary_text += f', 窗口一致性:{consistency_str}'
                    woe_summary_map[item['feature_name']] = summary_text

        has_d08 = len(woe_summary_map) > 0

        rows = []
        for fea in remain_features:
            row = {
                '特征名': fea,
                '来源宽表': self.fea_table_map.get(fea, ''),
                '特征中文名': self.feature_comment_map.get(fea, ''),
                'importance': f'{importance_map.get(fea, 0):.1f}',
                'shap': f'{shap_map.get(fea, 0):.1f}',
                'max_psi': f'{psi_map.get(fea, 0):.4f}',
            }
            if has_d08:
                row['woe解释性'] = woe_summary_map.get(fea, '')
            rows.append(row)

        remain_df = pd.DataFrame(rows)
        return remain_df, has_d08, woe_summary_map

    def _load_all_window_data(self, remain_features):
        """
        加载全窗口数据(DEV+OOT): 通过 SQL JOIN 一次性从多张宽表取数
        支持断点续跑: 检查缓存的 feather 文件
        """
        main_logger = get_main_logger()
        step_start = time.time()

        # 检查缓存文件
        cache_feather_path = os.path.join(self.proc_cache_path, 'full_window_data.feather')
        if os.path.exists(cache_feather_path):
            main_logger.info(f"发现缓存的全窗口数据文件, 直接加载: {cache_feather_path}")
            full_df = pd.read_feather(cache_feather_path)

            # 特殊值替换
            rh_in_data = [col for col in remain_features if col in self.rh_feature_list and col in full_df.columns]
            if rh_in_data:
                for col in rh_in_data:
                    full_df[col] = full_df[col].replace([np.inf, -np.inf, -999, -998], np.nan)

            elapsed = time.time() - step_start
            main_logger.info(f"全窗口数据加载完成(从缓存), shape: {full_df.shape}, 耗时: {elapsed:.1f}s")
            return full_df

        # 缓存不存在, 执行 SQL
        main_logger.info("缓存文件不存在, 开始执行 SQL 加载全窗口数据")

        if self.merge_table_fea_map is not None:
            table_fea_map = self.merge_table_fea_map
        else:
            # 从原宽表取, 只取剩余特征
            table_fea_map = {}
            remain_set = set(remain_features)
            for table, fea_list in self.table_fea_map.items():
                sub = [f for f in fea_list if f in remain_set]
                if sub:
                    table_fea_map[table] = sub

        # 对 merge 表也过滤只取剩余特征
        if self.merge_table_fea_map is not None:
            remain_set = set(remain_features)
            table_fea_map = {
                table: [f for f in (list(fea_list) if not isinstance(fea_list, list) else fea_list) if f in remain_set]
                for table, fea_list in table_fea_map.items()
            }
            table_fea_map = {t: fl for t, fl in table_fea_map.items() if fl}

        # 拼接表分区与样本表一致，原始宽表用自身分区
        if self.merge_table_fea_map is not None:
            bigtable_partition = self.sample_partition
            bigtable_ds_range = None  # merge表不需要ds_range
        else:
            bigtable_partition = self.bigtable_partition
            bigtable_ds_range = self.bigtable_ds_range

        # 生成多表 JOIN SQL (使用公共函数)
        join_sql = cons_join_sql(
            table_fea_map=table_fea_map,
            sample_table=self.sample_table,
            id_col=self.id_col,
            target_col=self.target_col,
            tw_col_or_ins_oos_col=[self.tw_col, self.ins_oos_col],
            dev_tw_filter=None,  # 全窗口不过滤
            random_num=self.random_num,
            random_seed=self.random_seed,
            sample_partition=self.sample_partition,
            bigtable_partition=bigtable_partition,
            rh_feature_list=self.rh_feature_list,
            bigtable_ds_range=bigtable_ds_range,
        )

        # 保存 SQL
        sql_save_path = os.path.join(self.proc_cache_path, 'load_all_window_join.sql')
        with open(sql_save_path, 'w') as f:
            f.write(join_sql)

        # 一次性执行
        main_logger.info(f"通过 SQL JOIN 加载全窗口数据, 涉及 {len(table_fea_map)} 张表")
        client = TMLSQLClient()
        try:
            full_df = safe_sql_execute(client, join_sql, main_logger, desc="summary-全窗口数据")
        finally:
            client.stop()

        # 特殊值替换
        rh_in_data = [col for col in remain_features if col in self.rh_feature_list and col in full_df.columns]
        if rh_in_data:
            for col in rh_in_data:
                full_df[col] = full_df[col].replace([np.inf, -np.inf, -999, -998], np.nan)

        # 保存缓存
        main_logger.info(f"保存全窗口数据到缓存: {cache_feather_path}")
        full_df.to_feather(cache_feather_path)

        elapsed = time.time() - step_start
        main_logger.info(f"全窗口数据加载完成(首次执行), shape: {full_df.shape}, 耗时: {elapsed:.1f}s")
        return full_df

    @capture_print
    def _train_baseline_model(self, full_df, remain_features):
        """
        用最终剩余特征训练LightGBM基线模型, 在各窗口评估KS/AUC
        返回: metrics_df, roc_fig, score_fig
        """
        main_logger = get_main_logger()
        step_start = time.time()
        result_path = self.metadata['result_path']

        # 构建tw标签: DEV窗口→DEV_INS/DEV_OOS, 非DEV窗口→OOT_{period}
        # 按tw_col+ins_oos_col拆分数据集
        tw_col = self.tw_col
        target_col = self.target_col

        # 构建数据集标签
        all_tw_values = sorted(full_df[tw_col].unique())
        dev_tw_set = set(self.dev_tw)

        full_df['_dataset_label'] = full_df.apply(
            lambda r: f"DEV_{r[self.ins_oos_col]}" if r[tw_col] in dev_tw_set else f"OOT_{r[tw_col]}",
            axis=1
        )

        # INS训练集
        ins_mask = full_df['_dataset_label'] == 'DEV_INS'
        ins_df = full_df[ins_mask]

        valid_features = [f for f in remain_features if f in full_df.columns]

        # 训练LightGBM
        train_start = time.time()
        main_logger.info(f"训练基线模型, 特征数: {len(valid_features)}, INS样本: {len(ins_df)}")
        dtrain = lgb.Dataset(ins_df[valid_features], label=ins_df[target_col])
        model = lgb.train(self.lgb_params, dtrain, num_boost_round=self.num_boost_round)
        train_elapsed = time.time() - train_start
        main_logger.info(f"模型训练完成, 耗时: {train_elapsed:.1f}s")

        # 各窗口评估
        eval_start = time.time()
        main_logger.info(f"开始评估各窗口效果")
        dataset_labels = sorted(full_df['_dataset_label'].unique())
        metrics_rows = []
        pred_prob_dict = {}
        y_true_dict = {}

        for label in dataset_labels:
            sub_df = full_df[full_df['_dataset_label'] == label]
            if len(sub_df) == 0 or sub_df[target_col].nunique() < 2:
                continue
            pred = model.predict(sub_df[valid_features])
            ks_val, auc_val, _ = ks_auc(pred, sub_df[target_col].values)
            metrics_rows.append({
                '样本窗口': label,
                '样本量': len(sub_df),
                '样本浓度': f'{sub_df[target_col].mean() * 100:.2f}%',
                'KS': f'{ks_val:.4f}',
                'AUC': f'{auc_val:.4f}',
            })
            pred_prob_dict[label] = pred
            y_true_dict[label] = sub_df[target_col].values

        metrics_df = pd.DataFrame(metrics_rows)
        eval_elapsed = time.time() - eval_start
        main_logger.info(f"各窗口评估完成, 耗时: {eval_elapsed:.1f}s")

        # ROC曲线
        roc_fig = plot_roc_multi(y_true_dict, pred_prob_dict, title=self.project_name)
        roc_path = os.path.join(result_path, 'roc_curve.png')
        roc_fig.savefig(roc_path, dpi=150, bbox_inches='tight')
        main_logger.info(f"ROC曲线保存到: {roc_path}")

        # 分组逾期率 (用模型得分作为特征分箱)
        full_df['model_score'] = model.predict(full_df[valid_features])
        score_stt, score_fig = split_plot_feature(
            df=full_df, feature_list=['model_score'],
            tw_col='_dataset_label', dev_tw='DEV_INS',
            tgt_col=target_col, num_nbins=self.score_num_nbins
        )
        if 'model_score' in score_fig:
            score_fig_obj = score_fig['model_score']
            score_path = os.path.join(result_path, 'score_distribution.png')
            score_fig_obj.savefig(score_path, dpi=150, bbox_inches='tight')
            main_logger.info(f"分组逾期率图保存到: {score_path}")

        # 特征重要性Top20
        importance_df = pd.DataFrame({
            'feature': valid_features,
            'importance': model.feature_importance(importance_type='gain'),
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        full_df.drop(columns=['_dataset_label', 'model_score'], inplace=True, errors='ignore')

        total_elapsed = time.time() - step_start
        main_logger.info(f"基线模型训练和评估完成, 总耗时: {total_elapsed:.1f}s")

        return metrics_df, importance_df, model

    def _gen_report_md(self, report_data):
        """
        使用固化的md模板生成报告文件
        """
        main_logger = get_main_logger()
        result_path = self.metadata['result_path']

        # === 第一部分: Summary ===
        md_lines = []
        md_lines.append('# 特征筛选报告\n')
        md_lines.append('---\n')
        md_lines.append('## 第一部分: Summary\n')
        md_lines.append('### 配置参数说明\n')
        md_lines.append(f'- **项目名称**: {self.project_name}')
        md_lines.append(f'- **样本描述**:')
        md_lines.append(f'  - 样本表名: {self.sample_table}')
        id_col_str = ', '.join(self.id_col) if isinstance(self.id_col, list) else self.id_col
        md_lines.append(f'  - 主键: {id_col_str}')
        md_lines.append(f'  - Y标签: {self.target_col}')
        md_lines.append(f'  - 时间窗口划分列: {self.tw_col}')
        md_lines.append(f'- **特征宽表列表**:\n')

        # 宽表列表
        md_lines.append('| 宽表名 | 特征数 |')
        md_lines.append('|--------|--------|')
        for table, fea_list in self.table_fea_map.items():
            md_lines.append(f'| {table} | {len(fea_list)} |')

        md_lines.append('')
        thresholds = self.config['thresholds']
        md_lines.append(f'- **筛选参数**:')
        md_lines.append(f'  - IV 阈值: {thresholds.get("iv", "")}')
        md_lines.append(f'  - 缺失率阈值: {thresholds.get("empty", "")}')
        md_lines.append(f'  - 相关性阈值: {thresholds.get("corr", "")}')
        md_lines.append(f'  - PSI 阈值: {thresholds.get("psi", "")}')
        md_lines.append(f'- **项目路径**: {self.config["project_path"]}')
        md_lines.append('')

        # 文字总结
        md_lines.append('### 筛选结果概览\n')
        total_features = report_data['total_features']
        remain_count = report_data['remain_count']
        remove_ratio = (total_features - remain_count) / total_features * 100 if total_features > 0 else 0
        top_source = report_data.get('top_source', '')
        top_source_ratio = report_data.get('top_source_ratio', 0)
        md_lines.append(
            f'**文字总结**: 原始特征总数 {total_features} 个，经过多轮筛选后最终剩余 {remain_count} 个，'
            f'剔除比例 {remove_ratio:.2f}%。其中 {top_source} 的特征效果较好，'
            f'在剩余特征中占比较高，达到 {top_source_ratio:.2f}%。'
        )
        md_lines.append('')

        # 宽表维度统计表
        md_lines.append('**宽表维度统计表**:\n')
        stats_df = report_data['bigtable_stats']
        md_lines.append(self._df_to_md_table(stats_df))
        md_lines.append('')

        # === 第二部分: 特征剔除分析 ===
        md_lines.append('---\n')
        md_lines.append('## 第二部分: 特征剔除分析\n')

        # 2.1 筛选步骤漏斗
        md_lines.append('### 2.1 筛选步骤漏斗\n')
        drop_funnel = report_data['drop_funnel']
        md_lines.append(self._df_to_md_table(drop_funnel))
        md_lines.append('')

        # 2.2 各步骤 TOP 宽表
        md_lines.append('### 2.2 各步骤剔除 TOP5 宽表\n')
        step_top_tables = report_data['step_top_tables']
        if step_top_tables:
            for step, top_df in step_top_tables.items():
                step_name_map = {
                    'd01_toad': 'd01 TOAD初筛',
                    'd02_psi': 'd02 PSI稳定性',
                    'd03_random': 'd03 随机数重要性',
                    'd04_null_importance': 'd04 Null Importance',
                    'd05_top_importance': 'd05 TOP重要性',
                    'd06_shap': 'd06 SHAP重要性',
                    'd07_woe_trend': 'd07 WOE趋势',
                }
                md_lines.append(f'#### {step_name_map.get(step, step)}\n')
                md_lines.append(self._df_to_md_table(top_df))
                md_lines.append('')
        else:
            md_lines.append('无特征被剔除。')
        md_lines.append('')

        # === 第三部分: 剩余特征评估 ===
        md_lines.append('---\n')
        md_lines.append('## 第三部分: 剩余特征评估\n')
        if report_data['has_d08']:
            md_lines.append('> **注意**: woe解释性 列基于 d08（WOE 解释性生成）步骤的结果。\n')
        remain_df = report_data['remain_eval']
        if len(remain_df) > 0:
            md_lines.append(self._df_to_md_table(remain_df))
        else:
            md_lines.append('无剩余特征。')
        md_lines.append('')

        # === 第四部分: 基准模型效果 ===
        md_lines.append('---\n')
        md_lines.append('## 第四部分: 基准模型效果\n')

        if report_data.get('train_baseline_model', True):
            md_lines.append('### 训练参数\n')
            md_lines.append('```python')
            md_lines.append(f'params_dict = {self.lgb_params}')
            md_lines.append(f'num_boost_round = {self.num_boost_round}')
            md_lines.append(f'特征数 = {report_data["num_features"]}')
            md_lines.append('```\n')

            md_lines.append('### 基础效果表格\n')
            metrics_df = report_data['metrics']
            if metrics_df is not None and len(metrics_df) > 0:
                md_lines.append(self._df_to_md_table(metrics_df))
            md_lines.append('')

            md_lines.append('### ROC曲线\n')
            md_lines.append('![ROC曲线](roc_curve.png)\n')

            md_lines.append('### 分组逾期率\n')
            md_lines.append('![分组逾期率](score_distribution.png)\n')

            # 特征重要性Top20
            importance_df = report_data.get('importance_top20')
            if importance_df is not None and len(importance_df) > 0:
                md_lines.append('### 特征重要性 Top20\n')
                top20 = importance_df.head(20).copy()
                top20['特征中文名'] = top20['feature'].map(self.feature_comment_map).fillna('')
                top20 = top20.rename(columns={'feature': '特征名', 'importance': '重要性(gain)'})
                top20['重要性(gain)'] = top20['重要性(gain)'].apply(lambda x: f'{x:.1f}')
                md_lines.append(self._df_to_md_table(top20[['特征名', '特征中文名', '重要性(gain)']]))
            md_lines.append('')
        else:
            md_lines.append('> 本次运行未训练基线模型（train_baseline_model=False），如需查看模型效果请开启该配置后重新运行。\n')
            md_lines.append('')

        # 写入文件
        report_path = os.path.join(result_path, '特征筛选报告.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))

        main_logger.info(f"报告已保存到: {report_path}")

        # 同时保存report_data供CC后续使用
        report_data_path = os.path.join(result_path, 'Proc03Summary_report_data.pkl')
        safe_pickle_dump(report_data, report_data_path)

        return report_path

    @staticmethod
    def _df_to_md_table(df):
        """将DataFrame转为markdown表格字符串"""
        cols = df.columns.tolist()
        header = '| ' + ' | '.join(str(c) for c in cols) + ' |'
        sep = '|' + '|'.join(['------' for _ in cols]) + '|'
        rows = []
        for _, row in df.iterrows():
            rows.append('| ' + ' | '.join(str(row[c]) for c in cols) + ' |')
        return '\n'.join([header, sep] + rows)

    def run(self):
        main_logger = get_main_logger()
        main_logger.info("开始执行Proc03Summary流程")
        run_start = time.time()
        main_logger.info(f"[参数] train_baseline_model: {self.train_baseline_model}")

        # 1. 收集各步骤筛选结果
        step_start = time.time()
        feature_drop_map, remain_features = self._collect_step_results()
        elapsed = time.time() - step_start
        main_logger.info(f"收集步骤结果完成, 耗时: {elapsed:.1f}s")

        # 2. 构建宽表维度统计
        step_start = time.time()
        bigtable_stats = self._build_bigtable_stats(feature_drop_map, remain_features)
        main_logger.info(f"宽表维度统计:\n{bigtable_stats.to_string()}")
        elapsed = time.time() - step_start
        main_logger.info(f"宽表维度统计完成, 耗时: {elapsed:.1f}s")

        # 3. 构建特征剔除明细（保存为pkl，不展示在报告中）
        step_start = time.time()
        drop_detail = self._build_drop_detail(feature_drop_map)
        drop_detail_pkl_path = os.path.join(self.metadata['result_path'], 'Proc03Summary_drop_detail.pkl')
        safe_pickle_dump(drop_detail, drop_detail_pkl_path)
        main_logger.info(f"特征剔除明细: {len(drop_detail)} 条，已保存到 {drop_detail_pkl_path}")
        elapsed = time.time() - step_start
        main_logger.info(f"特征剔除明细构建完成, 耗时: {elapsed:.1f}s")

        # 3.1 构建筛选漏斗统计
        step_start = time.time()
        total_features = sum(len(v) for v in self.table_fea_map.values())
        drop_funnel = self._build_drop_funnel(feature_drop_map, remain_features, total_features)
        main_logger.info(f"筛选漏斗统计:\n{drop_funnel.to_string()}")
        elapsed = time.time() - step_start
        main_logger.info(f"筛选漏斗统计完成, 耗时: {elapsed:.1f}s")

        # 3.2 构建各步骤 TOP 宽表
        step_start = time.time()
        step_top_tables = self._build_step_top_tables(feature_drop_map, top_n=5)
        main_logger.info(f"各步骤 TOP 宽表统计完成, 涉及 {len(step_top_tables)} 个步骤")
        elapsed = time.time() - step_start
        main_logger.info(f"各步骤 TOP 宽表统计完成, 耗时: {elapsed:.1f}s")

        # 4. 构建剩余特征评估
        step_start = time.time()
        remain_eval, has_d08, woe_summary_map = self._build_remain_eval(remain_features)
        main_logger.info(f"剩余特征评估: {len(remain_eval)} 条, 含WOE解释性: {has_d08}")
        elapsed = time.time() - step_start
        main_logger.info(f"剩余特征评估构建完成, 耗时: {elapsed:.1f}s")

        # 5. 加载全窗口数据, 训练基线模型(可选)
        metrics_df = None
        importance_df = None
        if self.train_baseline_model:
            step_start = time.time()
            full_df = self._load_all_window_data(remain_features)
            load_elapsed = time.time() - step_start
            main_logger.info(f"全窗口数据加载完成, 耗时: {load_elapsed:.1f}s")

            metrics_df, importance_df, model = self._train_baseline_model(full_df, remain_features)
            main_logger.info(f"基线模型效果:\n{metrics_df.to_string()}")
        else:
            main_logger.info("train_baseline_model=False, 跳过基线模型训练")

        # 6. 计算top_source
        step_start = time.time()
        remain_set = set(remain_features)
        table_remain_counts = {}
        for table, fea_list in self.table_fea_map.items():
            cnt = len(set(fea_list) & remain_set)
            if cnt > 0:
                table_remain_counts[table] = cnt
        total_remain = len(remain_features)
        if table_remain_counts:
            top_source = max(table_remain_counts, key=table_remain_counts.get)
            top_source_ratio = table_remain_counts[top_source] / total_remain * 100 if total_remain > 0 else 0
        else:
            top_source = ''
            top_source_ratio = 0
        elapsed = time.time() - step_start
        main_logger.info(f"top_source 计算完成, 耗时: {elapsed:.1f}s")

        # 7. 汇总report_data
        step_start = time.time()
        total_features = sum(len(v) for v in self.table_fea_map.values())
        report_data = {
            'total_features': total_features,
            'remain_count': len(remain_features),
            'remain_features': remain_features,
            'top_source': top_source,
            'top_source_ratio': top_source_ratio,
            'bigtable_stats': bigtable_stats,
            'drop_detail': drop_detail,  # 不再展示在报告中，仅供调试
            'drop_funnel': drop_funnel,  # 新增
            'step_top_tables': step_top_tables,  # 新增
            'remain_eval': remain_eval,
            'has_d08': has_d08,
            'woe_summary_map': woe_summary_map,
            'metrics': metrics_df,
            'num_features': len(remain_features),
            'importance_top20': importance_df,
            'feature_drop_map': feature_drop_map,
            'train_baseline_model': self.train_baseline_model,
        }
        elapsed = time.time() - step_start
        main_logger.info(f"report_data 汇总完成, 耗时: {elapsed:.1f}s")

        # 8. 生成报告
        step_start = time.time()
        report_path = self._gen_report_md(report_data)
        elapsed = time.time() - step_start
        main_logger.info(f"报告生成生成, 耗时: {elapsed:.1f}s")

        total_elapsed = time.time() - run_start
        main_logger.info(f"*** Proc03Summary 完成 ***")
        main_logger.info(f"初始特征: {total_features}, 最终剩余: {len(remain_features)}")
        main_logger.info(f"报告路径: {report_path}")
        main_logger.info(f"Proc03Summary 总耗时: {total_elapsed:.1f}s")
