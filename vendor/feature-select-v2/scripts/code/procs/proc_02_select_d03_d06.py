# 重要性相关的筛选步骤, 用的都是DEV数据, 可以合并在一个脚本中
import os
import gc
import time
import pickle
import pandas as pd
import numpy as np
from tmlpatch.database import TMLSQLClient
from procs.base_proc import BaseProc
from utils.data_utility import str_format, safe_sql_execute, safe_pickle_dump, cons_join_sql
from utils.log_config import get_main_logger
from utils.feature_select import (
    gen_data_iter,
    replace_special_values,
    d03_random_importance_select,
    NullImportanceScore,
    d04_select_by_null_importance,
    d05_select_by_top_importance,
    d06_select_by_shap,
)
from utils.decorators import capture_print
from utils.remain_resolver import resolve_table_remain_fea, resolve_merge_table_info
from utils.param_manager import resolve_params



class Proc02SelectD03D06(BaseProc):
    '''
    d03-d06筛选流程:
    1. 加载DEV数据(从merge表或原宽表)
    2. 按ins_oos_col拆分INS/OOS
    3. 顺序执行d03→d04→d05→d06, 每步输出作为下一步输入
    4. 每步结果保存checkpoint, 支持断点续跑
    '''

    PROC_CACHE_NAME = 'Proc02SelectD03D06'

    def __init__(self, config):
        super().__init__(config)

        # 校验 metadata 是否存在
        metadata_save_path = os.path.join(config['project_path'], 'data', 'metadata.pkl')
        if not os.path.exists(metadata_save_path):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_save_path}，请先执行 Proc01Prepare")

        self.project_name = config['project_name']
        self.id_col = config['sample']['id_col'] if config.get('sample', {}).get('id_col') else self.metadata['id_col']
        self.sample_table = config['sample']['table']
        self.target_col = config['sample']['target_col']
        self.tw_col = config['sample']['tw_col']
        self.ins_oos_col = config['sample']['ins_oos_col']
        self.dev_tw = self.metadata['dev_tw']
        self.sample_partition = self.metadata['sample_table_partition_type']
        self.bigtable_partition = list(self.metadata['bigtable_partition_type'].values())[0]
        self.bigtable_ds_range = config.get('bigtable_ds_range')

        # 加载上一个实际执行步骤的剩余特征(表级字典)
        self.steps = config.get('steps') or self.metadata['steps']
        self.table_remain_fea = resolve_table_remain_fea(config['project_path'], self.steps, 'd03')

        # 加载merge表信息(如果d01执行过)
        merge_info = resolve_merge_table_info(config['project_path'], self.steps)
        self.merge_table_fea_map = merge_info['merge_table_fea_map'] if merge_info else None

        # LightGBM 参数(筛选专用: 不做行列采样, 确保特征重要性评估准确)
        screening_lgb_params = resolve_params('SCREENING_LGB_PARAMS', config)
        self.lgb_params = screening_lgb_params.copy()

        # d03-d06 参数(支持用户覆盖和Claude动态调整)
        d03_d06_params = resolve_params('D03_D06_PARAMS', config)
        self.num_boost_round = d03_d06_params['num_boost_round']

        # d03 参数
        self.d03_bagging_round = d03_d06_params['d03_bagging_round']
        self.d03_bagging_fraction = d03_d06_params['d03_bagging_fraction']
        self.d03_thresholds = d03_d06_params['d03_thresholds']

        # d04 参数
        self.d04_real_round = d03_d06_params['d04_real_round']
        self.d04_null_round = d03_d06_params['d04_null_round']
        self.d04_thresholds_list = d03_d06_params['d04_thresholds_list']

        # d05 参数
        self.d05_thresholds_list = d03_d06_params['d05_thresholds_list']

        # d06 参���
        self.d06_thresholds_list = d03_d06_params['d06_thresholds_list']

        # 抽样参数
        self.ins_random_num = d03_d06_params['ins_random_num']
        self.oos_random_num = d03_d06_params['oos_random_num']
        self.random_seed = d03_d06_params['random_seed'] or np.random.randint(1, 10000)

    def _execute_sql_load_split(self, dataset, random_num):
        """
        执行 SQL 加载 DEV 的 INS 或 OOS 数据
        :param dataset: str, 'INS' 或 'OOS'
        :param random_num: int, 抽样数
        :return: (df, all_fea_list)
        """
        main_logger = get_main_logger()

        # 确定取数表和特征列表
        if self.merge_table_fea_map is not None:
            # merge 表包含 d01 的全部特征，需用 table_remain_fea 过滤出上一步剩余特征
            remain_set = set(f for flist in self.table_remain_fea.values() for f in flist)
            table_fea_map = {
                table: [f for f in fea_list if f in remain_set]
                for table, fea_list in self.merge_table_fea_map.items()
            }
            table_fea_map = {t: fl for t, fl in table_fea_map.items() if fl}
            bigtable_partition = self.sample_partition
            bigtable_ds_range = None  # merge表不需要ds_range
        else:
            table_fea_map = self.table_remain_fea
            bigtable_partition = self.bigtable_partition
            bigtable_ds_range = self.bigtable_ds_range

        # 生成 SQL，增加 INS/OOS 过滤条件
        extra_where = f'and {self.ins_oos_col} = "{dataset}"'
        join_sql = cons_join_sql(
            table_fea_map=table_fea_map,
            sample_table=self.sample_table,
            id_col=self.id_col,
            target_col=self.target_col,
            tw_col_or_ins_oos_col=[self.ins_oos_col],
            dev_tw_filter=(self.tw_col, self.dev_tw),
            random_num=random_num,
            random_seed=self.random_seed,
            sample_partition=self.sample_partition,
            bigtable_partition=bigtable_partition,
            rh_feature_list=self.rh_feature_list,
            extra_where=extra_where,
            bigtable_ds_range=bigtable_ds_range,
        )

        # 保存 SQL
        sql_save_path = os.path.join(self.proc_cache_path, f'load_dev_{dataset.lower()}_join.sql')
        with open(sql_save_path, 'w') as f:
            f.write(join_sql)

        # 执行
        main_logger.info(f"加载 DEV-{dataset} 数据, 涉及 {len(table_fea_map)} 张表, 抽样上限: {random_num}")
        client = TMLSQLClient()
        try:
            df = safe_sql_execute(client, join_sql, main_logger, desc=f"d03_d06-加载{dataset}数据", project='dw_backdate')
        finally:
            client.stop()

        # 特征列表
        all_fea_list = [fea for fea_list in table_fea_map.values()
                        for fea in (list(fea_list) if not isinstance(fea_list, list) else fea_list)]

        # 特殊值替换
        rh_fea_in_data = [col for col in all_fea_list if col in self.rh_feature_list and col in df.columns]
        if rh_fea_in_data:
            for col in rh_fea_in_data:
                df[col] = df[col].replace([np.inf, -np.inf, -999, -998], np.nan)

        main_logger.info(f"{dataset} shape: {df.shape}, bad_rate: {df[self.target_col].mean():.4f}")
        return df, all_fea_list

    def _prepare_dev_data(self):
        """
        准备 DEV 数据: 分别 SQL 取 INS 和 OOS, 缓存为 feather 文件
        返回: all_fea_list (全部特征列表)
        """
        main_logger = get_main_logger()
        ins_path = os.path.join(self.proc_cache_path, 'ins_df.feather')
        oos_path = os.path.join(self.proc_cache_path, 'oos_df.feather')

        if not os.path.exists(ins_path):
            ins_df, _ = self._execute_sql_load_split('INS', self.ins_random_num)
            ins_df.to_feather(ins_path)
            main_logger.info(f"已缓存 INS({ins_df.shape}) 到 feather")
            del ins_df
            gc.collect()
        else:
            main_logger.info("发现 INS feather 缓存, 跳过 SQL")

        if not os.path.exists(oos_path):
            oos_df, _ = self._execute_sql_load_split('OOS', self.oos_random_num)
            oos_df.to_feather(oos_path)
            main_logger.info(f"已缓存 OOS({oos_df.shape}) 到 feather")
            del oos_df
            gc.collect()
        else:
            main_logger.info("发现 OOS feather 缓存, 跳过 SQL")

        # 从 table_remain_fea 获取全部剩余特征列表
        all_fea_list = [f for flist in self.table_remain_fea.values() for f in flist]
        main_logger.info(f"特征数: {len(all_fea_list)}")

        return all_fea_list

    def _read_data(self, model_features, dataset='both'):
        """
        从 feather 缓存按需读取指定列, 每步只加载当前需要的特征
        :param model_features: list, 当前步骤需要的特征列表
        :param dataset: str, 'ins'/'oos'/'both'
        :return: (ins_df, oos_df), 不需要的返回 None
        """
        base_cols = [self.target_col]
        read_cols = base_cols + model_features

        ins_path = os.path.join(self.proc_cache_path, 'ins_df.feather')
        oos_path = os.path.join(self.proc_cache_path, 'oos_df.feather')

        ins_df = None
        oos_df = None
        if dataset in ('ins', 'both'):
            ins_df = pd.read_feather(ins_path, columns=read_cols)
        if dataset in ('oos', 'both'):
            oos_df = pd.read_feather(oos_path, columns=read_cols)

        return ins_df, oos_df

    @capture_print
    def _run_d03(self, ins_df, model_features):
        """d03: 随机重要性筛选"""
        main_logger = get_main_logger()
        checkpoint_path = os.path.join(self.proc_cache_path, 'd03_result.pkl')
        if os.path.exists(checkpoint_path):
            main_logger.info("d03 已有checkpoint, 跳过")
            with open(checkpoint_path, 'rb') as f:
                result = pickle.load(f)
            return result['all_drop']

        step_start = time.time()
        main_logger.info(f"=== d03 随机重要性筛选, 输入特征数: {len(model_features)} ===")

        # 添加随机数列
        random_col = 'random_col'
        ins_df[random_col] = np.random.randint(1, 11, len(ins_df))

        # 生成数据迭代器
        data_iter = gen_data_iter(ins_df, round_num=self.d03_bagging_round, bagging_fraction=self.d03_bagging_fraction)

        round_select_rlt, all_drop = d03_random_importance_select(
            data_iter=data_iter,
            model_features=model_features,
            target=self.target_col,
            random_col=random_col,
            params_dict=self.lgb_params,
            thresholds=self.d03_thresholds,
            num_boost_round=self.num_boost_round,
        )

        # 移除随机数列
        if random_col in ins_df.columns:
            ins_df.drop(columns=[random_col], inplace=True)
        all_drop = [f for f in all_drop if f != random_col]

        elapsed = time.time() - step_start
        main_logger.info(f"d03 结果: 剔除 {len(all_drop)} 个特征, 剩余 {len(model_features) - len(all_drop)} 个, 耗时: {elapsed:.1f}s")

        # 保存checkpoint
        safe_pickle_dump({'round_select_rlt': round_select_rlt, 'all_drop': all_drop}, checkpoint_path)

        return all_drop

    @capture_print
    def _run_d04(self, ins_df, oos_df, model_features):
        """d04: Null Importance 筛选"""
        main_logger = get_main_logger()
        checkpoint_path = os.path.join(self.proc_cache_path, 'd04_result.pkl')
        if os.path.exists(checkpoint_path):
            main_logger.info("d04 已有checkpoint, 跳过")
            with open(checkpoint_path, 'rb') as f:
                result = pickle.load(f)
            if result['best_th_set'] is not None:
                # 从best_th_set中提取剔除特征
                best_key = result['best_th_set'][0]
                split_th, gain_th = best_key
                split_drop = result['th_drop_info'][split_th]['split']
                gain_drop = result['th_drop_info'][gain_th]['gain']
                all_drop = list(set(split_drop + gain_drop))
            else:
                all_drop = []
            return all_drop

        step_start = time.time()
        main_logger.info(f"=== d04 Null Importance 筛选, 输入特征数: {len(model_features)} ===")

        # 计算null importance score
        null_imp_scorer = NullImportanceScore(
            ins_df=ins_df,
            model_features=model_features,
            target_col=self.target_col,
            params_dict=self.lgb_params,
            num_boost_round=self.num_boost_round,
        )
        null_imp_scorer.call_real_importance(round=self.d04_real_round)
        null_imp_scorer.call_null_importance(round=self.d04_null_round)
        null_importance_score = null_imp_scorer.get_score(percent=75)
        del null_imp_scorer
        gc.collect()

        # 选择最优组合
        th_drop_info, split_gain_set_auc, best_th_set = d04_select_by_null_importance(
            ins_df=ins_df,
            oos_df=oos_df,
            model_features=model_features,
            target=self.target_col,
            null_importance_score=null_importance_score,
            thresholds_list=self.d04_thresholds_list,
            params_dict=self.lgb_params,
            num_boost_round=self.num_boost_round,
        )

        # 提取剔除特征
        if best_th_set is not None:
            best_key = best_th_set[0]
            split_th, gain_th = best_key
            split_drop = th_drop_info[split_th]['split']
            gain_drop = th_drop_info[gain_th]['gain']
            all_drop = list(set(split_drop + gain_drop))
        else:
            all_drop = []

        elapsed = time.time() - step_start
        main_logger.info(f"d04 结果: 剔除 {len(all_drop)} 个特征, 剩余 {len(model_features) - len(all_drop)} 个, 耗时: {elapsed:.1f}s")

        # 保存checkpoint
        safe_pickle_dump({
            'th_drop_info': th_drop_info,
            'split_gain_set_auc': split_gain_set_auc,
            'best_th_set': best_th_set,
            'null_importance_score': null_importance_score,
        }, checkpoint_path)

        return all_drop

    @capture_print
    def _run_d05(self, ins_df, oos_df, model_features):
        """d05: Top Importance 筛选"""
        main_logger = get_main_logger()
        checkpoint_path = os.path.join(self.proc_cache_path, 'd05_result.pkl')
        if os.path.exists(checkpoint_path):
            main_logger.info("d05 已有checkpoint, 跳过")
            with open(checkpoint_path, 'rb') as f:
                result = pickle.load(f)
            return result['drop_features']

        step_start = time.time()
        main_logger.info(f"=== d05 Top Importance 筛选, 输入特征数: {len(model_features)} ===")

        importance_df, candidate_auc, best_set, drop_features = d05_select_by_top_importance(
            ins_df=ins_df,
            oos_df=oos_df,
            model_features=model_features,
            target=self.target_col,
            params_dict=self.lgb_params,
            thresholds_list=self.d05_thresholds_list,
            num_boost_round=self.num_boost_round,
        )

        elapsed = time.time() - step_start
        main_logger.info(f"d05 结果: 剔除 {len(drop_features)} 个特征, 剩余 {len(model_features) - len(drop_features)} 个, 耗时: {elapsed:.1f}s")

        # 保存checkpoint
        safe_pickle_dump({
            'importance_df': importance_df,
            'candidate_auc': candidate_auc,
            'best_set': best_set,
            'drop_features': drop_features,
        }, checkpoint_path)

        return drop_features

    @capture_print
    def _run_d06(self, ins_df, oos_df, model_features):
        """d06: SHAP 筛选"""
        main_logger = get_main_logger()
        checkpoint_path = os.path.join(self.proc_cache_path, 'd06_result.pkl')
        if os.path.exists(checkpoint_path):
            main_logger.info("d06 已有checkpoint, 跳过")
            with open(checkpoint_path, 'rb') as f:
                result = pickle.load(f)
            return result['drop_features']

        step_start = time.time()
        main_logger.info(f"=== d06 SHAP 筛选, 输入特征数: {len(model_features)} ===")

        shap_importance_df, candidate_auc, best_set, drop_features = d06_select_by_shap(
            ins_df=ins_df,
            oos_df=oos_df,
            model_features=model_features,
            target=self.target_col,
            params_dict=self.lgb_params,
            thresholds_list=self.d06_thresholds_list,
            num_boost_round=self.num_boost_round,
        )

        elapsed = time.time() - step_start
        main_logger.info(f"d06 结果: 剔除 {len(drop_features)} 个特征, 剩余 {len(model_features) - len(drop_features)} 个, 耗时: {elapsed:.1f}s")

        # 保存checkpoint
        safe_pickle_dump({
            'shap_importance_df': shap_importance_df,
            'candidate_auc': candidate_auc,
            'best_set': best_set,
            'drop_features': drop_features,
        }, checkpoint_path)

        return drop_features

    def run(self):
        main_logger = get_main_logger()
        main_logger.info("开始执行Proc02SelectD03D06流程")
        run_start = time.time()

        # 1. 准备数据（首次 SQL 取数 + 缓存 feather，续跑直接跳过 SQL）
        all_fea_list = self._prepare_dev_data()
        model_features = [f for f in all_fea_list]

        d03_drop = []
        d04_drop = []
        d05_drop = []
        d06_drop = []

        # 2. d03: 随机重要性筛选（只需 INS）
        if 'd03' in self.steps:
            ins_df, _ = self._read_data(model_features, dataset='ins')
            main_logger.info(f"d03 加载 INS: {ins_df.shape}")
            d03_drop = self._run_d03(ins_df, model_features)
            model_features = [f for f in model_features if f not in d03_drop]
            main_logger.info(f"d03后剩余特征: {len(model_features)}")
            del ins_df
            gc.collect()
        else:
            main_logger.info("d03 不在 steps 中, 跳过")

        # 3. d04: Null Importance 筛选（需要 INS + OOS，特征已缩减）
        if 'd04' in self.steps:
            ins_df, oos_df = self._read_data(model_features, dataset='both')
            main_logger.info(f"d04 加载 INS: {ins_df.shape}, OOS: {oos_df.shape}")
            d04_drop = self._run_d04(ins_df, oos_df, model_features)
            model_features = [f for f in model_features if f not in d04_drop]
            main_logger.info(f"d04后剩余特征: {len(model_features)}")
            del ins_df, oos_df
            gc.collect()
        else:
            main_logger.info("d04 不在 steps 中, 跳过")

        # 4. d05: Top Importance 筛选
        if 'd05' in self.steps:
            ins_df, oos_df = self._read_data(model_features, dataset='both')
            main_logger.info(f"d05 加载 INS: {ins_df.shape}, OOS: {oos_df.shape}")
            d05_drop = self._run_d05(ins_df, oos_df, model_features)
            model_features = [f for f in model_features if f not in d05_drop]
            main_logger.info(f"d05后剩余特征: {len(model_features)}")
            del ins_df, oos_df
            gc.collect()
        else:
            main_logger.info("d05 不在 steps 中, 跳过")

        # 5. d06: SHAP 筛选
        if 'd06' in self.steps:
            ins_df, oos_df = self._read_data(model_features, dataset='both')
            main_logger.info(f"d06 加载 INS: {ins_df.shape}, OOS: {oos_df.shape}")
            d06_drop = self._run_d06(ins_df, oos_df, model_features)
            model_features = [f for f in model_features if f not in d06_drop]
            main_logger.info(f"d06后剩余特征: {len(model_features)}")
            del ins_df, oos_df
            gc.collect()
        else:
            main_logger.info("d06 不在 steps 中, 跳过")

        # 6. 保存最终结果
        remain_features = model_features
        remain_save_path = os.path.join(self.metadata['result_path'], 'Proc02SelectD03D06_remain_features.pkl')
        safe_pickle_dump(remain_features, remain_save_path)

        total_elapsed = time.time() - run_start
        main_logger.info(f"*** Proc02SelectD03D06 完成 ***")
        main_logger.info(f"初始特征: {len(all_fea_list)}, d03剔除: {len(d03_drop)}, d04剔除: {len(d04_drop)}, "
                        f"d05剔除: {len(d05_drop)}, d06剔除: {len(d06_drop)}, 最终剩余: {len(remain_features)}")
        main_logger.info(f"Proc02SelectD03D06 总耗时: {total_elapsed:.1f}s")
