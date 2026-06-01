# WOE筛选步骤: d07趋势稳定性筛选 + d08解释性摘要生成
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
from utils.binner import split_plot_feature, split_plot_to_excel
from utils.feature_select import (
    d07_select_by_woe_trend,
    d08_select_by_woe_explain,
)
from utils.remain_resolver import resolve_flat_remain_features, resolve_merge_table_info
from utils.param_manager import resolve_params
from utils.decorators import capture_print



class Proc02SelectD07D08(BaseProc):
    '''
    d07-d08 WOE筛选流程:
    1. 加载全窗口数据(DEV+OOT)
    2. 对剩余特征调用split_plot_feature生成WOE分箱数据
    3. d07: WOE趋势稳定性筛选
    4. d08: WOE解释性摘要生成(供AI补充)
    '''

    PROC_CACHE_NAME = 'Proc02SelectD07D08'

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
        self.tw_period_map = self.metadata['tw_period_map']
        self.dev_tw = self.metadata['dev_tw']
        self.sample_partition = self.metadata['sample_table_partition_type']
        self.bigtable_partition = list(self.metadata['bigtable_partition_type'].values())[0]
        self.bigtable_ds_range = config.get('bigtable_ds_range')

        # 加载上一个实际执行步骤的剩余特征(扁平列表)
        self.steps = config.get('steps') or self.metadata['steps']
        self.remain_features = resolve_flat_remain_features(config['project_path'], self.steps, 'd07')

        # 加载merge表信息(如果d01执行过)
        merge_info = resolve_merge_table_info(config['project_path'], self.steps)
        self.merge_table_fea_map = merge_info['merge_table_fea_map'] if merge_info else None

        # 加载特征中文名
        feature_dict_path = os.path.join(self.metadata['data_path'], 'feature_dict.feather')
        feature_dict_df = pd.read_feather(feature_dict_path)
        self.feature_comment_map = dict(zip(feature_dict_df['feature_name'], feature_dict_df['feature_comment']))

        # d07-d08 参数(支持用户覆盖和Claude动态调整)
        self.d07_d08_params = resolve_params('D07_D08_PARAMS', config)

        # d07参数
        self.d07_concordance_threshold = self.d07_d08_params['d07_concordance_threshold']
        self.d07_range_ratio_threshold = self.d07_d08_params['d07_range_ratio_threshold']
        self.d07_spearman_threshold = self.d07_d08_params['d07_spearman_threshold']
        self.d07_min_cnt_pct = self.d07_d08_params['d07_min_cnt_pct']
        self.d07_require_all_windows = self.d07_d08_params['d07_require_all_windows']

        # 抽样参数
        self.random_num = self.d07_d08_params['random_num']
        self.random_seed = self.d07_d08_params['random_seed'] or np.random.randint(1, 10000)

    def _get_table_fea_map_filtered(self):
        """构建过滤后的 table→remain_features 映射"""
        if self.merge_table_fea_map is not None:
            table_fea_map = self.merge_table_fea_map
        else:
            table_fea_map = self.table_fea_map

        remain_set = set(self.remain_features)
        table_fea_map_filtered = {}
        for table, fea_list in table_fea_map.items():
            fea_list = list(fea_list) if not isinstance(fea_list, list) else fea_list
            remain_in_table = [f for f in fea_list if f in remain_set]
            if remain_in_table:
                table_fea_map_filtered[table] = remain_in_table
        return table_fea_map_filtered

    def _load_single_table_data(self, table, fea_list):
        """加载单张宽表的全窗口数据, 返回 DataFrame(样本基础列 + 特征列)"""
        single_table_map = {table: fea_list}

        # 使用公共函数生成SQL (全窗口模式, 不过滤tw)
        # 拼接表分区与样本表一致，原始宽表用自身分区
        if self.merge_table_fea_map is not None and table in self.merge_table_fea_map:
            bigtable_partition = self.sample_partition
            bigtable_ds_range = None  # merge表不需要ds_range
        else:
            bigtable_partition = self.bigtable_partition
            bigtable_ds_range = self.bigtable_ds_range

        join_sql = cons_join_sql(
            table_fea_map=single_table_map,
            sample_table=self.sample_table,
            id_col=self.id_col,
            target_col=self.target_col,
            tw_col_or_ins_oos_col=[self.tw_col],
            dev_tw_filter=None,  # 全窗口不过滤
            random_num=self.random_num,
            random_seed=self.random_seed,
            sample_partition=self.sample_partition,
            bigtable_partition=bigtable_partition,
            rh_feature_list=self.rh_feature_list,
            bigtable_ds_range=bigtable_ds_range,
        )

        # 保存 SQL
        table_rename = '__dot__'.join(table.split('.'))
        sql_save_path = os.path.join(self.proc_cache_path, f'{table_rename}_all_window.sql')
        with open(sql_save_path, 'w') as f:
            f.write(join_sql)

        main_logger = get_main_logger()
        client = TMLSQLClient()
        try:
            df = safe_sql_execute(client, join_sql, main_logger, desc=f"d07_d08-{table}", project='dw_backdate')
        finally:
            client.stop()

        # 特殊值替换
        rh_fea_in_data = [col for col in fea_list if col in self.rh_feature_list and col in df.columns]
        for col in rh_fea_in_data:
            df[col] = df[col].replace([np.inf, -np.inf, -999, -998], np.nan)

        return df

    @capture_print
    def _run_d07(self, all_fea_stt):
        """d07: WOE趋势稳定性筛选"""
        main_logger = get_main_logger()
        checkpoint_path = os.path.join(self.metadata['result_path'], 'Proc02SelectD07D08_d07_detail.pkl')
        if os.path.exists(checkpoint_path):
            main_logger.info("d07 已有checkpoint, 跳过")
            with open(checkpoint_path, 'rb') as f:
                result = pickle.load(f)
            return result['drop_features']

        step_start = time.time()
        main_logger.info(f"=== d07 WOE趋势稳定性筛选, 输入特征数: {len(all_fea_stt)} ===")

        # 提取OOT窗口列表
        all_windows = list(self.tw_period_map.keys())
        compare_tw_list = [tw for tw in all_windows if tw not in self.dev_tw]
        main_logger.info(f"DEV窗口: {self.dev_tw}, OOT窗口: {compare_tw_list}")

        # 调用d07筛选
        stability_detail, drop_reasons, drop_features = d07_select_by_woe_trend(
            all_fea_stt=all_fea_stt,
            compare_tw_list=compare_tw_list,
            dev_tw=self.dev_tw[0] if isinstance(self.dev_tw, list) else self.dev_tw,
            min_cnt_pct=self.d07_min_cnt_pct,
            concordance_threshold=self.d07_concordance_threshold,
            range_ratio_threshold=self.d07_range_ratio_threshold,
            spearman_threshold=self.d07_spearman_threshold,
            require_all_windows=self.d07_require_all_windows,
        )

        elapsed = time.time() - step_start
        main_logger.info(f"d07 结果: 剔除 {len(drop_features)} 个特征, 剩余 {len(all_fea_stt) - len(drop_features)} 个, 耗时: {elapsed:.1f}s")

        # 保存结果
        safe_pickle_dump({
            'stability_detail': stability_detail,
            'drop_reasons': drop_reasons,
            'drop_features': drop_features,
        }, checkpoint_path)

        return drop_features

    @capture_print
    def _run_d08(self, all_fea_stt_filtered):
        """d08: WOE解释性摘要生成"""
        main_logger = get_main_logger()
        checkpoint_path = os.path.join(self.metadata['result_path'], 'Proc02SelectD07D08_d08_summary.pkl')
        if os.path.exists(checkpoint_path):
            main_logger.info("d08 已有checkpoint, 跳过")
            return

        step_start = time.time()
        main_logger.info(f"=== d08 WOE解释性摘要生成, 输入特征数: {len(all_fea_stt_filtered)} ===")

        # 提取OOT窗口列表
        all_windows = list(self.tw_period_map.keys())
        compare_tw_list = [tw for tw in all_windows if tw not in self.dev_tw]

        # 调用d08生成摘要
        woe_summary_list, woe_summary_text = d08_select_by_woe_explain(
            all_fea_stt=all_fea_stt_filtered,
            feature_comment_map=self.feature_comment_map,
            dev_tw=self.dev_tw[0] if isinstance(self.dev_tw, list) else self.dev_tw,
            compare_tw_list=compare_tw_list,
        )

        elapsed = time.time() - step_start
        main_logger.info(f"d08 结果: 生成 {len(woe_summary_list)} 个特征的WOE摘要, 耗时: {elapsed:.1f}s")

        # 保存结果
        safe_pickle_dump({
            'woe_summary_list': woe_summary_list,
            'woe_summary_text': woe_summary_text,
        }, checkpoint_path)

    def run(self):
        main_logger = get_main_logger()
        main_logger.info("开始执行Proc02SelectD07D08流程")
        run_start = time.time()

        table_fea_map_filtered = self._get_table_fea_map_filtered()
        total_tables = len(table_fea_map_filtered)
        total_features = sum(len(fl) for fl in table_fea_map_filtered.values())
        main_logger.info(f"涉及 {total_tables} 张表, 共 {total_features} 个特征")

        # 汇总 checkpoint 路径
        all_fea_stt_path = os.path.join(self.proc_cache_path, 'all_fea_stt.pkl')
        all_fea_fig_path = os.path.join(self.proc_cache_path, 'all_fea_fig.pkl')

        if os.path.exists(all_fea_stt_path) and os.path.exists(all_fea_fig_path):
            # 全量 checkpoint 已存在, 直接加载
            main_logger.info("WOE分箱数据(全量)已存在, 直接加载")
            with open(all_fea_stt_path, 'rb') as f:
                all_fea_stt = pickle.load(f)
            with open(all_fea_fig_path, 'rb') as f:
                all_fea_fig = pickle.load(f)
            model_features = list(all_fea_stt.keys())
        else:
            # 逐宽表加载数据 → 计算WOE → 释放内存, 支持单表级 checkpoint
            all_fea_stt = {}
            all_fea_fig = {}
            dev_tw = self.dev_tw[0] if isinstance(self.dev_tw, list) else self.dev_tw

            for idx, (table, fea_list) in enumerate(table_fea_map_filtered.items()):
                table_rename = '__dot__'.join(table.split('.'))
                table_stt_path = os.path.join(self.proc_cache_path, f'{table_rename}_stt.pkl')
                table_fig_path = os.path.join(self.proc_cache_path, f'{table_rename}_fig.pkl')

                if os.path.exists(table_stt_path) and os.path.exists(table_fig_path):
                    main_logger.info(f"[{idx+1}/{total_tables}] {table} WOE checkpoint已存在, 跳过")
                    with open(table_stt_path, 'rb') as f:
                        table_stt = pickle.load(f)
                    with open(table_fig_path, 'rb') as f:
                        table_fig = pickle.load(f)
                else:
                    table_start = time.time()
                    main_logger.info(f"[{idx+1}/{total_tables}] 加载 {table}, 特征数: {len(fea_list)}")
                    df = self._load_single_table_data(table, fea_list)
                    valid_feas = [f for f in fea_list if f in df.columns]
                    main_logger.info(f"  数据 shape: {df.shape}, 有效特征: {len(valid_feas)}")

                    with capture_print():
                        table_stt, table_fig = split_plot_feature(
                            df=df,
                            feature_list=valid_feas,
                            tw_col=self.tw_col,
                            dev_tw=dev_tw,
                            tgt_col=self.target_col,
                            method=self.d07_d08_params['woe_method'],
                            num_nbins=self.d07_d08_params['woe_num_nbins'],
                        )

                    table_elapsed = time.time() - table_start
                    main_logger.info(f"  WOE分箱完成, 特征数: {len(table_stt)}, 耗时: {table_elapsed:.1f}s")

                    # 保存单表 checkpoint (原子性写入)
                    safe_pickle_dump(table_stt, table_stt_path)
                    safe_pickle_dump(table_fig, table_fig_path)

                    # 释放大 DataFrame
                    del df
                    gc.collect()

                all_fea_stt.update(table_stt)
                all_fea_fig.update(table_fig)

            # 保存全量汇总 checkpoint (原子性写入)
            main_logger.info(f"WOE分箱全部完成, 共 {len(all_fea_stt)} 个特征, 保存全量checkpoint")
            safe_pickle_dump(all_fea_stt, all_fea_stt_path)
            safe_pickle_dump(all_fea_fig, all_fea_fig_path)

            model_features = list(all_fea_stt.keys())

        main_logger.info(f"实际可用特征数: {len(model_features)}")

        # 3. d07: WOE趋势稳定性筛选
        d07_drop = []
        if 'd07' in self.steps:
            d07_drop = self._run_d07(all_fea_stt)
            remain_after_d07 = [f for f in model_features if f not in d07_drop]
            main_logger.info(f"d07后剩余特征: {len(remain_after_d07)}")
        else:
            remain_after_d07 = model_features
            main_logger.info("d07 不在 steps 中, 跳过")

        # 4. d08: WOE解释性摘要生成(仅对d07保留的特征)
        if 'd08' in self.steps:
            all_fea_stt_filtered = {fea: stt for fea, stt in all_fea_stt.items() if fea in remain_after_d07}
            self._run_d08(all_fea_stt_filtered)
        else:
            main_logger.info("d08 不在 steps 中, 跳过")

        # 5. 保存最终剩余特征 (原子性写入)
        remain_save_path = os.path.join(self.metadata['result_path'], 'Proc02SelectD07D08_remain_features.pkl')
        safe_pickle_dump(remain_after_d07, remain_save_path)

        # 6. 保存WOE图到Excel (只保存d07后剩余特征的图)
        woe_plots_dir = os.path.join(self.metadata['result_path'], 'woe_plots')
        os.makedirs(woe_plots_dir, exist_ok=True)
        main_logger.info(f"保存WOE图到: {woe_plots_dir}")

        all_fea_fig_filtered = {fea: fig for fea, fig in all_fea_fig.items() if fea in remain_after_d07}
        split_plot_to_excel(
            output_dir=woe_plots_dir,
            prefix=self.project_name,
            all_fea_fig=all_fea_fig_filtered,
            feature_comment_map=self.feature_comment_map,
        )

        total_elapsed = time.time() - run_start
        main_logger.info(f"*** Proc02SelectD07D08 完成 ***")
        main_logger.info(f"初始特征: {len(model_features)}, d07剔除: {len(d07_drop)}, 最终剩余: {len(remain_after_d07)}")
        main_logger.info(f"Proc02SelectD07D08 总耗时: {total_elapsed:.1f}s")
