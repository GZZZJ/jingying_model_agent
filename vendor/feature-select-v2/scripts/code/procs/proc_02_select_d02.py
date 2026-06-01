import os
import time
import pickle
import pandas as pd
import numpy as np
from tmlpatch.database import TMLSQLClient
from procs.base_proc import BaseProc
from utils.data_utility import str_format, safe_sql_execute, safe_pickle_dump
from utils.log_config import get_main_logger
from utils.feature_select import d02_psi_select, batch_psi
from utils.decorators import capture_print
from utils.remain_resolver import resolve_table_remain_fea, resolve_merge_table_info
from utils.param_manager import resolve_params



class Proc02SelectD02PSI(BaseProc):
    '''
    d02PSI筛选流程
    '''

    PROC_CACHE_NAME = 'Proc02SelectD02PSI'

    def __init__(self, config):
        super().__init__(config)

        # 校验 metadata 是否存在
        metadata_save_path = os.path.join(config['project_path'], 'data', 'metadata.pkl')
        if not os.path.exists(metadata_save_path):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_save_path}，请先执行 Proc01Prepare")

        self.project_name = config['project_name']
        self.id_col = config['sample']['id_col'] if config.get('sample', {}).get('id_col') else self.metadata['id_col']
        self.sample_table = config['sample']['table']
        self.tw_col = config['sample']['tw_col']
        self.period_col = config['sample']['period_col']
        self.dev_tw = self.metadata['dev_tw']
        self.dev_period = [p for dev in self.dev_tw for p in self.metadata['tw_period_map'][dev]]
        self.exp_period = [i for k, v in self.metadata['tw_period_map'].items() for i in v if i not in self.dev_period]
        self.sample_partition = self.metadata['sample_table_partition_type']
        self.bigtable_partition = list(self.metadata['bigtable_partition_type'].values())[0]
        self.bigtable_ds_range = config.get('bigtable_ds_range')
        self.psi_threshold = config['thresholds']['psi']

        # 读取merge结果(如果d01执行过且有拼接结果, 后续从拼接表取数)
        self.steps = config.get('steps') or self.metadata['steps']
        merge_info = resolve_merge_table_info(config['project_path'], self.steps)
        self.merge_table_fea_map = merge_info['merge_table_fea_map'] if merge_info else None

        # 参数加载(支持用户覆盖和Claude动态调整)
        d02_params = resolve_params('D02_PARAMS', config)
        self.random_num = d02_params['random_num']
        self.random_seed = d02_params['random_seed'] or np.random.randint(1, 10000)

        # exp_period 过滤: 默认只取 dev 之后的 period
        if d02_params['exp_after_dev_only'] and self.dev_period:
            max_dev = max(self.dev_period)
            self.exp_period = [p for p in self.exp_period if p > max_dev]

        # 加载上一个实际执行步骤的剩余特征
        self.table_remain_fea = resolve_table_remain_fea(config['project_path'], self.steps, 'd02')

        self.load_sql_template = self._cons_sql()

    def _cons_sql(self):
        sql_template = """
        select {main_id_col},
            {{fea_cols}}
        from
        (
            select {id_col}
            from {sample_table}
            where {period_col} in ({{periods}})
            {sample_ds_info}
            order by rand({{random_seed}}) -- 随机抽样种子
            limit {{random_num}} -- 随机抽样数
        ) as a
        left join
        (
            select *
            from
            (
                select {id_col},
                    {{fea_cols_cast_float}},
                    row_number() over(partition by {id_col} order by rand(10) asc) as rn
                from {{bigtable}} -- 特征宽表
                {bigtable_ds_info}
            ) as bb
            where bb.rn = 1 -- 去重
        ) as b
        {id_join}
        """
        sql_params = {
            'main_id_col': ','.join([f'a.{col}' for col in self.id_col]),
            'id_col': ','.join(self.id_col),
            'sample_table': self.sample_table,
            'period_col': self.period_col,
            'id_join': 'on ' + ' and '.join([f'a.{col} = b.{col}' for col in self.id_col]),
            'sample_ds_info': 'and {} is not null'.format(self.sample_partition[0]) if self.sample_partition else '',
        }
        if self.merge_table_fea_map is not None:
            bigtable_ds = self.sample_partition
            ds_range = None  # merge表不需要ds_range
        else:
            bigtable_ds = self.bigtable_partition
            ds_range = self.bigtable_ds_range

        if ds_range and bigtable_ds:
            sql_params['bigtable_ds_info'] = f'where {bigtable_ds[0]} >= "{ds_range[0]}" and {bigtable_ds[0]} <= "{ds_range[1]}"'
        elif bigtable_ds:
            sql_params['bigtable_ds_info'] = f'where {bigtable_ds[0]} is not null'
        else:
            sql_params['bigtable_ds_info'] = ''

        return sql_template.format(**sql_params)

    def _replace_special_values(self, df, fea_list):
        """对人行特征做特殊值替换"""
        rh_fea_in_data = [col for col in fea_list if col in self.rh_feature_list and col in df.columns]
        for col in rh_fea_in_data:
            df[col] = df[col].replace([np.inf, -np.inf, -999, -998], np.nan)

    def _load_data(self, table, fea_list):
        table_rename = '__dot__'.join(table.split('.'))
        fea_cols_cast_float = ','.join([f'cast({fea} as float) as {fea}' for fea in fea_list])
        dev_sql = self.load_sql_template.format(
            fea_cols=','.join(fea_list),
            fea_cols_cast_float=fea_cols_cast_float,
            bigtable=table,
            random_seed=self.random_seed,
            random_num=self.random_num,
            periods=','.join([str_format(i) for i in self.dev_period]),
        )

        # 保存 DEV SQL
        sql_save_path = os.path.join(self.proc_cache_path, f'{table_rename}_dev.sql')
        with open(sql_save_path, 'w') as f:
            f.write(dev_sql)

        client = TMLSQLClient()
        try:
            df = safe_sql_execute(client, dev_sql, get_main_logger(), desc=f"d02-DEV-{table}")
        finally:
            client.stop()
        self._replace_special_values(df, fea_list)

        yield 'base_DEV', df

        for p_idx, p in enumerate(self.exp_period):
            exp_sql = self.load_sql_template.format(
                fea_cols=','.join(fea_list),
                fea_cols_cast_float=fea_cols_cast_float,
                bigtable=table,
                random_seed=self.random_seed,
                random_num=self.random_num,
                periods=str_format(p),
            )

            # 保存 EXP SQL
            exp_sql_save_path = os.path.join(self.proc_cache_path, f'{table_rename}_exp_{p}.sql')
            with open(exp_sql_save_path, 'w') as f:
                f.write(exp_sql)

            client = TMLSQLClient()
            try:
                df = safe_sql_execute(client, exp_sql, get_main_logger(), desc=f"d02-EXP-{p}-{table}")
            finally:
                client.stop()
            self._replace_special_values(df, fea_list)
            yield f'exp_{p}', df

    def run(self):
        main_logger = get_main_logger()
        main_logger.info("开始执行Proc02SelectD02PSI流程")
        main_logger.info(f"[参数] PSI阈值: {self.psi_threshold}")
        run_start = time.time()

        if self.merge_table_fea_map is not None:
            table_fea_map = self.merge_table_fea_map.copy()
        else:
            table_fea_map = self.table_fea_map.copy()

        total_tables = len(table_fea_map)
        total_features = sum(len(fl) for fl in table_fea_map.values())
        main_logger.info(f"涉及 {total_tables} 张表, 共 {total_features} 个特征")

        # 逐表计算PSI, 每张表完成后保存pkl, 支持断点续跑
        all_psi_rlt = []
        for idx, (table, fea_list) in enumerate(table_fea_map.items()):
            table_rename = '__dot__'.join(table.split('.'))
            psi_cache_path = os.path.join(self.proc_cache_path, f'{table_rename}_psi.pkl')

            if os.path.exists(psi_cache_path):
                main_logger.info(f"[{idx+1}/{total_tables}] {table} PSI结果已存在, 跳过")
                with open(psi_cache_path, 'rb') as f:
                    psi_rlt = pickle.load(f)
            else:
                main_logger.info(f"[{idx+1}/{total_tables}] 开始计算 {table} 的PSI, 特征数: {len(fea_list)}")
                step_start = time.time()
                data_iter = self._load_data(table, fea_list)
                with capture_print():
                    psi_rlt = batch_psi(data_iter, fea_list, method='quantile', num_nbins=10)
                safe_pickle_dump(psi_rlt, psi_cache_path)
                elapsed = time.time() - step_start
                main_logger.info(f"[{idx+1}/{total_tables}] {table} PSI完成, 耗时: {elapsed:.1f}s")

            all_psi_rlt.append(psi_rlt)

        # 聚合: 计算每个特征的最大PSI, 超阈值的剔除
        fea_max_psi = {fea: max(psi_info.values()) for psi_rlt in all_psi_rlt for fea, psi_info in psi_rlt[2].items()}
        psi_drop_fea = [fea for fea, psi in fea_max_psi.items() if psi > self.psi_threshold]

        savepath = os.path.join(self.metadata['result_path'], 'Proc02SelectD02PSI_psi_info.pkl')
        safe_pickle_dump({
            'all_psi_rlt': all_psi_rlt,
            'fea_max_psi': fea_max_psi,
            'psi_drop_fea': psi_drop_fea,
        }, savepath)

        # 汇总结果
        table_remain_fea = {k: set(v) - set(psi_drop_fea) for k, v in self.table_remain_fea.items() if set(v) - set(psi_drop_fea)}
        table_remain_fea_save_path = os.path.join(self.metadata['result_path'], 'Proc02SelectD02PSI_table_remain_fea.pkl')
        safe_pickle_dump(table_remain_fea, table_remain_fea_save_path)

        remain_count = sum(len(v) for v in table_remain_fea.values())
        main_logger.info(f"PSI筛选完成: 剔除 {len(psi_drop_fea)} 个特征, 剩余 {remain_count} 个")

        total_elapsed = time.time() - run_start
        main_logger.info(f"Proc02SelectD02PSI 总耗时: {total_elapsed:.1f}s")
