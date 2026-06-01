import os
import time
import pickle
import pandas as pd
import numpy as np
from tmlpatch.database import TMLSQLClient
from procs.base_proc import BaseProc
from utils.data_utility import str_format, safe_sql_execute, safe_pickle_dump
from utils.log_config import get_main_logger
from utils.feature_select import d01_preselect_by_toad
from utils.decorators import capture_print
from utils.param_manager import resolve_params



main_logger = get_main_logger()


def gen_bigtable_sql(sql_template, bigtable, fea_cols, random_seed, random_num):
    cast_float_template = """cast({fea} as float) as {fea}"""

    cast_float_sql_list = [cast_float_template.format(fea=fea) for fea in fea_cols]
    bigtable_sql = sql_template.format(
        bigtable=bigtable,
        fea_cols=','.join(fea_cols),
        fea_cols_cast_float=','.join(cast_float_sql_list),
        random_seed=random_seed,
        random_num=random_num,
    )

    return bigtable_sql


class Proc02SelectD01(BaseProc):
    '''
    d01toad初筛流程:
    1. 根据配置生成sql模版
    2. 遍历宽表进行取数和筛选
    3. 保存筛选结果
    '''

    PROC_CACHE_NAME = 'Proc02SelectD01'

    def __init__(self, config):
        super().__init__(config)

        # 校验 metadata 是否存在
        metadata_save_path = os.path.join(config['project_path'], 'data', 'metadata.pkl')
        if not os.path.exists(metadata_save_path):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_save_path}，请先执行 Proc01Prepare")

        self.id_col = config['sample']['id_col'] if config.get('sample', {}).get('id_col') else self.metadata['id_col']
        self.sample_table = config['sample']['table']
        self.target_col = config['sample']['target_col']
        self.tw_col = config['sample']['tw_col']
        self.dev_tw = self.metadata['dev_tw']
        self.sample_partition = self.metadata['sample_table_partition_type']
        self.bigtable_partition = list(self.metadata['bigtable_partition_type'].values())[0]
        self.bigtable_ds_range = config.get('bigtable_ds_range')
        self.preselect_condition = {k: v for k, v in config['thresholds'].items() if k in ['iv', 'empty', 'corr']}

        # 参数加载(支持用户覆盖和Claude动态调整)
        d01_params = resolve_params('D01_PARAMS', config)
        self.random_num = d01_params['random_num']
        self.random_seed = d01_params['random_seed'] or np.random.randint(1, 10000)
        self.round_num = d01_params['round_num']
        self.use_native = d01_params['use_native']
        self.max_round = d01_params['max_round']

        self.load_sql_template = self._cons_sql()

    def _cons_sql(self):
        sql_template = """
        select {main_id_col},
            {target_col},
            {{fea_cols}}
        from
        (
            select {id_col}, {target_col}
            from {sample_table}
            where {target_col} in (0, 1)
            and {tw_col} in ({dev_tw})
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
                from {{bigtable}}
                {bigtable_ds_info}
            ) as bb
            where bb.rn = 1
        ) as b
        {id_join}
        """
        sql_params = {
            'main_id_col': ','.join([f'a.{col}' for col in self.id_col]),
            'target_col': self.target_col,
            'id_col': ','.join(self.id_col),
            'sample_table': self.sample_table,
            'tw_col': self.tw_col,
            'dev_tw': ','.join([str_format(i) for i in self.dev_tw]),
            'id_join': 'on ' + ' and '.join([f'a.{col} = b.{col}' for col in self.id_col]),
            'sample_ds_info': 'and {} is not null'.format(self.sample_partition[0]) if self.sample_partition else '',
            'bigtable_ds_info': (f'where {self.bigtable_partition[0]} >= "{self.bigtable_ds_range[0]}" and {self.bigtable_partition[0]} <= "{self.bigtable_ds_range[1]}"'
                                 if self.bigtable_ds_range and self.bigtable_partition
                                 else ('where {} is not null'.format(self.bigtable_partition[0]) if self.bigtable_partition else '')),
        }

        return sql_template.format(**sql_params)

    def _cons_result(self):
        # 宽表剩余特征
        all_files = os.listdir(self.proc_cache_path)
        all_files = [i for i in all_files if '__dot__' in i and i.endswith('.pkl')]
        table_remain_fea = dict()
        table_drop_info = dict()
        for file in all_files:
            # 从文件名反推表名：先去 .pkl 扩展名，再替换 __dot__ 为 .
            table = file.replace('.pkl', '').replace('__dot__', '.')
            filepath = os.path.join(self.proc_cache_path, file)
            with open(filepath, 'rb') as f:
                select_rlt = pickle.load(f)

            empty_drop = [b for k, v in select_rlt.items() for a in v for b in a[0]['empty']]
            corr_drop = [b for k, v in select_rlt.items() for a in v for b in a[0]['corr']]
            iv_drop = [b for k, v in select_rlt.items() for a in v for b in a[0]['iv']]
            table_drop_info[table] = {'empty': empty_drop, 'corr': corr_drop, 'iv': iv_drop, 'total': empty_drop + corr_drop + iv_drop}

            max_round = max(select_rlt)
            remain_fea = [b for a in select_rlt[max_round] for b in a[2]] # 取最后一轮的每组(500个)剩余变量
            table_remain_fea[table] = remain_fea

        drop_info_df = pd.concat({k: pd.DataFrame([[len(i) for i in v.values()]], columns=['empty', 'corr', 'iv', 'total_drop']) for k, v in table_drop_info.items()}).reset_index().drop(columns=['level_1']).rename(columns={'level_0': 'table'})
        drop_info_df['all'] = drop_info_df.table.apply(lambda x: len(self.table_fea_map[x]))
        drop_info_df['remain'] = drop_info_df.table.apply(lambda x: len(table_remain_fea[x]))
        drop_info_df['remain_rate'] = drop_info_df['remain'] / drop_info_df['all']

        return table_remain_fea, drop_info_df

    def run(self):
        main_logger.info("开始执行Proc02SelectD01流程")
        main_logger.info(f"[参数] 筛选条件: {self.preselect_condition}")
        run_start = time.time()

        # 要排除已经筛完的表
        all_files = os.listdir(self.proc_cache_path)
        all_files = [i for i in all_files if '__dot__' in i and i.endswith('.pkl')]
        exist_tables = [i.replace('.pkl', '').replace('__dot__', '.') for i in all_files]
        remain_tables = [i for i in self.table_fea_map if i not in exist_tables]
        total_tables = len(self.table_fea_map)
        main_logger.info(f"宽表总数: {total_tables}, 已完成: {len(exist_tables)}, 待筛选: {len(remain_tables)}")

        for idx, (bigtable, fea_cols) in enumerate(self.table_fea_map.items()):
            if bigtable in remain_tables: # 只跑剩余的表
                step_start = time.time()
                main_logger.info(f"[{idx+1}/{total_tables}] 开始筛选宽表: {bigtable}, 特征数: {len(fea_cols)}")
                table_rename = '__dot__'.join(bigtable.split('.'))
                client = TMLSQLClient()
                try:
                    # 生成sql并保存
                    bigtable_sql = gen_bigtable_sql(self.load_sql_template, bigtable, fea_cols, self.random_seed, self.random_num)
                    save_filepath = os.path.join(self.proc_cache_path, f'{table_rename}.sql')
                    with open(save_filepath, 'w') as f:
                        f.write(bigtable_sql)

                    # 取数并替换特殊值
                    df = safe_sql_execute(client, bigtable_sql, main_logger, desc=f"d01取数-{bigtable}")
                finally:
                    client.stop()
                main_logger.info("shape: {}, bad cnt: {}, bad rate: {}".format(df.shape, df[self.target_col].sum(), df[self.target_col].mean()))
                for col in fea_cols:
                    if col in self.rh_feature_list:
                        df[col] = df[col].replace([np.inf, -np.inf, -999, -998], np.nan)

                # 筛选
                main_logger.info(f"[参数] use_native={self.use_native}")
                with capture_print():
                    round_select_rlt = d01_preselect_by_toad(df=df, target_col=self.target_col, feature_list=fea_cols, preselect_condition=self.preselect_condition, round_num=self.round_num, use_native=self.use_native, max_round=self.max_round)

                # 保存筛选结果
                save_filepath = os.path.join(self.proc_cache_path, f'{table_rename}.pkl')
                safe_pickle_dump(round_select_rlt, save_filepath)

                elapsed = time.time() - step_start
                main_logger.info(f"[{idx+1}/{total_tables}] 宽表 {bigtable} 筛选完成, 耗时: {elapsed:.1f}s")

        # 汇总结果
        table_remain_fea, drop_info_df = self._cons_result()
        table_remain_fea_save_path = os.path.join(self.metadata['result_path'], 'Proc02SelectD01_table_remain_fea.pkl')
        safe_pickle_dump(table_remain_fea, table_remain_fea_save_path)

        drop_info_save_path = os.path.join(self.metadata['result_path'], 'Proc02SelectD01_drop_info.csv')
        drop_info_df.to_csv(drop_info_save_path, index=False)

        main_logger.info(f"*** 筛选结果: ***")
        for table, remain_fea_list in table_remain_fea.items():
            main_logger.info(f'宽表: {table}, 剩余特征数: {len(remain_fea_list)}')

        total_elapsed = time.time() - run_start
        main_logger.info(f"Proc02SelectD01 总耗时: {total_elapsed:.1f}s")
