import os
import time
import pickle
import pandas as pd
import numpy as np
from tmlpatch.database import TMLSQLClient
from procs.base_proc import BaseProc
from utils.data_utility import str_format, merge_table_set, safe_sql_execute, safe_pickle_dump
from utils.log_config import get_main_logger
from utils.param_manager import resolve_params



class Proc02SelectD01Merge(BaseProc):
    '''
    对d01的筛选结果进行宽表合并
    使用动态分区(INSERT OVERWRITE TABLE ... PARTITION)一次性写入所有分区
    '''

    PROC_CACHE_NAME = 'Proc02SelectD01Merge'

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
        self.dev_tw = self.metadata['dev_tw']
        self.sample_partition = self.metadata['sample_table_partition_type']
        self.bigtable_partition = list(self.metadata['bigtable_partition_type'].values())[0]
        self.bigtable_ds_range = config.get('bigtable_ds_range')

        # 读取d01的剩余特征
        table_remain_fea_save_path = os.path.join(self.metadata['result_path'], 'Proc02SelectD01_table_remain_fea.pkl')
        with open(table_remain_fea_save_path, 'rb') as f:
            self.table_remain_fea = pickle.load(f)

        # ---- 宽表分组方案加载 ----
        # merge_table_set() 的分组结果(merge_set)是确定性的(相同输入 → 相同分组),
        # 但生成的表名含时间戳, 每次调用都不同。因此断点续跑时必须从缓存恢复,
        # 否则新表名与数据库中已创建的表对不上, 导致重跑或报错。
        #
        # 三条路径:
        #   1. merge_plan.pkl 存在 → 直接加载(正常续跑)
        #   2. merge_plan.pkl 不存在, 但有 *_sql.sql 文件 → 旧版代码遗留的中间状态,
        #      重算分组(确定性) + 从 SQL 文件名提取旧表名, 按索引(_000, _001...)对齐还原
        #   3. 都不存在 → 全新运行
        merge_plan_path = os.path.join(self.proc_cache_path, 'merge_plan.pkl')
        main_logger = get_main_logger()
        d01_merge_params = resolve_params('D01_MERGE_PARAMS', config)
        if os.path.exists(merge_plan_path):
            with open(merge_plan_path, 'rb') as f:
                merge_plan = pickle.load(f)
            self.merge_set = merge_plan['merge_set']
            self.merge_table_map = merge_plan['merge_table_map']
            self.merge_table_fea_map = merge_plan['merge_table_fea_map']
            main_logger.info(f"[merge plan] 从缓存加载, {len(self.merge_table_map)} 张merge表")
        else:
            # 路径2: 旧版代码中断恢复
            # 旧版 _cons_sql() 会将 SQL 保存为 {merge_table_name}_sql.sql,
            # 文件名中包含旧的时间戳表名, 可用于还原。
            # 原理: merge_table_set() 分组确定性 → 重算得到相同的 merge_set →
            #        按索引与旧表名一一对应 → 还原完整的 merge_table_map
            existing_sql_files = sorted([f for f in os.listdir(self.proc_cache_path) if f.endswith('_sql.sql')])
            if existing_sql_files:
                main_logger.info(f"[merge plan] 检测到旧版SQL文件 {len(existing_sql_files)} 个, 从文件名恢复merge plan")
                merge_set_new, _, _ = merge_table_set(
                    self.table_remain_fea,
                    max_fea_num=d01_merge_params['max_fea_num'],
                    max_table_num=d01_merge_params['max_table_num'],
                    prefix=f'{self.project_name}_preselect',
                    project='dw_backdate',
                )
                old_names = sorted([f.replace('_sql.sql', '') for f in existing_sql_files])
                self.merge_set = merge_set_new
                self.merge_table_map = {name: group for name, group in zip(old_names, merge_set_new)}
                self.merge_table_fea_map = {
                    name: [fea for table in group for fea in self.table_remain_fea[table]]
                    for name, group in self.merge_table_map.items()
                }
                main_logger.info(f"[merge plan] 恢复完成, 表名: {list(self.merge_table_map.keys())}")
            else:
                # 路径3: 全新运行
                main_logger.info("[merge plan] 全新生成分组方案")
                self.merge_set, self.merge_table_map, self.merge_table_fea_map = merge_table_set(
                    self.table_remain_fea,
                    max_fea_num=d01_merge_params['max_fea_num'],
                    max_table_num=d01_merge_params['max_table_num'],
                    prefix=f'{self.project_name}_preselect',
                    project='dw_backdate',
                )
                main_logger.info(f"[merge plan] 生成 {len(self.merge_table_map)} 张merge表")

            safe_pickle_dump({
                'merge_set': self.merge_set,
                'merge_table_map': self.merge_table_map,
                'merge_table_fea_map': self.merge_table_fea_map,
            }, merge_plan_path)

    def _cons_sql(self):
        # ---- SQL模板缓存加载 ----
        # 两条路径:
        #   1. merge_table_sql_map.pkl 存在 → 直接加载
        #   2. 不存在 → 全新生成(动态分区: CREATE TABLE + INSERT OVERWRITE)
        main_logger = get_main_logger()
        sql_cache_path = os.path.join(self.proc_cache_path, 'merge_table_sql_map.pkl')

        if os.path.exists(sql_cache_path):
            with open(sql_cache_path, 'rb') as f:
                cached = pickle.load(f)
            # 校验缓存格式: 动态分区模式期望 dict 格式 {table: {'create':..., 'insert':...}}
            # 如果缓存是旧版逐分区模式的字符串格式, 需要丢弃重新生成
            sample_val = next(iter(cached.values())) if cached else None
            if isinstance(sample_val, dict):
                main_logger.info("[cons_sql] 从缓存加载SQL模板")
                return cached
            else:
                main_logger.info("[cons_sql] 缓存为旧版格式, 将重新生成")
                os.remove(sql_cache_path)

        # 全新生成
        main_logger.info("[cons_sql] 全新生成SQL模板")

        join_sql_template = """
        left join
        (
            select *
            from
            (
                select {id_col}, {{fea_cols_cast_float}}, row_number() over(partition by {id_col} order by rand(10) asc) as rn
                from {{bigtable}}
                {bigtable_ds_info}
            ) as tt
            where tt.rn = 1 -- 去重
        ) as t{{idx}}
        {id_join}
        """

        cast_float_template = """cast({fea} as float) as {fea}"""

        common_sql_params = {
            'id_col': ','.join(self.id_col),
            'bigtable_ds_info': (f'where {self.bigtable_partition[0]} >= "{self.bigtable_ds_range[0]}" and {self.bigtable_partition[0]} <= "{self.bigtable_ds_range[1]}"'
                                 if self.bigtable_ds_range and self.bigtable_partition
                                 else (f'where {self.bigtable_partition[0]} is not null' if self.bigtable_partition else '')),
            'id_join': 'on ' + ' and '.join([f'a.{col} = t{{idx}}.{col}' for col in self.id_col]),
        }
        join_sql_template = join_sql_template.format(**common_sql_params)

        if self.sample_partition:
            # 有分区: 动态分区模式
            # CREATE TABLE: 建空的分区表(id列STRING, 特征列FLOAT, 分区列ds)
            # INSERT OVERWRITE: 一条SQL写入所有分区, 引擎按ds列值自动路由
            ds_col = self.sample_partition[0]

            create_sql_template = """
        create table if not exists {{merge_table_name}} (
            {id_col_defs},
            {{fea_col_defs}}
        ) partitioned by ({ds_col} string)
            """
            insert_sql_template = """
        insert overwrite table {{merge_table_name}} partition ({ds_col})
        select {main_id_col},
            {{all_fea_cols}},
            a.{ds_col}
        from
        (
            select {id_col}, {ds_col}
            from {sample_table}
            where {ds_col} is not null
        ) as a -- 样本表(全量分区)
        {{join_sql_all}}
            """

            create_sql_params = {
                'id_col_defs': ',\n            '.join([f'{col} string' for col in self.id_col]),
                'ds_col': ds_col,
            }
            insert_sql_params = {
                'main_id_col': ','.join([f'a.{col}' for col in self.id_col]),
                'id_col': ','.join(self.id_col),
                'sample_table': self.sample_table,
                'ds_col': ds_col,
            }
            create_sql_template = create_sql_template.format(**create_sql_params)
            insert_sql_template = insert_sql_template.format(**insert_sql_params)
        else:
            # 无分区: CREATE TABLE AS SELECT
            create_sql_template = None
            insert_sql_template = """
        create table if not exists {{merge_table_name}} as
        select {main_id_col},
            {{all_fea_cols}}
        from
        (
            select {id_col}
            from {sample_table}
        ) as a -- 样本表
        {{join_sql_all}}
            """
            insert_sql_params = {
                'main_id_col': ','.join([f'a.{col}' for col in self.id_col]),
                'id_col': ','.join(self.id_col),
                'sample_table': self.sample_table,
            }
            insert_sql_template = insert_sql_template.format(**insert_sql_params)

        merge_table_sql_map = dict()
        for merge_table_name, table_set in self.merge_table_map.items():
            all_fea_cols = self.merge_table_fea_map[merge_table_name]

            join_sql_list = list()
            for idx, table in enumerate(table_set):
                fea_cols = self.table_remain_fea[table]
                fea_cols_cast_float = ','.join([cast_float_template.format(fea=fea) for fea in fea_cols])
                join_sql = join_sql_template.format(
                    idx=idx,
                    bigtable=table,
                    fea_cols_cast_float=fea_cols_cast_float,
                )
                join_sql_list.append(join_sql)

            join_sql_all = '\n'.join(join_sql_list)

            if self.sample_partition:
                # 动态分区: 返回 {create, insert} 两条SQL
                fea_col_defs = ',\n            '.join([f'{fea} float' for fea in all_fea_cols])
                create_sql = create_sql_template.format(
                    merge_table_name=merge_table_name,
                    fea_col_defs=fea_col_defs,
                )
                insert_sql = insert_sql_template.format(
                    merge_table_name=merge_table_name,
                    all_fea_cols=','.join(all_fea_cols),
                    join_sql_all=join_sql_all,
                )
                merge_table_sql_map[merge_table_name] = {'create': create_sql, 'insert': insert_sql}

                sql_save_path = os.path.join(self.proc_cache_path, f'{merge_table_name}_sql.sql')
                with open(sql_save_path, 'w') as f:
                    f.write(f"-- CREATE TABLE\n{create_sql}\n\n-- INSERT OVERWRITE\n{insert_sql}")
            else:
                # 非分区: 返回单条CTAS SQL
                merge_sql = insert_sql_template.format(
                    merge_table_name=merge_table_name,
                    all_fea_cols=','.join(all_fea_cols),
                    join_sql_all=join_sql_all,
                )
                merge_table_sql_map[merge_table_name] = {'create': None, 'insert': merge_sql}

                sql_save_path = os.path.join(self.proc_cache_path, f'{merge_table_name}_sql.sql')
                with open(sql_save_path, 'w') as f:
                    f.write(merge_sql)

        safe_pickle_dump(merge_table_sql_map, sql_cache_path)
        return merge_table_sql_map

    def run(self):
        main_logger = get_main_logger()
        main_logger.info("开始执行Proc02SelectD01Merge流程")
        run_start = time.time()

        total_merge_tables = len(self.merge_table_map)
        total_source_tables = sum(len(v) for v in self.merge_table_map.values())
        main_logger.info(f"[参数] 实际生成merge表数: {total_merge_tables}, 涉及宽表: {total_source_tables} 张")

        merge_table_sql_map = self._cons_sql()

        # 断点续跑: 跳过已完成的merge表
        temp_path = os.path.join(self.proc_cache_path, 'temp_merge_info.pkl')
        if os.path.exists(temp_path):
            with open(temp_path, 'rb') as f:
                exist_tables = pickle.load(f)
        else:
            exist_tables = []

        for idx, (merge_table_name, table_sql) in enumerate(merge_table_sql_map.items()):
            if merge_table_name not in exist_tables:
                table_start = time.time()
                main_logger.info(f"[{idx+1}/{total_merge_tables}] 开始拼接宽表: {merge_table_name}")

                create_sql = table_sql['create']
                insert_sql = table_sql['insert']

                # 建表(有分区时为空分区表, 无分区时create_sql为None跳过)
                if create_sql:
                    client = TMLSQLClient()
                    try:
                        safe_sql_execute(client, create_sql, main_logger, desc=f"建表-{merge_table_name}", project='dw_backdate')
                    finally:
                        client.stop()

                # 写入数据(有分区: INSERT OVERWRITE动态分区; 无分区: CTAS)
                client = TMLSQLClient()
                try:
                    safe_sql_execute(client, insert_sql, main_logger, desc=f"写入-{merge_table_name}", project='dw_backdate')
                finally:
                    client.stop()

                exist_tables.append(merge_table_name)
                safe_pickle_dump(exist_tables, temp_path)

                table_elapsed = time.time() - table_start
                main_logger.info(f"[{idx+1}/{total_merge_tables}] 宽表 {merge_table_name} 拼接完成, 耗时: {table_elapsed:.1f}s")
            else:
                main_logger.info(f"[{idx+1}/{total_merge_tables}] 宽表 {merge_table_name} 已存在, 跳过")

        # 保存拼接结果表信息
        save_path = os.path.join(self.metadata['result_path'], 'Proc02SelectD01Merge_merge_info.pkl')
        safe_pickle_dump({
            'merge_set': self.merge_set,
            'merge_table_map': self.merge_table_map,
            'merge_table_fea_map': self.merge_table_fea_map,
        }, save_path)

        total_elapsed = time.time() - run_start
        main_logger.info(f"*** Proc02SelectD01Merge 完成 ***")
        main_logger.info(f"生成 {total_merge_tables} 张merge表, 合并了 {total_source_tables} 张源宽表")
        main_logger.info(f"Proc02SelectD01Merge 总耗时: {total_elapsed:.1f}s")
