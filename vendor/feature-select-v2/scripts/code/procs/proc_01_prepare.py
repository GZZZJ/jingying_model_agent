import os
import pickle
import pandas as pd
import numpy as np
from tmlpatch.database import TMLSQLClient
from utils.data_utility import show_table_dp, desc_table_dp
from utils.log_config import get_main_logger



main_logger = get_main_logger()


def check_config(config):
    '''
    检查配置项的完整性和正确性
    '''
    # 检查配置项是否存在
    required_keys = ['project_name', 'sample', 'thresholds', 'bigtable', 'feature_info', 'project_path']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置项缺失: {key}")

    # project_name 合法性校验（用于生成数据库表名）
    import re
    pn = config['project_name']
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', pn):
        raise ValueError(
            f"project_name 不合法: '{pn}'。"
            f"必须以字母开头，只能包含字母、数字和下划线"
        )
    if len(pn) > 60:
        raise ValueError(
            f"project_name 过长: {len(pn)} 字符（最大 60）。"
            f"合并表名 = project_name + '_preselect_YYYYMMDDHHmmSS_NNN__partition'，总长不能超过 128"
        )

    # 检查样本配置项
    sample_keys = ['table', 'id_col', 'target_col', 'tw_col', 'time_col', 'period_col', 'ins_oos_col']
    for key in sample_keys:
        if key not in config['sample']:
            raise ValueError(f"样本配置项缺失: {key}")

        if key == 'id_col' and not isinstance(config['sample'][key],list):
            raise ValueError(f"样本主键字段 id_col 应该是列表: {config['sample'][key]}")

    # 检查阈值配置项
    threshold_keys = ['iv', 'empty', 'corr', 'psi']
    for key in threshold_keys:
        if key not in config['thresholds']:
            raise ValueError(f"阈值配置项缺失: {key}")
    
    # 检查特征宽表配置项
    if not isinstance(config['bigtable'], list) or len(config['bigtable']) == 0:
        raise ValueError("特征宽表配置项必须是非空列表")


def check_table_exists(client, table_list):
    '''
    检查表是否存在
    '''
    for table in table_list:
        if '.' in table:
            database, kw = table.split('.')
        else:
            database, kw = None, table

        exist_tables = show_table_dp(client, kw, prec_mode=True, database=database)
        if not exist_tables:
            raise ValueError(f"表不存在: {table}")


def check_path_exists(project_path):
    '''
    检查项目路径是否存在
    '''
    if not os.path.exists(project_path):
        raise ValueError(f"项目路径不存在: {project_path}")


def check_thresholds(thresholds):
    '''
    检查阈值配置项的合理性
    '''
    # 检查各个阈值必须是数值型
    for key, value in thresholds.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"阈值配置项 {key} 必须是数值型: {value}")

    # 检查缺失率阈值的合理性
    if thresholds['empty'] < 0 or thresholds['empty'] > 1:
        raise ValueError(f"缺失率阈值必须在0和1之间: {thresholds['empty']}")

    # 检查相关性阈值的合理性
    if thresholds['corr'] < 0 or thresholds['corr'] > 1:
        raise ValueError(f"相关性阈值必须在0和1之间: {thresholds['corr']}")

    # 检查psi阈值的合理性
    if thresholds['psi'] < 0:
        raise ValueError(f"psi阈值必须大于等于0: {thresholds['psi']}")
    if thresholds['psi'] > 1:
        main_logger.warning(f"PSI阈值 {thresholds['psi']} 较大, 通常建议 0.1~0.25, 请确认是否合理")

    # 检查iv阈值的合理性
    if thresholds['iv'] < 0:
        raise ValueError(f"iv阈值必须大于等于0: {thresholds['iv']}")
    if thresholds['iv'] > 1:
        main_logger.warning(f"IV阈值 {thresholds['iv']} 较大, IV>1 通常意味着特征存在信息泄漏风险, 请确认是否合理")


def get_file_type(file):
    '''
    检查文件类型, 返回csv/excel/table
    '''
    if file.endswith('.csv'):
        return 'csv'
    elif file.endswith('.xlsx') or file.endswith('.xls'):
        return 'excel'
    else:
        return 'table'


class Proc01Prepare:
    '''
    特征筛选的前置准备工作:
    1. 检查所有配置: 配置项的完整性和正确性, 表是否存在, 样本表结构, 项目路径是否存在, 阈值配置项的合理性, 环境依赖的python包
    2. 保存配置项元数据: 包括样本及字典文件类型, 表分区格式, 主键类型, 样本窗口映射关系, 以便后续流程使用
    3. 初始化路径
    4. 保存样本表和特征字典到项目路径, 以便后续流程使用
    5. 生成特征和宽表的映射字典
    '''
    def __init__(self, config):
        self.config = config
        self.required_packages = ['pandas', 'numpy', 'scipy', 'sklearn', 'lightgbm', 'matplotlib']
        self.steps = ['d01', 'd02', 'd03', 'd04', 'd05', 'd06', 'd07', 'd08']
    
    def _check_config(self):
        '''
        检查配置项的完整性和正确性
        '''
        check_config(self.config)

    def _check_table_exists(self):
        '''
        检查表是否存在, 样本表和特征字典如果是表的话需要检查, 特征宽表也需要检查
        '''
        client = TMLSQLClient()
        try:
            check_table_exists(client, [self.config['sample']['table']])
            check_table_exists(client, self.config['bigtable'])

            feature_info_type = self._get_feature_info_type()
            if feature_info_type == 'table':
                check_table_exists(client, [self.config['feature_info']])
        finally:
            client.stop()

    def _check_project_path_exists(self):
        '''
        检查项目路径是否存在
        '''
        check_path_exists(self.config['project_path'])

    def _check_thresholds(self):
        '''
        检查阈值配置项的合理性
        '''
        check_thresholds(self.config['thresholds'])

    def _check_env_package(self):
        '''
        检查环境依赖的python包
        '''
        for package in self.required_packages:
            try:
                __import__(package)
            except ImportError:
                raise ImportError(f"缺少必要的python包: {package}")

        # 获取包版本（pip 包名与 import 名不同的需要映射）
        import subprocess
        _pip_name_map = {'sklearn': 'scikit-learn'}
        pip_names = [_pip_name_map.get(p, p) for p in self.required_packages]
        pkg_str = "pip3 list | grep -E '{}'".format('|'.join(pip_names))
        result = subprocess.run(pkg_str, shell=True, capture_output=True, text=True)
        pkg_ver = result.stdout

        return pkg_ver

    def _check_steps(self):
        '''
        检查特征筛选步骤的合法性:
        1. 不存在的步骤名
        2. 重复步骤
        3. 步骤顺序必须与规范顺序一致(d01 不能出现在 d02 后面)
        4. d03-d06 连续性: 配了靠后的步骤, 前面的也应在列表中(否则 warn)
        5. d08 依赖 d07: 配了 d08 但没配 d07 时 warn
        '''
        canonical = self.steps  # ['d01', 'd02', ..., 'd08']
        steps = self.config.get('steps', [])

        if not steps:
            return list(canonical)

        # 1) 不存在的步骤
        unknown = set(steps) - set(canonical)
        if unknown:
            raise ValueError(f"步骤配置项错误, 不存在的步骤: {unknown}")

        # 2) 重复步骤
        if len(steps) != len(set(steps)):
            from collections import Counter
            dup = {k: v for k, v in Counter(steps).items() if v > 1}
            raise ValueError(f"步骤配置项错误, 存在重复步骤: {dup}")

        # 3) 顺序校验: 用户列表中每个步骤在规范顺序中的 index 必须单调递增
        indices = [canonical.index(s) for s in steps]
        for i in range(1, len(indices)):
            if indices[i] <= indices[i - 1]:
                raise ValueError(
                    f"步骤配置项错误, 步骤顺序不合法: '{steps[i]}' 不能出现在 '{steps[i-1]}' 之前或相同位置, "
                    f"规范顺序为 {canonical}"
                )

        # 4) d03-d06 连续性提示: 如果配了 d04/d05/d06 但缺少前置步骤, 给出警告
        d03_d06_set = {'d03', 'd04', 'd05', 'd06'}
        active_d03_d06 = [s for s in steps if s in d03_d06_set]
        if active_d03_d06:
            expected_prefix = canonical[canonical.index('d03'):canonical.index(active_d03_d06[-1]) + 1]
            missing = [s for s in expected_prefix if s in d03_d06_set and s not in steps]
            if missing:
                main_logger.warning(
                    f"步骤配置提醒: 配置了 {active_d03_d06}, 但缺少前置步骤 {missing}, "
                    f"d03-d06 在同一 proc 中顺序执行, 缺少前置步骤会导致特征未经充分筛选"
                )

        # 5) d08 依赖 d07 提示
        if 'd08' in steps and 'd07' not in steps:
            main_logger.warning(
                "步骤配置提醒: 配置了 d08 但未配置 d07, d08 将在未经 d07 趋势筛选的特征上生成 WOE 摘要"
            )

        return steps

    def _check_sample_cols(self):
        '''
        检查样本配置的字段是否存在于样本表中
        注意: id_col 可能是列表(联合主键), 需要展平处理
        '''
        client = TMLSQLClient()
        try:
            col_info, _ = desc_table_dp(client, self.config['sample']['table'])
        finally:
            client.stop()
        table_cols = set(i[0] for i in col_info)

        # 收集所有需要检查的字段(展平 id_col 列表)
        sample_cols = []
        for key, value in self.config['sample'].items():
            if key == 'table':
                continue
            if isinstance(value, list):
                sample_cols.extend(value)
            elif isinstance(value, str):
                sample_cols.append(value)

        missing_cols = set(sample_cols) - table_cols
        if missing_cols:
            raise ValueError(f"样本配置的字段在表中不存在: {missing_cols}")

    def _get_feature_info_type(self):
        '''
        获取特征字典类型, 返回csv/excel/table
        '''
        return get_file_type(self.config['feature_info'])
    
    def _get_id_col(self):
        '''
        获取样本主键字段, 返回字符串或者列表
        '''
        id_col = self.config['sample']['id_col']
        if isinstance(id_col, str):
            return id_col
        elif isinstance(id_col, list):
            return id_col
        else:
            raise ValueError(f"样本主键字段类型错误: {id_col}")
        
    def _get_tw_period_map(self, partitions=None, cache_path=None):
        '''
        获取tw_col和period_col的映射关系, 以便后续流程使用
        '''
        client = TMLSQLClient()
        try:
            sql = """
            select {tw_col}, {period_col}
            from {sample_table}
            {ds_info}
            group by {tw_col}, {period_col}
            """
            if partitions:
                ds_info = "where {} is not null".format(partitions[0])
            else:
                ds_info = ""

            tw_col = self.config['sample']['tw_col']
            period_col=self.config['sample']['period_col']
            sample_table = self.config['sample']['table']
            sql_new = sql.format(tw_col=tw_col, period_col=period_col, sample_table=sample_table, ds_info=ds_info)

            # 保存 SQL
            if cache_path:
                sql_save_path = os.path.join(cache_path, 'get_tw_period_map.sql')
                with open(sql_save_path, 'w') as f:
                    f.write(sql_new)

            df = client.sql(sql_new).to_pandas()
            tw_period_map = df.groupby(tw_col)[period_col].apply(list).to_dict()
        finally:
            client.stop()

        return tw_period_map

    def cons_metadata(self):
        '''
        构建配置项元数据, 包括样本表结构信息和特征字典信息, 以便后续流程使用
        '''
        client = TMLSQLClient()
        try:
            metadata = dict()
            metadata['id_col'] = self._get_id_col()
            metadata['feature_info_type'] = self._get_feature_info_type()
            metadata['sample_table_partition_type'] = [i[0] for i in desc_table_dp(client, self.config['sample']['table'])[1]]
            metadata['bigtable_partition_type'] = {table: [i[0] for i in desc_table_dp(client, table)[1]] for table in self.config['bigtable']}

            if metadata['feature_info_type'] == 'table':
                metadata['feature_table_partition_type'] = [i[0] for i in desc_table_dp(client, self.config['feature_info'])[1]]
            else:
                metadata['feature_table_partition_type'] = None
        finally:
            client.stop()
        
        metadata['env_package'] = self._check_env_package()
        metadata['steps'] = self._check_steps()
        metadata['log_path'] = os.path.join(self.config['project_path'], 'logs')
        metadata['data_path'] = os.path.join(self.config['project_path'], 'data')
        metadata['cache_path'] = os.path.join(self.config['project_path'], 'cache')
        metadata['result_path'] = os.path.join(self.config['project_path'], 'results')
        metadata['tw_period_map'] = self._get_tw_period_map(metadata['sample_table_partition_type'], cache_path=getattr(self, '_proc_cache_path', None))
        metadata['dev_tw'] = [i for i in metadata['tw_period_map'] if i.startswith('DEV')]

        # 校验: 必须存在 DEV 窗口
        if not metadata['dev_tw']:
            all_tw = list(metadata['tw_period_map'].keys())
            raise ValueError(
                f"样本表中未找到以 'DEV' 开头的时间窗口, 所有窗口值: {all_tw}, "
                f"请检查样本表的 {self.config['sample']['tw_col']} 列"
            )

        # 校验: 宽表分区类型一致性(后续 proc 假设所有宽表分区格式相同)
        partition_types = metadata['bigtable_partition_type']
        unique_partitions = set(tuple(v) for v in partition_types.values())
        if len(unique_partitions) > 1:
            main_logger.warning(
                f"多张宽表的分区格式不一致: {partition_types}, "
                f"后续流程将使用第一张宽表的分区格式, 可能导致部分宽表的 SQL 不正确"
            )

        return metadata


    def save_feature_dict(self, data_path, file_type, partitions=None, cache_path=None):
        '''
        保存特征字典feather到项目路径, 以便后续流程使用
        '''
        file = self.config['feature_info']
        savepath = os.path.join(data_path, 'feature_dict.feather')
        if file_type == 'csv':
            df = pd.read_csv(file)
        elif file_type == 'excel':
            df = pd.read_excel(file)
        elif file_type == 'table':
            client = TMLSQLClient()
            try:
                sql = """select * from {file} {ds_info}"""
                if partitions:
                    ds_info = "where {} = max_pt('{}')".format(partitions[0], file)
                else:
                    ds_info = ""
                sql_new = sql.format(file=file, ds_info=ds_info)

                # 保存 SQL
                if cache_path:
                    sql_save_path = os.path.join(cache_path, 'load_feature_dict.sql')
                    with open(sql_save_path, 'w') as f:
                        f.write(sql_new)

                df = client.sql(sql_new).to_pandas()
            finally:
                client.stop()

        # 校验特征字典必须包含 feature_name/feature_comment/category_name 列
        required_fea_cols = ['feature_name', 'feature_comment', 'category_name']
        missing_fea_cols = [c for c in required_fea_cols if c not in df.columns]
        if missing_fea_cols:
            raise ValueError(
                f"特征字典缺少必须列: {missing_fea_cols}, "
                f"特征字典必须包含 {required_fea_cols}, 当前列: {df.columns.tolist()}"
            )

        df.to_feather(savepath)

    def cons_table_fea_map(self, feature_list):
        '''
        构建特征和宽表的映射字典
        '''
        from collections import Counter

        client = TMLSQLClient()
        try:
            table_fea_map = dict()
            for table in self.config['bigtable']:
                col_info, parti_info = desc_table_dp(client, table)
                fea_list = [i[0] for i in col_info if i[0] in feature_list]
                table_fea_map[table] = fea_list
        finally:
            client.stop()

        # 宽表特征可能有重复, 进行去重处理
        all_cols = [fea for _, fea_list in table_fea_map.items() for fea in fea_list]
        num_tables, num_feas = len(table_fea_map), len(all_cols)
        main_logger.info(f"宽表总数: {num_tables}, 特征总数: {num_feas}")
        col_cnt = Counter(all_cols)
        dup_cols = {k: v for k, v in col_cnt.items() if v > 1}
        if dup_cols:
            main_logger.info(f"重复特征数: {len(dup_cols)}")
            # 去重, 使用第一次出现的表
            exist_cols = list()
            table_fea_map_new = dict()
            for table, fea_list in table_fea_map.items():
                new_fea_list = [i for i in fea_list if i not in exist_cols]
                table_fea_map_new[table] = new_fea_list
                exist_cols += new_fea_list

            main_logger.info(f"去重后宽表总数: {num_tables}, 特征总数: {len(exist_cols)}")

            return table_fea_map_new
        else:
            main_logger.info("无重复特征数")
            return table_fea_map

    def run(self):
        main_logger.info("开始任务01: 前置准备")
        # 1. 检查所有配置
        main_logger.info("检查配置项的完整性和正确性")
        self._check_config()
        self._check_table_exists()
        self._check_project_path_exists()
        self._check_thresholds()
        self._check_env_package()
        self._check_steps()
        self._check_sample_cols()

        # 2. 初始化路径(提前创建, 以便后续步骤保存SQL)
        main_logger.info("初始化路径")
        project_path = self.config['project_path']
        log_path = os.path.join(project_path, 'logs')
        data_path = os.path.join(project_path, 'data')
        cache_path = os.path.join(project_path, 'cache')
        result_path = os.path.join(project_path, 'results')
        proc_cache_path = os.path.join(cache_path, 'Proc01Prepare')
        for p in [log_path, data_path, cache_path, result_path, proc_cache_path]:
            os.makedirs(p, exist_ok=True)

        # 3. 构建配置项元数据
        main_logger.info("构建配置项元数据")
        self._proc_cache_path = proc_cache_path
        metadata = self.cons_metadata()

        # 4. 保存特征字典和元数据到项目路径, 以便后续流程使用
        main_logger.info("保存特征字典和元数据到data路径")
        self.save_feature_dict(metadata['data_path'], metadata['feature_info_type'], partitions=metadata['feature_table_partition_type'], cache_path=proc_cache_path)
        metadata_save_path = os.path.join(metadata['data_path'], 'metadata.pkl')
        with open(metadata_save_path, 'wb') as f:
            pickle.dump(metadata, f)

        # 5. 生成特征和宽表的映射字典, 以便后续流程使用
        main_logger.info("生成特征和宽表的映射字典")
        fea_info_df = pd.read_feather(os.path.join(metadata['data_path'], 'feature_dict.feather'))
        feature_list = fea_info_df['feature_name'].tolist()
        table_fea_map = self.cons_table_fea_map(feature_list)
        table_fea_map_save_path = os.path.join(metadata['data_path'], 'table_fea_map.pkl')
        with open(table_fea_map_save_path, 'wb') as f:
            pickle.dump(table_fea_map, f)

        # 人行特征需要做特殊值替换, 单独保存人行特征
        rh_feature_list = fea_info_df[fea_info_df['category_name'].str.contains('人行')]['feature_name'].tolist()
        rh_fea_save_path = os.path.join(metadata['data_path'], 'rh_fea_list.pkl')
        with open(rh_fea_save_path, 'wb') as f:
            pickle.dump(rh_feature_list, f)
