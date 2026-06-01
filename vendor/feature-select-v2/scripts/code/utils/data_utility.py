import os
import time
import pickle
import tempfile
import logging
from tmlpatch.database import TMLSQLClient



def str_format(val):
    if isinstance(val, str):
        return '"{}"'.format(val)
    else:
        return str(val)


def show_table_dp(client, kw, prec_mode=False, database=None):
    '''
    根据关键字查询表列表, prec_mode为True时精确匹配, 否则模糊匹配
    '''
    if prec_mode:
        sql = "show tables like '{}'".format(kw)
    else:
        sql = "show tables like '*{}*'".format(kw)

    if database is not None:
        df = client.sql(sql, project=database).to_pandas()
    else:
        df = client.sql(sql).to_pandas()

    df = df[df.result != '']

    if df is None or df.empty:
        table_list = []
    elif len(df.result.dropna()) == 0:
        table_list = []
    else:
        table_list = list(df.result.dropna().str.split(':', expand=True)[1])

        if database is not None:
            table_list = ['{}.{}'.format(database, i) for i in table_list]

    return table_list


def desc_table_dp(client, table):
    '''
    解析表结构
    '''
    sql = "desc {}".format(table)
    df = client.sql(sql).to_pandas()
    all_info = list(df.result.dropna())
    all_info = ['{}|'.format(i) if not i.endswith('|') else i for i in all_info]
    struct_idx = [idx for idx, itm in enumerate(all_info) if 'Native Columns' in itm][0] + 4
    parti_idx = [idx for idx, itm in enumerate(all_info) if 'Partition Columns' in itm]
    end_idx = len(all_info) - 1

    if parti_idx:
        parti_idx = parti_idx[0] + 2
        struct_end_idx = [idx for idx, itm in enumerate(all_info) if itm.startswith('+---') and idx < parti_idx and idx >= struct_idx][0]
    else:
        parti_idx = end_idx
        struct_end_idx = end_idx

    # field, type, label, comment
    col_info = [itm for idx, itm in enumerate(all_info) if idx >= struct_idx and idx < struct_end_idx]
    col_info = [list(map(lambda x: x.strip(), i.split('|')[1:-1])) for i in col_info]

    parti_info = [itm for idx, itm in enumerate(all_info) if idx >= parti_idx and idx < end_idx]
    parti_info = [list(map(lambda x: x.strip(), i.split('|')[1:-1])) for i in parti_info]

    col_info = [i for i in col_info if len(i)>0]
    parti_info = [i for i in parti_info if len(i)>0]
    
    return col_info, parti_info


def batch_insert_easy(client, table_dict, save_final_table, create=True, parti_col=['ds']):
    '''
    把指定的映射表插入分区表(单分区)

    :param table_dict: dict, 映射表, 
        如: {('dev',): 'vdm_risk_jupyter.model_result_dev', ('oot',): 'vdm_risk_jupyter.model_result_oot'}
        如: {('20250131',): 'vdm_risk_jupyter.dz_v18_preselect_001__20250131', ('20250228',): 'vdm_risk_jupyter.dz_v18_preselect_001__20250228'}
    :param save_final_table: str, 最终保存的分区表名, 如: vdm_risk_jupyter.dz_v18_preselect_001
    :param create: bool, 是否创建分区表再insert, 否则直接insert
    :param parti_col: tuple/list, 分区表的分区字段名
    '''
    for idx, itm in enumerate(table_dict.items()):
        table_key, table = itm

        print("===============【 insert table key: {}, name: {} 】===============".format(table_key, table))

        col_info, parti_info = desc_table_dp(client, table)
        col_info = [i for i in col_info if i not in parti_info and i[0] not in parti_col]
        if idx == 0:
            init_col_info = col_info.copy()
            init_parti_info = parti_info.copy()
            init_table = table

            if create:
                print("create table:")
                create_sql = """
                create table if not exists {save_final_table_name}
                (
                    {fea_seq}
                )
                partitioned by ({parti_col})
                """
                # create partition table
                client.sql(create_sql.format(save_final_table_name=save_final_table,
                                    fea_seq=',\n'.join(map(lambda x: "{} {}".format(x[0], x[1]), col_info)),
                                    parti_col=', '.join(['{} string'.format(i) for i in parti_col])))

        if col_info != init_col_info or parti_info != init_parti_info:
            raise Exception("table struct not match! init table: {}, now table: {}".format(init_table, table))

        # insert partition
        insert_sql = """
        insert overwrite table {save_final_table_name} partition({parti_col})
        select
            {fea_seq}
        from {final_table_name}
        """
        client.sql(insert_sql.format(save_final_table_name=save_final_table,
                            fea_seq=',\n'.join([i[0] for i in col_info]),
                            final_table_name=table,
                            parti_col=', '.join(['{}={}'.format(itm, str_format(table_key[idx])) for idx, itm in enumerate(parti_col)])
                            ))
        print("insert success!")


def merge_table_set(table_remain_fea, max_fea_num=800, max_table_num=10, prefix='preselect_merge_table_all', project=None):
    '''
    根据特征数和宽表数上限，将多张宽表的剩余特征分组合并为多个结果表

    分组算法:
        1. 超标表(特征数 > max_fea_num)独占一组
        2. 正常表用贪心策略按顺序填充, 同时受 max_fea_num(特征数硬上限) 和
           max_table_num(宽表数上限) 两个约束

    确定性说明:
        分组结果(merge_set)由输入数据和参数唯一确定, 相同输入多次调用分组一致。
        但生成的表名包含调用时刻的时间戳(精确到秒), 因此不同时刻调用会产生不同表名。
        断点续跑时应从缓存(merge_plan.pkl)恢复, 而非重新调用本函数, 否则表名不一致。

    :param table_remain_fea: dict, 每个宽表的剩余特征 {宽表名: [特征列表]}
    :param max_fea_num: int, 每个拼接表的特征数上限
    :param max_table_num: int, 每个拼接表的特征宽表数上限
    :param prefix: str, 每个拼接表的表名前缀
    :param project: str, 数据库项目空间名称，默认 None 表示不加项目名前缀
    :return: (merge_set, merge_table_map, merge_table_fea_map)
        - merge_set: list of list, 每组包含的宽表列表(确定性, 仅由输入决定)
        - merge_table_map: dict, {结果表名: [宽表列表]}(表名含时间戳, 非确定性)
        - merge_table_fea_map: dict, {结果表名: [全部特征列表]}(表名含时间戳, 非确定性)
    '''
    merge_set = list()

    # 过滤空表，分离超标单表
    normal_tables = []  # (table, fea_count)
    oversized_tables = []  # 单表超过 max_fea_num 的独占一组
    for table, fea_list in table_remain_fea.items():
        n = len(fea_list)
        if n == 0:
            continue
        if n > max_fea_num:
            oversized_tables.append(table)
            print(f"warning: 表 {table} 特征数 {n} 超过上限 {max_fea_num}，独占一组")
        else:
            normal_tables.append((table, n))

    # 超标表各自独占一组
    for t in oversized_tables:
        merge_set.append([t])

    # 计算正常表的理论组数（取 max_fea_num 和 max_table_num 两个约束的较大值）
    if normal_tables:
        total_fea = sum(n for _, n in normal_tables)
        total_table = len(normal_tables)
        min_groups_by_fea = max(1, -(-total_fea // max_fea_num))  # 向上取整
        min_groups_by_table = max(1, -(-total_table // max_table_num))
        num_groups = max(min_groups_by_fea, min_groups_by_table)
        target_fea_per_group = total_fea / num_groups

        # 贪心填充，以 target 为软目标，max 为硬上限
        cur_set = []
        cur_fea_num = 0
        for table, n in normal_tables:
            would_exceed_hard = (cur_fea_num + n) > max_fea_num
            would_exceed_table = len(cur_set) >= max_table_num
            reached_target = cur_fea_num >= target_fea_per_group and cur_set

            if cur_set and (would_exceed_hard or would_exceed_table or reached_target):
                merge_set.append(cur_set)
                cur_set = []
                cur_fea_num = 0

            cur_set.append(table)
            cur_fea_num += n

        if cur_set:
            merge_set.append(cur_set)

    import time as _time
    ts = _time.strftime('%Y%m%d%H%M%S')
    table_name_prefix = f'{project}.{prefix}' if project else prefix
    merge_table_map = {f'{table_name_prefix}_{ts}_{idx:03d}': table_list for idx, table_list in enumerate(merge_set)}

    merge_table_fea_map = dict()
    for merge_table_name, table_set in merge_table_map.items():
        all_fea_cols = [fea for table in table_set for fea in table_remain_fea[table]]
        merge_table_fea_map[merge_table_name] = all_fea_cols

    return merge_set, merge_table_map, merge_table_fea_map


def safe_sql_execute(client, sql, logger=None, desc="", max_retries=1, project=None):
    """
    安全执行 SQL 并返回 pandas DataFrame，带异常捕获和重试。

    :param client: TMLSQLClient 实例
    :param sql: SQL 字符串
    :param logger: 日志记录器，为 None 时使用 logging.getLogger(__name__)
    :param desc: SQL 描述信息，用于日志
    :param max_retries: 最大重试次数（默认 1，即失败后重试 1 次）
    :param project: 项目空间名称，传递给 client.sql() 的 project 参数，默认 None
    :return: pandas DataFrame
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    sql_kwargs = {'project': project} if project is not None else {}

    for attempt in range(max_retries + 1):
        try:
            result = client.sql(sql, **sql_kwargs)
            if result is not None:
                df = result.to_pandas()
            else:
                df = None
            return df
        except Exception as e:
            sql_preview = sql[:500].replace('\n', ' ')
            if attempt < max_retries:
                logger.warning(
                    f"[{desc}] SQL 执行失败 (第{attempt+1}次), 将重试. "
                    f"SQL前500字符: {sql_preview} | 错误: {e}"
                )
                try:
                    client.stop()
                except Exception:
                    pass
                time.sleep(3)
                client = TMLSQLClient()
            else:
                logger.error(
                    f"[{desc}] SQL 执行最终失败 (已重试{max_retries}次). "
                    f"SQL前500字符: {sql_preview} | 错误: {e}"
                )
                raise


def safe_pickle_dump(data, filepath):
    """
    原子性写入 pickle 文件：先写临时文件，flush + fsync 后 rename 替换。

    :param data: 要序列化的数据
    :param filepath: 目标文件路径
    """
    dir_path = os.path.dirname(filepath)
    fd, tmp_path = tempfile.mkstemp(suffix='.tmp', dir=dir_path)
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def cons_join_sql(table_fea_map, sample_table, id_col, target_col,
                  tw_col_or_ins_oos_col, dev_tw_filter=None,
                  random_num=5000000, random_seed=42,
                  sample_partition=None, bigtable_partition=None,
                  rh_feature_list=None, extra_where='',
                  bigtable_ds_range=None):
    """
    生成多表 LEFT JOIN 的一条 SQL，在 SQL 引擎侧完成宽表拼接。

    通过参数差异控制三种场景:
    - d03_d06: tw_col_or_ins_oos_col 传 [target_col, ins_oos_col], dev_tw_filter=(tw_col, dev_tw_list)
    - d07_d08: tw_col_or_ins_oos_col 传 [target_col, tw_col], dev_tw_filter=None (全窗口)
    - summary: tw_col_or_ins_oos_col 传 [target_col, tw_col], dev_tw_filter=None (全窗口)

    :param table_fea_map: dict, {表名: [特征列表]}
    :param sample_table: str, 样本表名
    :param id_col: str 或 list, 主键列
    :param target_col: str, Y标签字段
    :param tw_col_or_ins_oos_col: list[str], 需要 select 的额外列(如 [ins_oos_col] 或 [tw_col])
    :param dev_tw_filter: tuple (tw_col, dev_tw_list) 或 None, 非 None 时添加 tw 过滤
    :param random_num: int, 抽样数上限
    :param random_seed: int, 随机种子
    :param sample_partition: list 或 None, 样本表分区字段
    :param bigtable_partition: list 或 None, 宽表分区字段
    :param rh_feature_list: list 或 None (预留, 暂不使用)
    :param bigtable_ds_range: list 或 None, 宽表分区范围 [min, max], None 时用 ds IS NOT NULL
    :return: SQL 字符串
    """
    id_col_list = id_col if isinstance(id_col, list) else [id_col]
    id_col_str = ','.join(id_col_list)
    main_id_col = ','.join([f'a.{c}' for c in id_col_list])
    sample_ds_info = 'and {} is not null'.format(sample_partition[0]) if sample_partition else ''
    if bigtable_ds_range and bigtable_partition:
        bigtable_ds_info = f'where {bigtable_partition[0]} >= "{bigtable_ds_range[0]}" and {bigtable_partition[0]} <= "{bigtable_ds_range[1]}"'
    elif bigtable_partition:
        bigtable_ds_info = f'where {bigtable_partition[0]} is not null'
    else:
        bigtable_ds_info = ''

    # 构建 sample 子查询的 SELECT 和 WHERE
    extra_cols_str = ', '.join([f'{c}' for c in tw_col_or_ins_oos_col])
    extra_select_str = ', '.join([f'a.{c}' for c in tw_col_or_ins_oos_col])

    dev_tw_where = ''
    if dev_tw_filter is not None:
        tw_col, dev_tw_list = dev_tw_filter
        dev_tw_values = ','.join([f'"{v}"' if isinstance(v, str) else str(v) for v in dev_tw_list])
        dev_tw_where = f'and {tw_col} in ({dev_tw_values})'

    all_fea_cols = []
    join_sql_parts = []

    for idx, (table, fea_list) in enumerate(table_fea_map.items()):
        fea_list = list(fea_list) if not isinstance(fea_list, list) else fea_list
        if not fea_list:
            continue
        alias = f't{idx}'
        all_fea_cols.extend([f'{alias}.{fea}' for fea in fea_list])
        fea_cast = ','.join([f'cast({fea} as float) as {fea}' for fea in fea_list])
        id_join = ' and '.join([f'a.{c} = {alias}.{c}' for c in id_col_list])
        join_part = f"""
        left join (
            select * from (
                select {id_col_str}, {fea_cast},
                    row_number() over(partition by {id_col_str} order by rand(10) asc) as rn
                from {table}
                {bigtable_ds_info}
            ) as tt{idx} where tt{idx}.rn = 1
        ) as {alias} on {id_join}"""
        join_sql_parts.append(join_part)

    sql = f"""
        select {main_id_col}, a.{target_col}, {extra_select_str},
            {','.join(all_fea_cols)}
        from (
            select {id_col_str}, {target_col}, {extra_cols_str}
            from {sample_table}
            where {target_col} in (0, 1)
            {dev_tw_where}
            {extra_where}
            {sample_ds_info}
            order by rand({random_seed})
            limit {random_num}
        ) as a
        {''.join(join_sql_parts)}
        """
    return sql
