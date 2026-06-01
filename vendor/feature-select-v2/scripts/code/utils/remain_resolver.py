"""
Remain Feature Resolver: 根据 steps 配置动态解析前驱步骤的剩余特征

解决硬编码依赖链的问题，使得每个步骤可以自动找到"上一个实际执行的步骤"的输出。

输出格式:
  - 表级字典: {table_name: [features]} — d01, d02 的输出
  - 扁平列表: [feature1, feature2, ...] — d03-d06, d07-d08 的输出
"""
import os
import pickle


# 规范执行顺序
CANONICAL_ORDER = ['d01', 'd02', 'd03', 'd04', 'd05', 'd06', 'd07', 'd08']


def _get_prior_steps(steps, current_step):
    """返回 steps 中在 current_step 之前的步骤（按规范顺序）"""
    if current_step in CANONICAL_ORDER:
        current_idx = CANONICAL_ORDER.index(current_step)
        return [s for s in steps if s in CANONICAL_ORDER and CANONICAL_ORDER.index(s) < current_idx]
    return list(steps)


def resolve_table_remain_fea(project_path, steps, current_step):
    """
    解析表级剩余特征 -> {table: [features]}
    优先级: d02 输出 > d01 输出 > 初始全量 table_fea_map.pkl
    """
    result_path = os.path.join(project_path, 'results')
    data_path = os.path.join(project_path, 'data')
    prior = set(_get_prior_steps(steps, current_step))

    # 优先级: d02 > d01
    if 'd02' in prior:
        path = os.path.join(result_path, 'Proc02SelectD02PSI_table_remain_fea.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return {k: list(v) for k, v in pickle.load(f).items()}

    if 'd01' in prior:
        path = os.path.join(result_path, 'Proc02SelectD01_table_remain_fea.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

    # fallback: 初始全量特征
    path = os.path.join(data_path, 'table_fea_map.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def resolve_flat_remain_features(project_path, steps, current_step):
    """
    解析扁平剩余特征 -> [features]
    优先级: d03-d06 class 输出 > 表级展平
    """
    result_path = os.path.join(project_path, 'results')
    prior = set(_get_prior_steps(steps, current_step))

    # 检查 d03-d06 class 输出
    if {'d03', 'd04', 'd05', 'd06'} & prior:
        path = os.path.join(result_path, 'Proc02SelectD03D06_remain_features.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

    # fallback: 表级展平
    table_remain = resolve_table_remain_fea(project_path, steps, current_step)
    return [fea for fea_list in table_remain.values()
            for fea in (list(fea_list) if not isinstance(fea_list, list) else fea_list)]


def resolve_merge_table_info(project_path, steps):
    """
    检查 merge 信息是否存在（仅在 d01 被执行时才可能有 merge 输出）
    返回 merge_info dict 或 None
    """
    if 'd01' not in steps:
        return None
    path = os.path.join(project_path, 'results', 'Proc02SelectD01Merge_merge_info.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None
