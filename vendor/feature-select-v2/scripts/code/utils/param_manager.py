"""
参数管理模块: 三层合并(默认值 → 用户覆盖 → Claude调整) + 校验 + 审计日志

使用方式:
    from utils.param_manager import resolve_params
    d01_params = resolve_params('D01_PARAMS', config)

Claude 动态调整:
    from utils.param_manager import set_claude_override
    set_claude_override(config, 'D03_D06_PARAMS', 'ins_random_num', 2000000,
                        reason="特征数超过5000, 降低INS抽样数防止内存溢出")
"""
import copy
from utils.log_config import get_main_logger
from utils import default_params as _defaults

main_logger = get_main_logger()

SOURCE_DEFAULT = 'default'
SOURCE_USER = 'user_override'
SOURCE_CLAUDE = 'claude_override'

# 参数校验规则 (只校验明显不合理的边界)
_VALIDATION_RULES = {
    'random_num': {'type': int, 'min': 1000, 'max': 50000000},
    'random_seed': {'type': (int, type(None))},
    'round_num': {'type': int, 'min': 50, 'max': 5000},
    'max_fea_num': {'type': int, 'min': 100, 'max': 5000},
    'max_table_num': {'type': int, 'min': 1, 'max': 100},
    'num_boost_round': {'type': int, 'min': 50, 'max': 10000},
    'learning_rate': {'type': float, 'min': 0.001, 'max': 1.0},
    'num_leaves': {'type': int, 'min': 2, 'max': 256},
    'max_depth': {'type': int, 'min': -1, 'max': 20},
    'min_child_samples': {'type': int, 'min': 1},
    'subsample': {'type': float, 'min': 0.1, 'max': 1.0},
    'colsample_bytree': {'type': float, 'min': 0.1, 'max': 1.0},
    'reg_alpha': {'type': float, 'min': 0.0},
    'reg_lambda': {'type': float, 'min': 0.0},
    'd03_bagging_round': {'type': int, 'min': 1, 'max': 50},
    'd03_bagging_fraction': {'type': float, 'min': 0.1, 'max': 1.0},
    'd03_thresholds': {'type': float, 'min': 0.5, 'max': 1.0},
    'd04_real_round': {'type': int, 'min': 1, 'max': 50},
    'd04_null_round': {'type': int, 'min': 10, 'max': 500},
    'score_num_nbins': {'type': int, 'min': 2, 'max': 50},
    'd07_concordance_threshold': {'type': float, 'min': 0.0, 'max': 1.0},
    'd07_range_ratio_threshold': {'type': float, 'min': 0.0},
    'd07_spearman_threshold': {'type': float, 'min': -1.0, 'max': 1.0},
    'd07_min_cnt_pct': {'type': float, 'min': 0.0, 'max': 1.0},
    'd07_require_all_windows': {'type': bool},
    'woe_method': {'type': str, 'choices': ['quantile', 'step']},
    'woe_num_nbins': {'type': int, 'min': 2, 'max': 20},
}


def _validate_value(group_name, key, value):
    """校验单个参数值的类型和范围, 不合法则抛出 ValueError"""
    rule = _VALIDATION_RULES.get(key)
    if rule is None:
        return  # 无校验规则的参数(如 list 类型的 thresholds_list)跳过

    # 类型校验
    expected_type = rule['type']
    if value is not None and not isinstance(value, expected_type):
        # int/float 互容: 允许 int 值赋给 float 参数
        if expected_type is float and isinstance(value, int):
            pass
        else:
            raise ValueError(
                f"[参数校验] {group_name}.{key}: 期望类型 {expected_type}, "
                f"实际类型 {type(value).__name__}, 值={value}"
            )

    if value is None:
        return

    # choices 校验
    if 'choices' in rule and value not in rule['choices']:
        raise ValueError(
            f"[参数校验] {group_name}.{key}: 值 '{value}' 不在允许范围 {rule['choices']} 中"
        )

    # 范围校验
    if 'min' in rule and value < rule['min']:
        raise ValueError(
            f"[参数校验] {group_name}.{key}: 值 {value} 小于最小值 {rule['min']}"
        )
    if 'max' in rule and value > rule['max']:
        raise ValueError(
            f"[参数校验] {group_name}.{key}: 值 {value} 大于最大值 {rule['max']}"
        )


def resolve_params(group_name, config):
    """
    三层合并参数: default_params → config['params'] → config['_claude_overrides']

    Args:
        group_name: 参数组名称, 如 'D01_PARAMS'
        config: 运行配置 dict

    Returns:
        合并后的参数 dict (deepcopy, 不影响原始默认值)
    """
    # 1. 从 default_params 模块取默认值
    defaults = getattr(_defaults, group_name, None)
    if defaults is None:
        raise ValueError(f"[参数] default_params 中不存在参数组: {group_name}")
    merged = copy.deepcopy(defaults)

    # 记录覆盖日志
    overrides_log = []

    # 2. 用户覆盖层
    user_overrides = config.get('params', {}).get(group_name, {})
    for key, value in user_overrides.items():
        if key not in defaults:
            raise ValueError(
                f"[参数] {group_name} 中不存在参数 '{key}', "
                f"可用参数: {list(defaults.keys())}"
            )
        _validate_value(group_name, key, value)
        old_val = merged[key]
        merged[key] = value
        overrides_log.append(f"  {key}: {old_val} → {value} (用户覆盖)")

    # 3. Claude 覆盖层
    claude_overrides = config.get('_claude_overrides', {}).get(group_name, {})
    for key, value in claude_overrides.items():
        if key not in defaults:
            raise ValueError(
                f"[参数] {group_name} 中不存在参数 '{key}', "
                f"可用参数: {list(defaults.keys())}"
            )
        _validate_value(group_name, key, value)
        old_val = merged[key]
        merged[key] = value
        overrides_log.append(f"  {key}: {old_val} → {value} (Claude调整)")

    # 4. 审计日志
    if overrides_log:
        main_logger.info(f"[参数] === {group_name} ===")
        for line in overrides_log:
            main_logger.info(f"[参数]{line}")
        main_logger.info(f"[参数] {group_name} 最终值: {merged}")
    else:
        main_logger.info(f"[参数] === {group_name} === 全部使用默认值")

    return merged


def set_claude_override(config, group_name, key, value, reason=""):
    """
    Claude 动态调整参数的便捷入口

    Args:
        config: 运行配置 dict
        group_name: 参数组名称, 如 'D03_D06_PARAMS'
        key: 参数名
        value: 新值
        reason: 调整原因(记录到日志)
    """
    # 校验参数组和 key 是否存在
    defaults = getattr(_defaults, group_name, None)
    if defaults is None:
        raise ValueError(f"[Claude调整] default_params 中不存在参数组: {group_name}")
    if key not in defaults:
        raise ValueError(
            f"[Claude调整] {group_name} 中不存在参数 '{key}', "
            f"可用参数: {list(defaults.keys())}"
        )

    # 校验值
    _validate_value(group_name, key, value)

    # 写入 config
    if '_claude_overrides' not in config:
        config['_claude_overrides'] = {}
    if group_name not in config['_claude_overrides']:
        config['_claude_overrides'][group_name] = {}
    config['_claude_overrides'][group_name][key] = value

    reason_str = f", 原因: {reason}" if reason else ""
    main_logger.info(f"[Claude调整] {group_name}.{key} = {value}{reason_str}")
