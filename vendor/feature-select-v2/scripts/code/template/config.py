config = {
    'project_name': 'test_20260317_001',
    'sample': {
        'table': 'vdm_risk_jupyter.dz_trade_sample_20250701_20251031', # 样本表/本地样本文件路径
        'id_col': ['ord_no'], # 样本主键, 可以是复合主键, 比如['uid', 'mdl_dte']
        'target_col': 'is_stg6_30_yq', # Y标签字段, 取值0和1的样本是有风险表现的样本, 0表示未逾期样本, 1表示逾期样本
        'tw_col': 'tw_flag', # DEV/OOT样本划分逻辑
        'time_col': 'crt_date', # 样本时间字段
        'period_col': 'crt_month', # 样本窗口字段/样本窗口划分逻辑, 计算PSI时用于划分实验, 如果是划分逻辑, 比如: substr(crt_date, 1, 7)
        'ins_oos_col': 'sample_split_73_tag1', # INS和OOS标签字段, 重要性等筛选方法训练模型时用于样本隔离
    },
    'thresholds': {
        'iv': 0.005, # iv阈值, iv小于该值的特征会被剔除
        'empty': 0.97, # 缺失率阈值, 缺失率大于该值的特征会被剔除
        'corr': 0.90, # 相关性阈值, 相关性高于该值的特征会剔除iv较低的
        'psi': 0.05, # psi阈值, 实验组所有period的最大psi高于该值的特征会被剔除
    },
    'bigtable': [
        'ads_app_off_feature.ds25829_backtrack_trade_2507_2602_dwa_risk_dz_model_final_7fst_9all_10cur_11lst_orders_info_df_feature',
        'ads_app_off_feature.ds25829_backtrack_trade_2507_2602_pdm_risk_dz_daily_br_max_feature_feature',
        'ads_app_off_feature.ds25829_backtrack_trade_2507_2602_dwa_risk_dz_model_final_16draw_repay_crossed_df_feature',
    ], # 特征宽表
    # 'bigtable_ds_range': ['2024-01-01', '2025-06-30'],  # 可选, 特征宽表分区范围, 不配则用 ds IS NOT NULL(全量扫描)
    'feature_info': 'pdm_risk.dz_feature_comment_df', # 所有特征字典, 本地csv或者数据表, 必须包含feature_name/feature_comment/category_name字段
    'project_path': '/Users/zhouzhihua/模型/贷中/96_ai相关/特征筛选/测试/result', # 输出位置
    'steps': ['d01', 'd02', 'd03', 'd05', 'd07', 'd08'], # 筛选步骤
    'train_baseline_model': True, # 是否训练基线模型(使用筛选后的全部特征)
    # 'params': {  # 可选, 算法参数覆盖(只需写想修改的参数, 未指定的使用默认值)
    #     'D01_PARAMS': {'random_num': 1000000},
    #     'D03_D06_PARAMS': {'num_boost_round': 500},
    #     # 可覆盖的参数组: D01_PARAMS, D01_MERGE_PARAMS, D02_PARAMS, D03_D06_PARAMS,
    #     #   D07_D08_PARAMS, SUMMARY_PARAMS, DEFAULT_LGB_PARAMS, SCREENING_LGB_PARAMS
    #     # 各参数组的完整参数列表见 utils/default_params.py
    # },
}