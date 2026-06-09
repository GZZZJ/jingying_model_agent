# 复借G卡模型报告

生成日期：2026-06-09

## 一、模型描述

- 模型目标：预测观察日后 30 天内是否发起，标签字段为 `ftr_30d_ord_flag`。
- 建模样本：训练集 DEV，验证集 OOT，OOS DEV-OOS、OOT-OOS。
- 算法：lightgbm；最终入模变量 96 个；best iteration 1000。
- 验证集效果：AUC 0.936，KS 0.736，Train/Valid AUC gap 0.002。

## 二、变量筛选过程

- 当前复借G卡 Feather 主线借鉴 feature-select-v2 的随机重要性、Null Importance、Top Importance 思路，但阈值和局部实现为项目内自定义。

| 步骤 | 筛选方法 | 剩余变量个数 | 来源 |
| --- | --- | --- | --- |
| 初始 | 初始候选变量总数：70张特征表，共15,028个字段级候选变量 | 15028 | runs/d01_d02_batch_select/results/d01_d02_run_summary.json |
| 1 | 分表基础预筛：缺失率 < 0.95，相关性 < 0.80，IV >= 0.005 | 2958 | runs/d01_d02_batch_select/results/d01_d02_run_summary.json |
| 2 | 稳定性筛选：DEV vs OOT，PSI <= 0.10 | 2843 | runs/d01_d02_batch_select/results/d01_d02_run_summary.json |
| 3 | Feather观察样本可用特征：50,000行，过滤缺失率过低和常量字段 | 2825 | runs/feature_refine_feather/stage_summary.json |
| 4 | 全局相关性去重：相关性阈值 0.80，按单变量AUC保留更强特征 | 1945 | runs/feature_refine_feather/stage_summary.json |
| 5 | 随机噪声重要性筛选：3轮，5个随机噪声特征，存活率 >= 0.50 | 96 | runs/feature_refine_feather/stage_summary.json |
| 6 | 空标签重要性筛选：20轮空标签，空标签重要性75分位，score >= 1.00 | 96 | runs/feature_refine_feather/stage_summary.json |
| 7 | 基线模型重要性筛选：LightGBM gain importance，保留前500个，valid AUC=0.9292 | 96 | runs/feature_refine_feather/final_500_features.txt |

## 三、核心效果与历史版本对比

- OOT-OOS：本轮模型 AUC 0.931、KS 0.718；相对 G卡V2 KS 提升 0.019，相对 G卡V4 KS 提升 0.033，相对 G卡V6 KS 提升 0.003。

| final_flag | n_samples | positive | bad_rate | model_score_auc | model_score_ks |
| --- | --- | --- | --- | --- | --- |
| DEV | 3600000 | 550617 | 0.153 | 0.939 | 0.734 |
| DEV-OOS | 3600000 | 624900 | 0.174 | 0.933 | 0.720 |
| OOT | 1200000 | 154072 | 0.128 | 0.936 | 0.736 |
| OOT-OOS | 1200000 | 171478 | 0.143 | 0.931 | 0.718 |

| final_flag | model_score_auc | model_score_ks | ks_uplift_vs_gcard_v2 | ks_uplift_vs_gcard_v4 | ks_uplift_vs_gcard_v5 | ks_uplift_vs_gcard_v6 |
| --- | --- | --- | --- | --- | --- | --- |
| DEV | 0.939 | 0.734 | 0.021 | 0.029 | 0.027 | 0.006 |
| DEV-OOS | 0.933 | 0.720 | 0.018 | 0.029 | 0.023 | 0.003 |
| OOT | 0.936 | 0.736 | 0.017 | 0.027 | 0.022 | 0.002 |
| OOT-OOS | 0.931 | 0.718 | 0.019 | 0.033 | 0.023 | 0.003 |

## 四、分客群效果

| segment | final_flag | n_samples | bad_rate | model_score_auc | model_score_ks | ks_uplift_vs_gcard_v2 |
| --- | --- | --- | --- | --- | --- | --- |
| 全客群 | DEV | 3600000 | 0.153 | 0.939 | 0.734 | 0.021 |
| 全客群 | DEV-OOS | 3600000 | 0.174 | 0.933 | 0.720 | 0.018 |
| 全客群 | OOT | 1200000 | 0.128 | 0.936 | 0.736 | 0.017 |
| 全客群 | OOT-OOS | 1200000 | 0.143 | 0.931 | 0.718 | 0.019 |
| 老户次新 | DEV | 1184621 | 0.419 | 0.830 | 0.502 | 0.051 |
| 老户次新 | DEV-OOS | 1287620 | 0.437 | 0.823 | 0.490 | 0.041 |
| 老户次新 | OOT | 337008 | 0.402 | 0.804 | 0.456 | 0.040 |
| 老户次新 | OOT-OOS | 372417 | 0.404 | 0.802 | 0.453 | 0.041 |
| 老户 | DEV | 1057723 | 0.435 | 0.832 | 0.505 | 0.052 |
| 老户 | DEV-OOS | 1129863 | 0.457 | 0.825 | 0.492 | 0.042 |
| 老户 | OOT | 309770 | 0.411 | 0.806 | 0.459 | 0.041 |
| 老户 | OOT-OOS | 327473 | 0.417 | 0.806 | 0.458 | 0.043 |
| 次新 | DEV | 126898 | 0.285 | 0.781 | 0.421 | 0.070 |
| 次新 | DEV-OOS | 157757 | 0.297 | 0.774 | 0.406 | 0.052 |
| 次新 | OOT | 27238 | 0.302 | 0.756 | 0.378 | 0.052 |
| 次新 | OOT-OOS | 44944 | 0.310 | 0.750 | 0.376 | 0.051 |
| 流失户 | DEV | 2415379 | 0.022 | 0.879 | 0.605 | 0.018 |
| 流失户 | DEV-OOS | 2312380 | 0.027 | 0.882 | 0.613 | 0.019 |
| 流失户 | OOT | 862992 | 0.022 | 0.884 | 0.619 | 0.031 |
| 流失户 | OOT-OOS | 827583 | 0.026 | 0.893 | 0.638 | 0.040 |

## 五、模型 sloping、意愿交叉风险与稳定性

- sloping 详见 Excel 中 `模型效果-模型sloping`，已按全客群、老户次新、流失户分别横向对比本轮模型和历史 G 卡版本；累计和剩余 lift 按参考文档口径从低分尾部开始累计。
- 意愿交叉风险详见 Excel 中 `模型效果-意愿交叉风险（DEV-OOS）`。当前 artifact 缺少老户/流失户、score version、final_flag 和金额风险 x 资产评级维度，不伪造缺失矩阵。
- 本轮模型 PSI 最高的 5 个观测如下；分箱占比变化明细需补充产出后回填：
| month | psi | n_samples | score_column |
| --- | --- | --- | --- |
| 2026-01 | 0.046 | 1200000 | model_score |
| 2025-12 | 0.037 | 1200000 | model_score |
| 2025-11 | 0.033 | 1200000 | model_score |
| 2025-10 | 0.027 | 1200000 | model_score |
| 2025-09 | 0.018 | 1200000 | model_score |

## 六、重要变量

| feature | gain | split |
| --- | --- | --- |
| ord_apl_sum_prc_amt_90_day | 5426063.174 | 204 |
| unpaid_principal_future_light_add_heavy | 1985538.499 | 445 |
| ord_apl_max_prc_amt_180_day | 1820451.862 | 340 |
| d360_apl_ord_ddf_mdl_ord_crt_dte_min | 1086284.381 | 969 |
| his_360_day_csh_apl_ord_cnt | 1068670.777 | 224 |
| d180_apl_ord_days_cnt_all | 870015.297 | 220 |
| cnt_event_result_1_uid_recent_days_120 | 637491.652 | 347 |
| his_360_day_csh_apl_ord_cnt_his_rto | 289071.587 | 173 |
| cnt_other_info_cash_uid_recent_days_30 | 249303.482 | 197 |
| dau_90d | 200764.363 | 602 |
| cnt_avg_ord_span_30d | 158264.697 | 422 |
| his_180_day_csh_apl_ord_cnt_his_rto | 157546.802 | 242 |
| stg_pln_max_rep_prc_1m_his | 116375.389 | 353 |
| ddf_lst_app_str_tim_to_mdl_tim_sec | 112144.534 | 791 |
| ord_apl_sum_prc_amt_360_day | 105323.065 | 280 |

## 七、待补充事项

- 历史文档中的变量分布/分箱图、变量中文描述与业务标签、MOB1/MOB3 历史风险精确定义仍需在另一环境补充计算。
- 详见 `model_report_missing_results.md`。
