# 复借G卡模型报告缺失结果清单

本文件只记录当前已注册 run artifact 无法可靠还原的内容，不伪造指标。

## 补齐状态

> 生成日期：2026-06-09 | 数据来源：`runs/model_scores/scores_all_splits.feather`

### 不可补齐（3 项）

| # | 缺少字段/结果 | 原因 |
|---|---|---|
| 1 | 变量分布/分箱图 | feather 仅含模型分数和标签，不含原始特征值 |
| 2 | 变量中文描述、业务标签 | 需要业务知识和 D01/D02 特征字典 |
| 3 | MOB1/MOB3 历史风险 | 需要未来期还款数据（MOB1=表现1个月, MOB3=表现3个月），仅含30天标签 |

### 已补齐 — 本轮 model_score（8 项）

| # | 产出文件 | 内容 |
|---|---|---|
| 4 | `evaluation/decile_lift_bins.csv` | 分客群 x final_flag 十分位 lift，含 score 边界 |
| 5-6 | `evaluation/intent_zc_segment_distribution.csv` | 老户/流失户 DEV-OOS 意愿资产占比矩阵 |
| 7 | `evaluation/intent_zc_segment_ftr_rate.csv` | 老户/流失户 DEV-OOS 30天发起率矩阵 |
| 8 | `evaluation/intent_zc_segment_amount_risk.csv` | 老户/流失户 DEV-OOS 3期金额逾期率矩阵 |
| 9 | `evaluation/monthly_segment_metrics_oot_oos.csv` | 老户次新/流失户 OOT-OOS 按月 AUC/KS |
| 10 | `evaluation/segment_model_comparison.csv` | 分客群 vs 全客群同口径对比 |
| 11 | `evaluation/model_score_bin_distribution_by_month.csv` | 按月分箱占比 + PSI 组件 |

### 已补齐 — 历史版本横向对比 (6 项)

覆盖 score_version: model_score, gcard_v2, gcard_v4, gcard_v5, gcard_v6

| # | 产出文件 | 内容 |
|---|---|---|
| 12 | `evaluation/intent_zc_segment_distribution_by_version.csv` | 意愿资产占比矩阵（310 rows） |
| 13 | `evaluation/intent_zc_segment_ftr_rate_by_version.csv` | 30天发起率矩阵（310 rows） |
| 14 | `evaluation/intent_zc_segment_amount_risk_by_version.csv` | 3期金额逾期率矩阵（310 rows） |
| 15 | `evaluation/decile_lift_bins_by_version.csv` | sloping 分箱上下界（800 rows） |
| 16 | `evaluation/monthly_segment_metrics_oot_oos_by_version.csv` | OOT-OOS 按月版本横向效果（20 rows） |
| 17 | `evaluation/score_bin_distribution_by_month_by_version.csv` | 各版本稳定性分箱（400 rows） |

## 数据口径

| 概念 | 说明 |
|---|---|
| segment | `blue_customer_flag` 映射：E3→老户, E2→次新, B2→流失户；老户次新=老户∪次新 |
| intent_level | 各 segment × score_version 组合内按对应分数等频三等分 |
| zc_level | 原始字段，整数值 1-7 |
| 金额逾期率 | `sum(ovd_amt_xz_30d_3m) / sum(prc_amt_xz_30d_3m)` |
| 稳定性基线 | 各 score_version 以 2025-06 DEV 全样本固定分箱边界计算 |

## 生成脚本

| 脚本 | 产出 |
|---|---|
| `evaluation/generate_missing_results.py` | 本轮 model_score 单版本（#4-11） |
| `evaluation/generate_versioned_results.py` | 历史版本横向对比（#12-17） |
