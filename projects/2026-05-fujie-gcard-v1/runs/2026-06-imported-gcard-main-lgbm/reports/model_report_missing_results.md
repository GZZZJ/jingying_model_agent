# 复借G卡模型报告缺失结果清单

本文件只记录当前已注册 run artifact 无法可靠还原的内容，不伪造指标。

## 缺失项补齐状态

| # | 缺少字段/结果 | 期望粒度 | 建议产出文件 | 可填入目标 sheet | 状态 |
|---|---|---|---|---|---|
| 1 | 历史文档中的图片化变量分布/分箱图 | 每个重要变量、每个分箱 | `reports/variable_bin_plots/*.png` 或 `evaluation/variable_bins.csv` | `重要变量` | **不可补齐**：feather 文件仅含模型分数和标签，不含原始特征值，无法生成分箱图 |
| 2 | 变量中文描述、业务标签 | feature 级别 | `feature_selection/feature_metadata.csv`，字段建议包含 `feature,desc,label` | `重要变量` | **不可补齐**：需要业务知识和 D01/D02 阶段的特征字典，当前环境无此数据 |
| 3 | MOB1/MOB3 历史风险精确定义 | final_flag、segment、score、decile、intent_level、zc_level | `evaluation/mob_risk_metrics.csv` | `模型效果-模型sloping`、`模型效果-意愿交叉风险（DEV-OOS）` | **不可补齐**：需要未来期的还款表现数据（MOB1=表现期1个月, MOB3=表现期3个月），当前 feather 仅含30天标签 |
| 4 | sloping 分箱上下界 | `segment`、`score/version`、`decile`、`lower_bound`、`upper_bound` | `evaluation/decile_lift_bins.csv` | `模型效果-模型sloping` | **已补齐** → `evaluation/decile_lift_bins.csv` |
| 5 | 明确 DEV-OOS 过滤后的意愿资产交叉结果 | `segment in (老户, 流失户)`、`score/version`、`intent_level`、`zc_level`，限定 `final_flag=DEV-OOS` | `evaluation/intent_zc_segment_distribution.csv` | `模型效果-意愿交叉风险（DEV-OOS）` | **已补齐** → `evaluation/intent_zc_segment_distribution.csv` |
| 6 | 老户/流失户意愿资产占比矩阵 | `segment in (老户, 流失户)`、`score/version`、`intent_level`、`zc_level`，意愿按对应模型分等频三份 | `evaluation/intent_zc_segment_distribution.csv` | `模型效果-意愿交叉风险（DEV-OOS）` | **已补齐**（与 #5 合并到同一文件） |
| 7 | 老户/流失户意愿资产 30 天发起率矩阵 | `segment in (老户, 流失户)`、`score/version`、`intent_level`、`zc_level` | `evaluation/intent_zc_segment_ftr_rate.csv` | `模型效果-意愿交叉风险（DEV-OOS）` | **已补齐** → `evaluation/intent_zc_segment_ftr_rate.csv` |
| 8 | 老户/流失户意愿资产新增订单 3 期金额逾期率矩阵 | `segment in (老户, 流失户)`、`score/version`、`intent_level`、`zc_level` | `evaluation/intent_zc_segment_amount_risk.csv` | `模型效果-意愿交叉风险（DEV-OOS）` | **已补齐** → `evaluation/intent_zc_segment_amount_risk.csv` |
| 9 | 老户次新、流失户 OOT-OOS 客群 by 月模型效果 | `mdl_month`、`segment in (老户次新, 流失户)`、`final_flag=OOT-OOS` | `evaluation/monthly_segment_metrics_oot_oos.csv` | `模型效果-每月效果` | **已补齐** → `evaluation/monthly_segment_metrics_oot_oos.csv` |
| 10 | 分客群训练模型与全客群模型的同口径对比 | segment、final_flag、score、AUC、KS | `evaluation/segment_model_comparison.csv` | `模型效果-每月效果`、`模型效果-模型sloping` | **已补齐** → `evaluation/segment_model_comparison.csv` |
| 11 | 本轮模型分箱稳定性明细 | `score_column=model_score`、`month`、`score_bin/decile`；PSI分箱占比 | `evaluation/model_score_bin_distribution_by_month.csv` | `模型稳定性` | **已补齐** → `evaluation/model_score_bin_distribution_by_month.csv` |

## 补齐说明

- **补齐时间**：2026-06-09
- **数据来源**：`runs/model_scores/scores_all_splits.feather`（960万行，含 model_score、final_flag、blue_customer_flag、zc_level、label 等字段）
- **生成脚本**：`evaluation/generate_missing_results.py`
- **补齐文件**：7 个 CSV 文件，覆盖 items 4-11
- **不可补齐**：items 1-3 需要原始特征值、业务字典、MOB 还款数据，这些在当前环境中不可获取

## 数据口径说明

| 概念 | 数据来源 | 说明 |
|---|---|---|
| segment（客群） | `blue_customer_flag` 映射：E3→老户, E2→次新, B2→流失户 | 老户次新 = 老户 ∪ 次新 |
| intent_level（意愿等级） | `model_score` 等频三等分（per segment） | 低意愿/中意愿/高意愿 |
| zc_level（资产等级） | 原始 `zc_level` 字段 | 整数值 1-7 |
| final_flag（样本划分） | 原始 `final_flag` 字段 | DEV / DEV-OOS / OOT / OOT-OOS |
| mdl_month（观察月） | `mdl_dte` 转换 | 2025-06 至 2026-01，共 8 个月 |

补齐这些文件后，应继续通过 `jm report` 统一生成报告，避免人工改写 xlsx 中的计算结果。
