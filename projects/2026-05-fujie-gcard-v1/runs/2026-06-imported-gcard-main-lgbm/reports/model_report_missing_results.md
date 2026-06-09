# 复借G卡模型报告缺失结果清单

本文件只记录当前已注册 run artifact 无法可靠还原的内容，不伪造指标。

| 缺少字段/结果 | 期望粒度 | 建议产出文件 | 可填入目标 sheet |
|---|---|---|---|
| 历史文档中的图片化变量分布/分箱图 | 每个重要变量、每个分箱 | `reports/variable_bin_plots/*.png` 或 `evaluation/variable_bins.csv` | `重要变量` |
| 变量中文描述、业务标签 | feature 级别 | `feature_selection/feature_metadata.csv`，字段建议包含 `feature,desc,label` | `重要变量` |
| MOB1/MOB3 历史风险精确定义 | final_flag、segment、score、decile、intent_level、zc_level | `evaluation/mob_risk_metrics.csv` | `模型效果-模型sloping`、`模型效果-意愿交叉风险（DEV-OOS）` |
| sloping 分箱上下界 | `segment`、`score/version`、`decile`、`lower_bound`、`upper_bound`，需要能展示为 `001:(-inf, x]` 这类区间 | `evaluation/decile_lift_bins.csv`，或在各 `decile_lift_*.csv` 增加 `score_min,score_max` | `模型效果-模型sloping` |
| 明确 DEV-OOS 过滤后的意愿资产交叉结果 | `segment in (老户, 流失户)`、`score/version`、`intent_level`、`zc_level`，限定 `final_flag=DEV-OOS` | `evaluation/intent_zc_dev_oos_*.csv` | `模型效果-意愿交叉风险（DEV-OOS）` |
| 老户/流失户意愿资产占比矩阵 | `segment in (老户, 流失户)`、`score/version`、`intent_level`、`zc_level`，意愿按对应模型分等频三份；指标包含 `n_samples,sample_pct,row_pct,col_pct` 和行/列 sum | `evaluation/intent_zc_segment_distribution.csv` | `模型效果-意愿交叉风险（DEV-OOS）` |
| 老户/流失户意愿资产 30 天发起率矩阵 | `segment in (老户, 流失户)`、`score/version`、`intent_level`、`zc_level`；指标包含 `n_samples,ftr_30d_count,ftr_30d_rate` 和行/列加权整体 | `evaluation/intent_zc_segment_ftr_rate.csv` | `模型效果-意愿交叉风险（DEV-OOS）` |
| 老户/流失户意愿资产新增订单 3 期金额逾期率矩阵 | `segment in (老户, 流失户)`、`score/version`、`intent_level`、`zc_level`；指标包含 `total_principal,total_overdue,amount_overdue_rate` 和行/列加权整体 | `evaluation/intent_zc_segment_amount_risk.csv` | `模型效果-意愿交叉风险（DEV-OOS）` |
| 老户次新、流失户 OOT-OOS 客群 by 月模型效果 | `mdl_month`、`segment in (老户次新, 流失户)`、`final_flag=OOT-OOS`、`score/version`，指标包含 `n_samples, positive, bad_rate, AUC, KS` | `evaluation/monthly_segment_metrics.csv` 或 `evaluation/monthly_segment_metrics_oot_oos.csv` | `模型效果-每月效果` |
| 分客群训练模型与全客群模型的同口径对比 | segment、final_flag、score、AUC、KS、lift | `evaluation/segment_model_comparison.csv` | `模型效果-每月效果`、`模型效果-模型sloping` |
| 本轮模型分箱稳定性明细 | `score_column=model_score`、`month`、`score_bin/decile`；指标包含 `n_samples,pct,bad_rate,psi_component`，需能展示每个分箱占比随月份变化 | `evaluation/model_score_bin_distribution_by_month.csv` | `模型稳定性` |

补齐这些文件后，应继续通过 `jm report` 统一生成报告，避免人工改写 xlsx 中的计算结果。
