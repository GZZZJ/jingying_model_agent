---
request_id: "2026-06-fujie-gcard-baseline"
title: "复借 G 卡 baseline 建模需求"
project: "2026-05-fujie-gcard-v1"
workflow: full_modeling
owner: "牧羊咕"

target_column: ftr_30d_ord_flag
id_columns:
  - uid
  - mdl_dte
time_column: mdl_dte
period_column: ds
split_column: final_flag

sample_checks:
  - sample_check_profile
  - sample_check_stability

feature_selection:
  rounds:
    - metadata
    - prescreen
    - refine
  require_sql_approval: true

experiments:
  - name: baseline_all
    method: xgboost
    segment: all
  - name: baseline_e2e3
    method: xgboost
    segment: e2e3
  - name: baseline_b2
    method: xgboost
    segment: b2
  - name: weighted_by_segment_v1
    method: xgboost
    segment: all

evaluation:
  metrics:
    - auc
    - ks
    - decile_lift
    - ranking_inversion
  champions:
    - gcard_v2
    - gcard_v4
    - gcard_v5
    - gcard_v6
  segments:
    - all
    - e2e3
    - b2
    - e2
    - e3

# evaluate 显式子步骤（含稳定性分箱明细、意愿矩阵分客群，避免漏跑）
stage_steps:
  evaluate:
    - auc_ks
    - decile_lift
    - monthly_stability
    - score_psi
    - score_psi_bin_detail      # 稳定性·分箱 PSI component 明细（base/current 占比 + component）
    - segment_metrics
    - intent_zc_cross_risk       # 全量口径意愿×资产矩阵
    - intent_risk_segmented      # 意愿矩阵分客群（老户次新 e2e3 / 流失户 b2）
    - cross_gain_matrix
    - feature_gain_summary

reports:
  outputs:
    - model_report.md
    - model_card.md
    - executive_summary.md
---

# 建模目标

以复借 G 卡为样板，训练候选复借意愿模型，并与历史 GCard 分数进行 champion/challenger 对比。

# 样本要求

使用 `project.yml` 中定义的样本表、标签、切分字段和客群口径。真实拉数前必须先 dry-run SQL 并获得明确批准。

# 特征要求

参考已回溯的候选特征表口径，但本轮 run 的特征筛选产物必须重新生成并注册到当前 run。`vendor/feature-select-v2/scripts/code/` 视为只读。

# 实验要求

先跑全客群 baseline，再补老户次新、流失户和分客群加权实验。

# 评估要求

必须包含 DEV/OOS/OOT、by 月、by 客群、十分箱 lift、排序倒挂检查，以及 `gcard_v2/v4/v5/v6` 对比。

# 报告要求

报告必须基于 registered artifacts。缺失真实训练或评估结果时明确标记 scaffold，不得编造指标。
