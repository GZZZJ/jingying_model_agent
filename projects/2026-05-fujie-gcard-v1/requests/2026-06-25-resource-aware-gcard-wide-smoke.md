---
request_id: "2026-06-25-resource-aware-gcard-wide-smoke"
title: "GCard wide table resource-aware smoke"
project: "2026-05-fujie-gcard-v1"
workflow: full_modeling
task_mode: "完整建模"
owner: "guzijun"
business_domain: "inloan_operation"
scenario_profile: "fujie_gcard_main_lgbm"

data_source_mode: remote_table
sample_location: pdm_risk.pdm_risk_fujie_gcard_d01_d02_wide_feature_v6_1
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
    - refine
  require_sql_approval: true

experiments:
  - name: baseline_all
    method: xgboost
    segment: all

evaluation:
  metrics:
    - auc
    - ks
  champions:
    - gcard_v2

reports:
  outputs:
    - model_report.md
---

# 建模目标

使用复借 G 卡 D01/D02 宽表做资源感知全流程 smoke test。

# 执行边界

本 smoke test 只验证 request、plan、runtime config、资源估算、采样计划、批次计划、
SQL evidence 和精筛 dry-run SQL。真实远端 profile、DP 拉数、训练、评估和报告执行
必须在 SQL 审批和拉数引擎可用后另行触发。
