# 复借G卡 项目 Workspace

场景：复借意愿

创建日期：2026-05-29

本目录是一次独立建模项目的 workspace。所有配置、脚本、运行记录、模型、指标和报告都保存在本目录下，避免不同模型项目互相污染。

## 当前数据口径

- 样本表：`ads_app_off_feature.ds29531_backtrack_fj_gcard_model_v6_1_sample`
- 特征表：70 张，见 `configs/feature_tables.txt`
- 候选特征：15,028 个，见 `data/profile/feature_metadata/feature_columns.csv`
- 样本主键：`uid`、`mdl_dte`
- Y 标签：`ftr_30d_ord_flag`
- 样本切分：`final_flag`

## 目录说明

- `project.yaml`：项目总索引，记录模型、样本、标签、切分、客群等项目级口径。
- `configs/`：各步骤配置。
- `scripts/`：当前项目可执行脚本。
- `queries/`：DP 探查 SQL。
- `data/`：本地样本、抽样、加工和探查结果；当前已保存特征元数据，不保存真实大样本。
- `runs/`：每次运行的 manifest、日志、快照和产物。
- `reports/`：最终报告。

## 已沉淀产物

- `docs/特征表清单.md`：特征表汇总说明。
- `data/profile/feature_metadata/feature_table_summary.csv`：表级统计。
- `data/profile/feature_metadata/feature_columns.csv`：字段级特征清单。
- `data/profile/feature_metadata/feature_tables_meta.json`：完整 DP 表元数据。

## 建议执行顺序

1. 必要时重新导出元数据：
   `python3 scripts/00_export_feature_metadata.py`
2. 生成 feature-select-v2 适配配置：
   `python3 scripts/02_feature_select.py`
3. 按特征表分批执行 D01/D02：
   `python3 scripts/06_run_d01_d02_batch_select.py`
4. 先试跑一张表：
   `python3 scripts/06_run_d01_d02_batch_select.py --max-tables 1`
5. 后续接入训练、评估和报告脚本。

## D01/D02 当前口径

- 取数条件：`rand_flag0 < 0.1 and final_flag in ('DEV','OOT')`
- D01：仅使用 `DEV`，阈值为缺失率 `0.95`、相关性 `0.80`、IV `0.005`
- D02：`DEV` 对比 `OOT`，PSI 阈值 `0.10`
- 输出目录：`runs/d01_d02_batch_select/`
