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
5. 生成 D01/D02 保留特征宽表 SQL：
   `python3 scripts/07_build_wide_feature_sql.py`
6. 从宽表抽样并收敛到 500 个以内特征：
   `python3 scripts/08_refine_wide_features.py`
7. 后续接入训练、评估和报告脚本。

## D01/D02 当前口径

- 取数条件：`rand_flag0 < 0.1 and final_flag in ('DEV','OOT')`
- D01：仅使用 `DEV`，阈值为缺失率 `0.95`、相关性 `0.80`、IV `0.005`
- D02：`DEV` 对比 `OOT`，PSI 阈值 `0.10`
- 输出目录：`runs/d01_d02_batch_select/`

## D01/D02 后宽表拼接

- 输入：`runs/d01_d02_batch_select/results/d01_d02_final_remain_features.json`
- SQL 输出：`queries/06_build_d01_d02_wide_table.sql`
- 字段映射：`runs/d01_d02_batch_select/results/d01_d02_wide_feature_map.csv`
- 默认底表：`pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_ben_v6_1`
- 默认目标表：`pdm_risk.pdm_risk_fujie_gcard_d01_d02_wide_feature_v6_1`
- Join 主键：`uid`、`mdl_dte`
- 底表和特征表子查询均过滤：`ds is not null`

当前生成器会把 70 张特征表中 D01/D02 后保留的 2,843 个特征拼成一张 MaxCompute 宽表 SQL。若不同来源表存在同名特征，生成器会自动给重复字段加来源表前缀别名，并在字段映射 CSV 中记录 `output_feature`、`source_feature`、`source_table`。

注意：`ds` 仍作为底表保留字段和过滤字段，但不再作为特征表 join key。宽表生成器会按 `uid`、`mdl_dte` 关联样本与特征表。

也可以从仓库根目录调用：

```bash
python3 agent.py build-wide-sql --project projects/2026-05-fujie-gcard-v1
```

## 后续收敛方向

- 全局相关性去重：在宽表层面对 2,843 个特征整体做相关性聚类/去重，解决当前 D01 只在单张特征表内部去相关的问题。
- D05 重要性筛选：全局去重后直接训练一个基线模型，用模型重要性、增益、覆盖度等指标筛到几百个以内。
- D03 随机重要性筛选：加入若干随机噪声特征，训练模型后剔除重要性不高于随机噪声的真实特征，用来过滤“看起来有值但实际弱于噪声”的字段。
- D04 Null Importance：打乱标签多次训练得到每个特征在无真实信号下的重要性分布，再和真实标签训练的重要性对比，保留显著高于空标签分布的特征。

## 宽表特征收敛

- 配置：`configs/refine_features.yaml`
- 脚本：`scripts/08_refine_wide_features.py`
- 宽表：`pdm_risk.pdm_risk_fujie_gcard_d01_d02_wide_feature_v6_1`
- 默认抽样：`ds is not null and final_flag in ('DEV','OOT') and ftr_30d_ord_flag in (0,1) and rand_flag0 < 0.2`
- 输出目录：`runs/feature_refine_wide/`
- 最终特征清单：`runs/feature_refine_wide/final_500_features.txt`

当前仓库不再保留历史 DP 宽表抽样结果；需要重新生成时，先执行 SQL review，再刷新本地 feather 缓存并重跑收敛流程。

### DP 取数与本地 feather 缓存

所有需要从 DP 读取数据的脚本必须先把 SQL 对应的数据落到本地 feather，再从 feather 继续处理。

- 本地数据目录：`data/local/dp_feather/`
- 目录状态：已写入仓库根目录 `.gitignore`，不提交真实样本数据。
- 元数据目录：`data/profile/dp_feather_datasets/`
- 元数据内容：数据含义、feather 存储位置、SQL、SQL hash、行列数、列名。

执行前先只检查 SQL，不拉数：

```bash
python3 scripts/08_refine_wide_features.py --dry-run-sql
```

确认 SQL 正确后，再执行 DP 拉数并刷新本地 feather：

```bash
python3 scripts/08_refine_wide_features.py --refresh-dp-cache --sql-approved
```

如果本地 feather 已存在，后续运行会直接 `read_feather`，不会重新访问 DP；需要重新取数时才加 `--refresh-dp-cache`。Codex 或其他自动化代理在自行执行任何 `TMLSQLClient` 取数前，必须先把待执行 SQL 展示给使用者确认，确认后才能带 `--sql-approved` 执行。

正式执行时会按以下顺序处理：

1. 全局相关性去重：按单变量 AUC 近似信号强弱排序，相关性超过阈值时保留得分更高的特征。
2. D03 随机重要性筛选：加入随机噪声特征，剔除重要性低于随机噪声阈值的真实特征。
3. D04 Null Importance：多轮打乱标签得到空标签重要性分布，保留真实重要性显著高于空标签分布的特征。
4. D05 基线模型重要性：训练 LightGBM 基线模型，按 gain importance 保留前 500 个特征。
