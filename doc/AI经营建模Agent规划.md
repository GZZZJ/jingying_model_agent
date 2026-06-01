# AI经营建模Agent规划

更新时间：2026-05-29

## 1. 第一版边界

第一版 Agent 定位为本地建模 Agent。它不直接负责 Dataphin 拉数、模型平台上线或策略平台部署，主要处理已经导出到本地或已经准备好的建模样本和特征数据。

第一版要跑通的目标是复借G卡建模流程：

1. 新建独立项目 workspace。
2. 登记样本、标签、时间、客群、切分口径。
3. 做数据探查和样本质量检查。
4. 做样本抽样和 INS/OOS/OOT 划分。
5. 调用或适配 `feature-select-v2` 完成特征筛选。
6. 做全客群、分客群、加权、损失函数等建模实验。
7. 输出 AUC、KS、PSI、分箱 lift、客群切片、风险交叉等评估结果。
8. 生成建模报告。
9. 保存每个环节的脚本、配置、日志、结果和数据版本，保证可回溯。

## 2. 参考资料

当前规划依据以下资料：

- `doc/现有经营模型梳理.md`
- `doc/复借G卡模型文档.xlsx`
- `/Users/guzijun/Desktop/AI攻坚/AutoNotebook/references/base建模_精简版.ipynb`
- `vendor/feature-select-v2/`
- DP 样本表：`pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_ben_v6_1`

## 3. 核心设计原则

### 3.1 项目隔离

每次开始一个新模型项目，都在当前 Agent 目录下创建独立 workspace。项目间不共享运行产物，避免样本、配置、模型和报告混在一起。

建议目录：

```text
projects/
  2026-05-fujie-gcard-v1/
    project.yaml
    data/
      raw/
      sampled/
      processed/
      profile/
    configs/
      sample.yaml
      feature_select.yaml
      train.yaml
      evaluate.yaml
      report.yaml
    scripts/
      01_prepare_sample.py
      02_feature_select.py
      03_train.py
      04_evaluate.py
      05_report.py
    runs/
      20260529_153000/
        run.yaml
        configs_snapshot/
        scripts_snapshot/
        inputs/
        outputs/
        logs/
        metrics/
        models/
        reports/
        artifacts_manifest.json
    reports/
```

### 3.2 配置驱动

通用脚本不写死模型字段和路径。每个项目用 `project.yaml` 描述建模项目本身，用 `configs/*.yaml` 描述各步骤细节。

`project.yaml` 是项目总索引，回答这些问题：

- 这个项目是什么模型。
- 样本和标签是什么。
- 时间字段、主键字段、客群字段是什么。
- INS/OOS/OOT 如何划分。
- 要跑哪些实验。
- 报告和结果输出到哪里。

示例：

```yaml
project:
  name: fujie_gcard_v1
  display_name: 复借G卡
  scenario: 复借意愿

data:
  raw_path: data/raw/sample.feather
  id_columns: [uid, mdl_dte]
  time_column: mdl_dte
  period_column: ds
  target_column: ftr_30d_ord_flag
  label_definition: 观察日30天内是否发起
  sample_logic: 可经营、当前未逾期用户、重资产订单

split:
  source_column: final_flag
  ins_values: [DEV]
  oos_values: [DEV-OOS]
  oot_values: [OOT, OOT-OOS]

segments:
  - name: all
    filter: null
  - name: e2e3
    filter: "blue_customer_flag in ['E2', 'E3']"
  - name: b2
    filter: "blue_customer_flag == 'B2'"
```

### 3.3 全链路可回溯

每次执行生成一个 `run_id`。运行结束后，`runs/<run_id>/artifacts_manifest.json` 至少记录：

- 执行时间、执行人、执行命令。
- 输入数据路径和文件 hash。
- `project.yaml` 和所有子配置快照。
- 脚本快照或脚本 hash。
- Python 版本和关键包版本。
- 随机种子。
- 样本行数、字段数、标签浓度、时间范围。
- 特征筛选每步输入输出数量和剔除原因。
- 模型参数、入模变量、模型文件路径。
- 评估指标和报告路径。

## 4. 复借G卡样本表初步探查

通过 `mcp_dp` 对 `pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_ben_v6_1` 做了结构和聚合探查。

字段结构：

```text
uid
mdl_dte
ds
blue_customer_flag
ftr_30d_ord_flag
ftr_30d_ord_amt
prc_amt_xz_30d_3m
ovd_amt_xz_30d_3m
liushi_days
due_date_flag
final_flag
fq_diff_grp
mob_group
zc_level
gd_lmt_grp
gcard_v2
gcard_v4
gcard_v5
gcard_v6
rand_flag0
rand_flag1
rand_flag2
rand_flag3
rand_flag4
rand_flag5
```

按 `final_flag` 的样本分布：

| final_flag | mdl_dte范围 | ds范围 | 样本量 | 30天发起率 | 平均30天发起金额 | 平均gcard_v6 |
|---|---|---|---:|---:|---:|---:|
| DEV | 2025-06-02 至 2025-11-19 | 20250601 至 20251118 | 3,600,000 | 0.1529 | 2157.72 | 0.1539 |
| DEV-OOS | 2025-06-02 至 2025-11-19 | 20250601 至 20251118 | 3,600,000 | 0.1736 | 2429.10 | 0.1686 |
| OOT | 2025-12-04 至 2026-01-31 | 20251203 至 20260130 | 1,200,000 | 0.1284 | 2087.69 | 0.1320 |
| OOT-OOS | 2025-12-04 至 2026-01-31 | 20251203 至 20260130 | 1,200,000 | 0.1429 | 2236.41 | 0.1433 |

初步判断：

- 该表是复借G卡基础样本、标签、客群、历史分数和随机标识表。
- `ftr_30d_ord_flag` 可作为第一版 30 天发起 Y。
- `final_flag` 已经包含 DEV、DEV-OOS、OOT、OOT-OOS 切分。
- `blue_customer_flag` 可作为分客群建模字段，当前可见取值包括 `B2`、`E2`、`E3`。根据已补充业务口径：`B2` 为流失户，`E2` 为次新户，`E3` 为老户；历史报告中的“老户次新”可先按 `E2 + E3` 处理，“流失户”可按 `B2` 处理。
- 该表不包含大规模候选特征宽表，特征宽表或本地特征文件还需要单独接入。

待确认：

- 本地建模时使用本表导出文件，还是使用已拼接好的本地特征宽表。
- 特征宽表来源和特征字典表来源。
- `prc_amt_xz_30d_3m`、`ovd_amt_xz_30d_3m` 是否与历史报告中的 MOB1/MOB3 人头逾期率、金额逾期率同口径；历史风险口径需后续找相关同事确认。

## 5. 数据探查模块

数据探查分两类。

### 5.1 DP 表探查

用于查看样本表结构和聚合分布，不直接拉取大批明细。默认只执行：

- `limit 0` 获取字段结构。
- 按 `final_flag`、月份、客群、风险等级聚合。
- 统计样本量、Y 率、时间范围、历史分数均值。
- 检查主键重复、标签空值、切分字段缺失。

不默认拉 UID 明细。

### 5.2 本地文件探查

用于已导出的 feather、parquet、csv 文件：

- 字段类型。
- 行数和列数。
- 缺失率。
- 标签浓度。
- 时间分布。
- 客群分布。
- 数值异常。
- 主键重复。
- 与 `project.yaml` 的字段配置一致性。

输出到：

```text
data/profile/schema.json
data/profile/sample_profile.json
data/profile/data_quality_report.md
```

## 6. 样本准备和划分模块

样本准备参考 `base建模_精简版.ipynb` 的习惯：

- 样本优先落地为 feather 或 parquet。
- 先固定随机列或使用已有随机列。
- 按时间和随机列划分 INS/OOS/OOT。
- 派生统计特征只能基于 INS 计算，再映射到 OOS/OOT，避免穿越。
- 所有特征统一做数值转换，常用 `float32`。

复借G卡第一版可以直接使用 `final_flag`：

```text
INS: DEV
OOS: DEV-OOS
OOT: OOT + OOT-OOS
```

如果未来项目没有现成切分字段，则使用 `sample.yaml` 生成：

```yaml
split:
  method: time_and_random
  dev_start: 2025-06-01
  dev_end: 2025-11-30
  oot_start: 2025-12-01
  oot_end: 2026-01-31
  oos_ratio: 0.3
  random_column: rand_flag0
```

## 7. 特征筛选模块

特征筛选优先复用仓库内 `vendor/feature-select-v2/`，不重写核心算法。

### 7.1 标准步骤

推荐第一版标准精筛流程：

```text
d01: toad 初筛，缺失率、IV、相关性
d02: PSI 稳定性筛选
d03: 随机重要性筛选
d05: Top importance 截断
d07: WOE 趋势稳定性筛选
d08: WOE 解释摘要
```

全量流程可以开启：

```text
d01, d02, d03, d04, d05, d06, d07, d08
```

其中：

- d04 是 Null Importance，耗时更高。
- d06 是 SHAP importance，解释性更强但计算更慢。

### 7.2 Agent 封装方式

本 Agent 默认不修改 `vendor/feature-select-v2/scripts/code/`，只做适配层：

1. 从 `project.yaml` 和 `configs/feature_select.yaml` 生成 `feature-select-v2` 所需 config。
2. 执行 `feature-select-v2` 的 Proc。
3. 将它的结果登记到本项目的 `runs/<run_id>/artifacts_manifest.json`。
4. 将关键产物复制或索引到本项目统一目录。

`feature-select-v2` 需要的关键配置：

```python
config = {
    "project_name": "...",
    "sample": {
        "table": "...",
        "id_col": ["uid", "mdl_dte"],
        "target_col": "ftr_30d_ord_flag",
        "tw_col": "final_flag",
        "time_col": "mdl_dte",
        "period_col": "ds",
        "ins_oos_col": "final_flag",
    },
    "thresholds": {
        "iv": 0.005,
        "empty": 0.97,
        "corr": 0.90,
        "psi": 0.05,
    },
    "bigtable": [],
    "feature_info": "...",
    "project_path": "...",
    "steps": ["d01", "d02", "d03", "d05", "d07", "d08"],
    "train_baseline_model": True,
}
```

### 7.3 本地模式和 DP 表模式

由于第一版 Agent 定位为本地建模，需要保留两个执行模式：

- `local_file_mode`：样本和特征已经导出为本地文件，Agent 在本地完成筛选。
- `dp_table_mode`：样本表和特征宽表仍在 DP，Agent 生成 `feature-select-v2` 配置并调用其 SQL 取数能力。

第一版优先支持 `local_file_mode`。如果 `feature-select-v2` 的某些 Proc 只支持 DP 表，则在 Agent 里做兼容层，或者把本地文件转换为它可识别的输入格式。

特征规模约束：

- 当前基础样本表不是完整特征宽表，后续待筛特征会放在其他表中。
- 候选特征可能超过 15,000 个，样本量为几百万级。
- 特征筛选必须支持随机抽样，且每次抽样需记录随机种子、抽样字段和样本量。
- 特征筛选必须支持分批处理特征，优先复用 `feature-select-v2` 的 `round_num` 和 checkpoint 机制。
- 报告中需要保留每步筛选前后变量数、剔除原因、最终变量清单和变量重要性。

## 8. 建模实验模块

建模模块支持多实验并行登记，而不是覆盖式训练。

推荐第一版实验类型：

```text
baseline_all: 全客群统一模型
segment_e2e3: E2/E3 分客群模型
segment_b2: B2 分客群模型
weighted_v1: 样本加权模型
loss_v1: 损失函数调整模型
```

模型算法：

- 第一优先：LightGBM。
- 兼容：XGBoost。
- 基线：LogisticRegression。

训练习惯参考 notebook：

- 固定随机种子。
- OOS early stopping。
- OOS 和 OOT 同时验证。
- 保存 `model.pkl`、`feature_list.txt`、`train_config.yaml`、预测分数文件、训练日志。

## 9. 效果评估模块

第一版评估输出：

- AUC、KS。
- 月度 AUC、KS。
- OOS/OOT 对比。
- 客群切片。
- `zc_level` 风险等级切片。
- 十分箱发起率、lift、样本占比。
- PSI。
- 单调性检查。
- 新旧分数对比，如 `gcard_v2`、`gcard_v4`、`gcard_v5`、`gcard_v6`。
- 如果有风险标签，则输出意愿交叉风险和 MOB 风险表现。

输出文件：

```text
metrics/auc_ks_by_split.csv
metrics/auc_ks_by_month.csv
metrics/auc_ks_by_segment.csv
metrics/bin_lift_table.csv
metrics/psi_by_month.csv
metrics/score_compare.csv
metrics/risk_cross_table.csv
```

## 10. 报告生成模块

报告输出 Markdown 和 HTML。

建议结构：

1. 模型背景。
2. 样本定义。
3. 样本分布。
4. 数据质量。
5. 变量筛选流程。
6. 入模变量解读。
7. 建模方案。
8. 整体效果。
9. 分客群效果。
10. 分箱 lift。
11. 稳定性分析。
12. 风险交叉分析。
13. 结论和建议。
14. 附录：配置、脚本、数据版本、运行记录。

## 11. 第一阶段实施里程碑

### M1：项目骨架和配置规范

- 创建 `projects/` 标准目录。
- 定义 `project.yaml` schema。
- 定义 `sample.yaml`、`feature_select.yaml`、`train.yaml`、`evaluate.yaml`。
- 实现 run manifest 规范。

### M2：数据探查和样本准备

- 支持本地 feather/parquet/csv 探查。
- 支持 DP 样本表结构和聚合探查。
- 支持 `final_flag` 切分。
- 支持时间加随机列切分。

### M3：特征筛选适配

- 读取 `feature_select.yaml`。
- 生成 `feature-select-v2` config。
- 调用 d01、d02、d03、d05、d07、d08。
- 登记筛选结果和剔除明细。

### M4：训练和评估

- 支持 LightGBM/XGBoost baseline。
- 支持全客群和分客群实验。
- 输出 AUC、KS、PSI、分箱、单调性、客群切片。

### M5：报告和复借G卡首轮跑通

- 生成复借G卡建模报告。
- 对齐历史文档中的关键表：样本描述、重要变量、变量筛选过程、模型效果、分客群效果。
- 形成可复用模板。

## 12. 当前未决问题

1. 复借G卡第一版使用哪份本地样本或特征文件作为输入。
2. 大样本抽样规则的正式口径。
3. `blue_customer_flag` 与“老户次新”“流失户”的准确映射关系。
4. 特征宽表或本地特征文件路径。
5. 特征字典表或本地特征说明文件路径。
6. 是否必须第一版就支持 `feature-select-v2` 的 DP 表模式，还是先实现本地文件模式。
