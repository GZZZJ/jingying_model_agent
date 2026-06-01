# {{display_name}} 项目 Workspace

场景：{{scenario}}

创建日期：{{created_date}}

本目录是一次独立建模项目的 workspace。所有配置、脚本、运行记录、模型、指标和报告都保存在本目录下，避免不同模型项目互相污染。

## 目录说明

- `project.yaml`：项目总索引，记录模型、样本、标签、切分、客群等项目级口径。
- `configs/`：各步骤配置。
- `scripts/`：当前项目可执行脚本。
- `queries/`：DP 探查 SQL。
- `data/`：本地样本、抽样、加工和探查结果。
- `runs/`：每次运行的 manifest、日志、快照和产物。
- `reports/`：最终报告。

## 建议执行顺序

1. 补齐 `project.yaml` 和 `configs/sample.yaml` 中的本地样本路径。
2. 使用 `queries/` 中的 SQL 复核 DP 样本表结构和分布。
3. 运行 `scripts/01_prepare_sample.py` 做本地样本探查和切分。
4. 运行 `scripts/02_feature_select.py` 生成或执行特征筛选配置。
5. 运行训练、评估和报告脚本。
