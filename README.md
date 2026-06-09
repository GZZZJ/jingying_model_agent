# jy-model-agent

经营场景模型本地建模 Agent。第一版目标是把复借G卡作为样板跑通：项目初始化、样本探查、样本切分、特征筛选适配、建模实验、效果评估、报告生成和全链路回溯。

详细规划见 [docs/legacy/AI经营建模Agent规划.md](docs/legacy/AI经营建模Agent规划.md)。

## 当前状态

- `jm`：标准 CLI 入口；`agent.py` 和 `jingying-agent` 保留兼容。
- `src/jingying_model_agent/`：通用建模工作台代码。
- `templates/project/`：新建模型项目 workspace 的标准模板。
- `projects/2026-05-fujie-gcard-v1/`：复借G卡第一版项目 workspace。
- `vendor/feature-select-v2/scripts/code/`：已固化进本仓库的特征筛选运行代码。
- `tools/model_request_builder/`：静态模型需求生成器，用户选择和填空后导出 Markdown 需求文档。

复借G卡项目已经完成平台回溯结果的元数据登记：

- 样本表：`ads_app_off_feature.ds29531_backtrack_fj_gcard_model_v6_1_sample`
- 特征表：70 张
- 候选特征字段：15,028 个
- 特征清单：`projects/2026-05-fujie-gcard-v1/data/profile/feature_metadata/feature_columns.csv`

远端真实复借 G 卡训练、评估和报告产物已经导入标准 run：

- 标准 run：`projects/2026-05-fujie-gcard-v1/runs/2026-06-imported-gcard-main-lgbm/`
- 训练产物：`modeling/main_lgbm/model.pkl`、`metrics_train_valid.json`、`feature_importance.csv`、`run_config.json`
- 评估产物：`evaluation/overall_metrics.csv`、`monthly_metrics.csv`、`segment_metrics.csv`、`decile_lift_*.csv`、`score_psi_by_month.csv`
- 报告产物：`reports/model_report.xlsx`
- 核心效果：Valid AUC `0.9363`，Valid KS `0.7356`，评估样本 `9,600,000`

## 拉取仓库

本仓库不依赖 Git submodule，直接 clone 即可：

```bash
git clone git@gitlab.caijj.net:risk-acquisition-member/jy-model-agent.git
```

## 常用命令

检查本地环境和关键参考文件：

```bash
jm doctor
```

新建一个模型项目：

```bash
jm init-project \
  --name 2026-xx-new-model \
  --display-name 新模型名称 \
  --scenario 业务场景 \
  --template generic
```

为某个项目登记一次运行：

```bash
jm run init --project projects/2026-05-fujie-gcard-v1 --workflow full_modeling
```

查看 run 状态：

```bash
jm run status \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id 2026-06-imported-gcard-main-lgbm
```

校验模型需求文档并生成执行计划：

```bash
open tools/model_request_builder/index.html

jm request validate \
  --project projects/2026-05-fujie-gcard-v1 \
  --request projects/2026-05-fujie-gcard-v1/requests/model_request_template.md

jm plan create \
  --project projects/2026-05-fujie-gcard-v1 \
  --request projects/2026-05-fujie-gcard-v1/requests/model_request_template.md
```

把需求文档和执行计划绑定到一次 run：

```bash
jm run init \
  --project projects/2026-05-fujie-gcard-v1 \
  --workflow full_modeling \
  --request projects/2026-05-fujie-gcard-v1/requests/model_request_template.md \
  --plan projects/2026-05-fujie-gcard-v1/requests/2026-06-fujie-gcard-baseline.execution_plan.yml
```

导出特征表元数据：

```bash
jm feature metadata --project projects/2026-xx-new-model --run-id <run_id>
```

生成 feature-select-v2 适配配置：

```bash
jm feature d01-d02 --project projects/2026-05-fujie-gcard-v1 --run-id <run_id> --dry-run-sql
```

先生成分表 D01/D02 取数 SQL 给使用者确认，不拉数：

```bash
jm feature d01-d02 --project projects/2026-xx-new-model --run-id <run_id> --dry-run-sql --max-tables 1
```

确认 SQL 后执行分表 D01/D02，数据先落本地 feather：

```bash
jm feature d01-d02 --project projects/2026-xx-new-model --run-id <run_id> --refresh-dp-cache --sql-approved
```

生成 D01/D02 后宽表 SQL：

```bash
jm build-wide-sql --project projects/2026-xx-new-model
```

先生成宽表后收敛取数 SQL 给使用者确认，不拉数：

```bash
jm feature refine --project projects/2026-xx-new-model --run-id <run_id> --dry-run-sql
```

确认 SQL 后执行全局相关性、随机噪声、空标签重要性和基线重要性筛选：

```bash
jm feature refine --project projects/2026-xx-new-model --run-id <run_id> --refresh-dp-cache --sql-approved
```

复借 G 卡历史项目脚本已经移到 `projects/2026-05-fujie-gcard-v1/legacy_scripts/`，仅作回溯和兼容参考。

导入真实复借 G 卡训练、评估和报告产物到标准 run：

```bash
jm run import-gcard-model-artifacts \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id 2026-06-imported-gcard-main-lgbm
```

在有本地 feather 训练数据和特征清单时执行 LightGBM 训练：

```bash
jm train \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id <run_id> \
  --experiment main_lgbm \
  --input-feather runs/modeling_input/modeling_sample.feather \
  --feature-list runs/modeling_feature_set/feature_list.txt
```

在有本地打分 feather 时执行标准评估：

```bash
jm evaluate \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id <run_id> \
  --scores-feather runs/model_scores/scores_all_splits.feather
```

从标准训练和评估产物生成报告：

```bash
jm report --project projects/2026-05-fujie-gcard-v1 --run-id <run_id>
```

## 当前筛选口径

- D01：TOAD 初筛，阈值默认缺失率 `0.95`、相关性 `0.80`、IV `0.005`，可在项目 `configs/feature_select.yaml` 覆盖。
- D02：PSI 筛选，默认阈值 `0.10`，训练/验证切分值由项目配置控制。
- 抽样：每张特征表使用 `feature_select.d01_d02.sampling.where`，宽表后收敛使用 `configs/refine_features.yaml`。
- 执行策略：每个特征组/特征表单独筛选，支持 checkpoint 跳过已完成表。

## 外部环境依赖

项目代码本身已经不依赖其他 Git 项目。实际执行取数和筛选时，运行环境仍需要安装公司内部/三方 Python 包：

- `tmlpatch`：提供 `TMLSQLClient`
- `pandas`
- `numpy`
- `toad`：D01 优先使用；缺失时可通过脚本参数 `--use-native` 使用 feature-select-v2 的 native selector

## 标准化边界

新建项目模板已包含从特征表元数据导出、分表 D01/D02、宽表 SQL、DP feather 缓存、宽表后特征收敛到筛选流程汇总的标准入口。执行 DP 取数前必须先 dry-run 展示 SQL，使用者确认后才能带 `--sql-approved` 执行。

真实项目里的临时脚本和产物应先导入到一次标准 run，再判断哪些逻辑值得固化为通用 CLI。通用逻辑放在 `src/jingying_model_agent/`，项目特定口径留在项目配置、请求文档、run workspace 或 `legacy_scripts/`。
