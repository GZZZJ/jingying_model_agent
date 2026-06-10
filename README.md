# jy-model-agent

经营场景模型本地建模工作台。当前主线是把复借 G 卡项目沉淀成可复用的本地建模流程：项目初始化、需求校验、执行计划、样本检查、特征筛选、模型训练、评估、对比、报告生成，以及跨会话连续性交接。

标准入口是 `jm` CLI。`agent.py` 和 `jingying-agent` 仅作为兼容入口保留。

详细规划见 [docs/legacy/AI经营建模Agent规划.md](docs/legacy/AI经营建模Agent规划.md)。

## 当前状态

截至 2026-06-09：

- 通用工作台代码在 `src/jingying_model_agent/`。
- 项目模板在 `templates/project/`。
- 工作流定义在 `workflows/`。
- 当前项目是 `projects/2026-05-fujie-gcard-v1/`。
- 当前项目断点文件是 `projects/2026-05-fujie-gcard-v1/project_state.yml`。
- 当前 active run 是 `2026-06-imported-gcard-main-lgbm`。
- 当前目标是 `复借G卡主模型产物标准化与连续性交接机制建设`。
- `jm project status` 显示项目状态为 `active`，active run 的 stage counts 为 `done=7, pending=3`。
- `jm run audit` 当前 verdict 是 `open`，因为 `feature_metadata`、`d01_d02_screening`、`build_wide_sql` 仍为 pending，且已完成阶段是 imported evidence。

active run 是远端真实复借 G 卡训练、评估和报告产物导入后的标准 run。它是真实历史产物的标准化登记，不是当前本地环境端到端重跑证据。

## 复借 G 卡现状

- 样本表：`ads_app_off_feature.ds29531_backtrack_fj_gcard_model_v6_1_sample`
- 特征表：70 张，见 `projects/2026-05-fujie-gcard-v1/configs/feature_tables.txt`
- 候选特征字段：15,028 个
- 特征元数据：`projects/2026-05-fujie-gcard-v1/data/profile/feature_metadata/feature_columns.csv`
- 标准 imported run：`projects/2026-05-fujie-gcard-v1/runs/2026-06-imported-gcard-main-lgbm/`
- 当前 imported run 最终特征数：96
- 训练产物：`modeling/main_lgbm/model.pkl`、`metrics_train_valid.json`、`feature_importance.csv`、`run_config.json`
- 评估产物：`evaluation/overall_metrics.csv`、`monthly_metrics.csv`、`segment_metrics.csv`、`decile_lift_*.csv`、`score_psi_by_month.csv`
- 报告产物：`reports/model_report.xlsx`、`model_report.md`、`model_report.html`、`model_card.md`、`executive_summary.md`
- 核心效果：OOT/Valid AUC `0.9363136508774058`，OOT/Valid KS `0.7356204499371735`
- 评估样本总量：`9,600,000`

## Source Of Truth

继续任何建模任务前先看这些文件：

- `projects/2026-05-fujie-gcard-v1/project_state.yml`
- `projects/2026-05-fujie-gcard-v1/runs/<run_id>/run_state.yml`
- `projects/2026-05-fujie-gcard-v1/runs/<run_id>/audit/artifact_manifest.json`
- `projects/2026-05-fujie-gcard-v1/handoffs/` 下最新交接文档
- `projects/2026-05-fujie-gcard-v1/retrospectives/` 下最新复盘文档
- `projects/2026-05-fujie-gcard-v1/docs/lessons.md`

阶段状态以 `run_state.yml` 和 registered artifacts 为准。目录里存在但未登记到 manifest 的文件不能直接当作阶段闭环证据。

## 安装与检查

```bash
pip install -e ".[modeling]"

jm doctor
jm project validate --project projects/2026-05-fujie-gcard-v1
pytest tests -q
```

`jm doctor` 会检查规划文档、历史模型资料、vendored feature-select-v2、项目模板和核心 workflow 是否存在。

## 常用命令

查看项目连续性状态：

```bash
jm project status --project projects/2026-05-fujie-gcard-v1
```

刷新项目断点文件：

```bash
jm project status --project projects/2026-05-fujie-gcard-v1 --write-state
```

更新项目断点：

```bash
jm project update-state \
  --project projects/2026-05-fujie-gcard-v1 \
  --active-run-id 2026-06-imported-gcard-main-lgbm \
  --objective "复借G卡主模型产物标准化与连续性交接机制建设" \
  --next-action "核对 imported run 中 pending 阶段是否需要标记为 skipped/imported" \
  --risk "imported run 不是本地全链路重跑证据"
```

查看 run 状态：

```bash
jm run status \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id 2026-06-imported-gcard-main-lgbm
```

审计 run 或单个阶段是否可收尾：

```bash
jm run audit \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id 2026-06-imported-gcard-main-lgbm

jm run audit \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id 2026-06-imported-gcard-main-lgbm \
  --stage report
```

写会话交接：

```bash
jm handoff write \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id 2026-06-imported-gcard-main-lgbm
```

写显式复盘：

```bash
jm retrospective write \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id 2026-06-imported-gcard-main-lgbm \
  --scope session \
  --note "本次会话完成连续性交接能力建设"
```

记录项目经验：

```bash
jm lesson add \
  --project projects/2026-05-fujie-gcard-v1 \
  --title "SQL approval gate" \
  --kind guardrail \
  --body "Dry-run SQL must be reviewed before any DP pull."
```

`handoff write` 和 `retrospective write` 都是显式收尾动作；CLI 不猜测会话是否结束。

## 需求驱动流程

使用静态需求生成器创建需求文档：

```bash
open tools/model_request_builder/index.html
```

校验需求并生成执行计划：

```bash
jm request validate \
  --project projects/2026-05-fujie-gcard-v1 \
  --request projects/2026-05-fujie-gcard-v1/requests/model_request_template.md

jm plan create \
  --project projects/2026-05-fujie-gcard-v1 \
  --request projects/2026-05-fujie-gcard-v1/requests/model_request_template.md
```

把需求文档和执行计划绑定到一次新 run：

```bash
jm run init \
  --project projects/2026-05-fujie-gcard-v1 \
  --workflow full_modeling \
  --request projects/2026-05-fujie-gcard-v1/requests/model_request_template.md \
  --plan projects/2026-05-fujie-gcard-v1/requests/2026-06-fujie-gcard-baseline.execution_plan.yml
```

不要覆盖已有 run。需要重跑时创建新的 `run_id`，除非使用者明确批准覆盖。

## 项目与 Run 初始化

新建模型项目：

```bash
jm init-project \
  --name 2026-xx-new-model \
  --display-name 新模型名称 \
  --scenario 业务场景 \
  --template generic
```

登记一次空 run：

```bash
jm run init \
  --project projects/2026-05-fujie-gcard-v1 \
  --workflow full_modeling
```

导入真实复借 G 卡训练、评估和报告产物到标准 run：

```bash
jm run import-gcard-model-artifacts \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id 2026-06-imported-gcard-main-lgbm
```

导入命令用于标准化历史产物，不代表本地重新执行了全链路。

## 特征筛选与 SQL Gate

导出特征表元数据：

```bash
jm feature metadata --project projects/2026-05-fujie-gcard-v1 --run-id <run_id>
```

先生成分表 D01/D02 取数 SQL 给使用者确认，不拉数：

```bash
jm feature d01-d02 \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id <run_id> \
  --dry-run-sql
```

确认 SQL 后执行分表 D01/D02，数据先落本地 feather：

```bash
jm feature d01-d02 \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id <run_id> \
  --refresh-dp-cache \
  --sql-approved
```

生成 D01/D02 后宽表 SQL：

```bash
jm build-wide-sql --project projects/2026-05-fujie-gcard-v1
```

先生成宽表后收敛取数 SQL 给使用者确认，不拉数：

```bash
jm feature refine \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id <run_id> \
  --dry-run-sql
```

确认 SQL 后执行全局相关性、随机噪声、空标签重要性和基线重要性筛选：

```bash
jm feature refine \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id <run_id> \
  --refresh-dp-cache \
  --sql-approved
```

任何 DP 或 `TMLSQLClient` 取数都必须先 dry-run 展示 SQL，得到使用者明确批准后才能带 `--sql-approved` 执行。

## 训练、评估和报告

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

生成 champion/challenger 对比：

```bash
jm compare \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id <run_id> \
  --champion gcard_v6
```

从标准训练和评估产物生成报告：

```bash
jm report --project projects/2026-05-fujie-gcard-v1 --run-id <run_id>
```

如果本地 feather 训练数据或打分结果不可用，部分命令可能生成 scaffold artifact。scaffold artifact 不能当成真实建模证据。

## 当前筛选口径

- D01：TOAD 初筛，阈值默认缺失率 `0.95`、相关性 `0.80`、IV `0.005`，可在项目 `configs/feature_select.yaml` 覆盖。
- D02：PSI 筛选，默认阈值 `0.10`，训练/验证切分值由项目配置控制。
- 抽样：每张特征表使用 `feature_select.d01_d02.sampling.where`，宽表后收敛使用 `configs/refine_features.yaml`。
- 执行策略：每个特征组或特征表单独筛选，支持 checkpoint 跳过已完成表。
- 宽表 join 主键：`uid`、`mdl_dte`。
- `ds` 作为底表保留字段和过滤字段，不作为特征表 join key。

## 目录说明

- `src/jingying_model_agent/`：通用工作台模块和 CLI 实现。
- `src/jingying_agent/`、`jingying_agent/`、`agent.py`：兼容层和历史入口。
- `projects/2026-05-fujie-gcard-v1/configs/`：项目配置。
- `projects/2026-05-fujie-gcard-v1/queries/`：SQL 草稿和生成 SQL。
- `projects/2026-05-fujie-gcard-v1/runs/`：run workspace、状态、审计、模型、评估和报告产物。
- `projects/2026-05-fujie-gcard-v1/legacy_scripts/`：历史项目脚本，仅作回溯和迁移参考。
- `tools/model_request_builder/`：静态模型需求生成器。
- `vendor/feature-select-v2/`：vendored feature selection 实现。

## 外部环境依赖

项目代码本身不依赖其他 Git 项目。实际执行取数、筛选和建模时，运行环境仍可能需要公司内部或三方 Python 包：

- `tmlpatch`：提供 `TMLSQLClient`
- `pandas`
- `numpy`
- `toad`：D01 优先使用；缺失时可通过脚本参数或配置使用 native selector
- `lightgbm`
- `xgboost`
- `pyarrow`

## 标准化边界

通用逻辑放在 `src/jingying_model_agent/`。项目特定口径放在项目配置、请求文档、run workspace 或 `legacy_scripts/`。

真实项目里的临时脚本和产物应先导入到一次标准 run，再判断哪些逻辑值得固化为通用 CLI。不要直接把一次性路径、样本口径或业务假设写进通用模块。

`vendor/feature-select-v2/scripts/code/` 视为只读，除非使用者明确要求修改。

## 拉取仓库

本仓库不依赖 Git submodule，直接 clone 即可：

```bash
git clone git@gitlab.caijj.net:risk-acquisition-member/jy-model-agent.git
```
