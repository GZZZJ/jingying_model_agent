# jy-model-agent

经营场景模型本地建模 Agent。第一版目标是把复借G卡作为样板跑通：项目初始化、样本探查、样本切分、特征筛选适配、建模实验、效果评估、报告生成和全链路回溯。

详细规划见 [doc/AI经营建模Agent规划.md](doc/AI经营建模Agent规划.md)。

## 当前状态

- `agent.py`：本地 CLI 入口。
- `jingying_agent/`：通用 Agent 工具包。
- `templates/project/`：新建模型项目 workspace 的标准模板。
- `projects/2026-05-fujie-gcard-v1/`：复借G卡第一版项目 workspace。
- `vendor/feature-select-v2/scripts/code/`：已固化进本仓库的特征筛选运行代码。

复借G卡项目已经完成平台回溯结果的元数据登记：

- 样本表：`ads_app_off_feature.ds29531_backtrack_fj_gcard_model_v6_1_sample`
- 特征表：70 张
- 候选特征字段：15,028 个
- 特征清单：`projects/2026-05-fujie-gcard-v1/data/profile/feature_metadata/feature_columns.csv`

## 拉取仓库

本仓库不依赖 Git submodule，直接 clone 即可：

```bash
git clone git@gitlab.caijj.net:risk-acquisition-member/jy-model-agent.git
```

## 常用命令

检查本地环境和关键参考文件：

```bash
python3 agent.py doctor
```

新建一个模型项目：

```bash
python3 agent.py init-project \
  --name 2026-xx-new-model \
  --display-name 新模型名称 \
  --scenario 业务场景 \
  --template generic
```

为某个项目登记一次运行：

```bash
python3 agent.py new-run \
  --project projects/2026-05-fujie-gcard-v1 \
  --step bootstrap
```

导出特征表元数据：

```bash
python3 agent.py export-feature-metadata --project projects/2026-xx-new-model
```

生成 feature-select-v2 适配配置：

```bash
python3 projects/2026-05-fujie-gcard-v1/scripts/02_feature_select.py
```

先生成分表 D01/D02 取数 SQL 给使用者确认，不拉数：

```bash
python3 agent.py run-d01-d02 --project projects/2026-xx-new-model --dry-run-sql --max-tables 1
```

确认 SQL 后执行分表 D01/D02，数据先落本地 feather：

```bash
python3 agent.py run-d01-d02 --project projects/2026-xx-new-model --refresh-dp-cache --sql-approved
```

生成 D01/D02 后宽表 SQL：

```bash
python3 agent.py build-wide-sql --project projects/2026-xx-new-model
```

先生成宽表后收敛取数 SQL 给使用者确认，不拉数：

```bash
python3 agent.py refine-features --project projects/2026-xx-new-model --dry-run-sql
```

确认 SQL 后执行全局相关性、随机噪声、空标签重要性和基线重要性筛选：

```bash
python3 agent.py refine-features --project projects/2026-xx-new-model --refresh-dp-cache --sql-approved
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
