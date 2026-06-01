# jy-model-agent

经营场景模型本地建模 Agent。第一版目标是把复借G卡作为样板跑通：项目初始化、样本探查、样本切分、特征筛选适配、建模实验、效果评估、报告生成和全链路回溯。

详细规划见 [doc/AI经营建模Agent规划.md](doc/AI经营建模Agent规划.md)。

## 当前状态

- `agent.py`：本地 CLI 入口。
- `jingying_agent/`：通用 Agent 工具包。
- `templates/project/`：新建模型项目 workspace 的标准模板。
- `projects/2026-05-fujie-gcard-v1/`：复借G卡第一版项目 workspace。
- `my-skills/`：特征筛选等本地技能仓库，以 Git submodule 方式引用。

复借G卡项目已经完成平台回溯结果的元数据登记：

- 样本表：`ads_app_off_feature.ds29531_backtrack_fj_gcard_model_v6_1_sample`
- 特征表：70 张
- 候选特征字段：15,028 个
- 特征清单：`projects/2026-05-fujie-gcard-v1/data/profile/feature_metadata/feature_columns.csv`

## 拉取仓库

本仓库包含 `my-skills` 子模块，首次拉取建议使用：

```bash
git clone --recurse-submodules git@gitlab.caijj.net:risk-acquisition-member/jy-model-agent.git
```

如果已经普通 clone：

```bash
git submodule update --init --recursive
```

## 常用命令

检查本地环境和关键参考文件：

```bash
python3 agent.py doctor
```

新建一个模型项目：

```bash
python3 agent.py init-project \
  --name 2026-05-fujie-gcard-v1 \
  --display-name 复借G卡 \
  --scenario 复借意愿 \
  --template fujie-gcard
```

为某个项目登记一次运行：

```bash
python3 agent.py new-run \
  --project projects/2026-05-fujie-gcard-v1 \
  --step bootstrap
```

导出本次回溯特征表元数据：

```bash
python3 projects/2026-05-fujie-gcard-v1/scripts/00_export_feature_metadata.py
```

生成 feature-select-v2 适配配置：

```bash
python3 projects/2026-05-fujie-gcard-v1/scripts/02_feature_select.py
```

按特征表分批执行 D01/D02 筛选：

```bash
python3 projects/2026-05-fujie-gcard-v1/scripts/06_run_d01_d02_batch_select.py
```

先试跑一张表：

```bash
python3 projects/2026-05-fujie-gcard-v1/scripts/06_run_d01_d02_batch_select.py --max-tables 1
```

## 当前筛选口径

- D01：TOAD 初筛，Y 为 `ftr_30d_ord_flag`，阈值为缺失率 `0.95`、相关性 `0.80`、IV `0.005`。
- D02：PSI 筛选，`DEV` vs `OOT`，阈值 `0.10`。
- 抽样：每张特征表使用 `rand_flag0 < 0.1 and final_flag in ('DEV','OOT')`。
- 执行策略：每个特征组/特征表单独筛选，支持 checkpoint 跳过已完成表。

## 后续待确认

- 是否将 D01/D02 结果合并成后续 D03/D05/D07/D08 的统一输入。
- MOB1/MOB3 人头逾期率、金额逾期率历史计算逻辑。
- 第一版训练、评估和报告脚本是否继续沿用当前模板口径。
