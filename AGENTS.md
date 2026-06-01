# jingying_model_agent 工作约定

## 项目目标

本仓库用于沉淀经营场景模型的本地建模 Agent。第一版以复借G卡为样板，先跑通本地样本、特征筛选、建模、评估、报告和回溯闭环。

## 当前边界

- 第一版是本地建模 Agent，不默认直接连接 Dataphin 拉数或模型平台上线。
- 可以用 `mcp_dp` 做样本表结构和聚合探查。
- 特征筛选优先适配仓库内 `vendor/feature-select-v2/`，不要重写核心筛选算法。
- `vendor/feature-select-v2/scripts/code/` 是从 feature-select-v2 固化进本仓库的运行代码，默认视为只读实现；除非用户明确要求，不要修改核心算法。

## 关键文件

- `doc/AI经营建模Agent规划.md`：总体设计。
- `doc/现有经营模型梳理.md`：经营模型资产梳理。
- `doc/复借G卡模型文档.xlsx`：复借G卡历史版本参考。
- `README.md`：本仓库初始化和命令说明。
- `agent.py`：CLI 入口。
- `templates/project/`：项目 workspace 模板。
- `vendor/feature-select-v2/scripts/code/`：特征筛选运行代码。
- `projects/2026-05-fujie-gcard-v1/`：复借G卡第一版 workspace。

## 常用命令

```bash
python3 agent.py doctor
python3 agent.py new-run --project projects/2026-05-fujie-gcard-v1 --step bootstrap
python3 projects/2026-05-fujie-gcard-v1/scripts/01_prepare_sample.py
python3 projects/2026-05-fujie-gcard-v1/scripts/02_feature_select.py
```

如果需要编译检查，macOS Python 可能会尝试写入 `~/Library/Caches`。在当前沙箱里用：

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/jingying_agent_pycache python3 -m compileall agent.py jingying_agent projects/2026-05-fujie-gcard-v1/scripts
```

## 复借G卡当前已知样本表

`pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_ben_v6_1`

已知字段包括：

- `uid`
- `mdl_dte`
- `ds`
- `blue_customer_flag`
- `ftr_30d_ord_flag`
- `ftr_30d_ord_amt`
- `prc_amt_xz_30d_3m`
- `ovd_amt_xz_30d_3m`
- `final_flag`
- `zc_level`
- `gcard_v2`
- `gcard_v4`
- `gcard_v5`
- `gcard_v6`
- `rand_flag0` 至 `rand_flag5`

`final_flag` 已见取值：`DEV`、`DEV-OOS`、`OOT`、`OOT-OOS`。

客群口径：

- `B2`：流失户，结清30天+；2024.08后交易侧口径变更为主营结清30+且2个月无发起；2025-07-02变更为180天未发起。
- `E2`：次新户，首借后30-100天。
- `E3`：老户，首借后100天+。
- 历史报告中的“老户次新”先按 `E2 + E3`，流失户按 `B2`。

风险口径：

- 当前表有 `prc_amt_xz_30d_3m`、`ovd_amt_xz_30d_3m`，可先用于金额风险观察。
- 是否能复刻历史报告中的 MOB1/MOB3 人头逾期率、金额逾期率，仍需找同事确认历史计算逻辑。

特征规模：

- 后续待筛特征来自其他表。
- 候选特征可能超过 15,000 个，样本量几百万级。
- 特征筛选必须随机抽样并分批处理，同时保留每步筛选产物和配置快照。

## 下个会话优先确认

1. 本地样本和特征文件路径。
2. 抽样规则。
3. MOB1/MOB3 人头逾期率、金额逾期率历史计算逻辑。
4. 特征宽表或本地特征文件来源。
5. 特征字典表或本地特征说明文件来源。
