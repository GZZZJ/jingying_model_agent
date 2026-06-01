# jingying_model_agent

经营场景模型本地建模 Agent。第一版目标是把复借G卡作为样板跑通：项目初始化、样本探查、样本切分、特征筛选适配、建模实验、效果评估、报告生成和全链路回溯。

详细规划见 [doc/AI经营建模Agent规划.md](doc/AI经营建模Agent规划.md)。

## 当前初始化内容

- `agent.py`：本地 CLI 入口。
- `jingying_agent/`：通用 Agent 工具包。
- `templates/project/`：新建模型项目 workspace 的标准模板。
- `projects/2026-05-fujie-gcard-v1/`：复借G卡第一版项目 workspace。
- `my-skills/develop/feature-select-v2/`：本地特征筛选参考实现，当前 Agent 只适配调用，不改动源码。

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

## 下一步

下一会话聚焦 `projects/2026-05-fujie-gcard-v1/`，补齐：

- 本地样本和特征文件路径。
- 抽样规则。
- `blue_customer_flag` 与老户次新、流失户的映射。
- 特征宽表或本地特征文件接入。
- 第一版训练和评估脚本。
