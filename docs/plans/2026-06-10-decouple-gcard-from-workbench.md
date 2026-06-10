# 工作台去复借 G 卡强绑定改造计划

## Summary

把 `jy-model-agent` 改造成通用经营建模工作台：复借 G 卡保留为已承接案例、legacy import 入口和回归样例，但通用模板、默认配置、执行计划、评估和报告不再默认假设存在 `gcard_v6` 或复借 G 卡业务口径。硬性约束：原 G 卡项目已能跑的流程必须继续可用，工作台已有能力不退化。

## Key Changes

- 文档口径：`README.md`、`AGENTS.md`、`CLAUDE.md`、`.agents/skills/jingying-modeling-workbench/SKILL.md` 先讲通用工作台能力，再把复借 G 卡描述为当前活跃项目/案例项目。
- 模板去 G 卡默认值：generic project 的 `evaluate.yaml` 默认只含 `model_score`；模板 SQL 移除 `gcard_v6` 聚合；G 卡字段只保留在 G 卡项目配置中。
- 配置驱动：`evaluation.score_columns`、`evaluation.score_labels`、`input.historical_score_columns` 成为历史分和展示名来源；通用默认不包含 G 卡字段。
- 执行计划：无 champion 时不再 fallback 到 `gcard_v6`；`jm compare` 支持无 champion，生成 skipped artifact 并闭合 compare 阶段。
- 报告生成：标题、score labels、score columns、report groups 从项目配置读取；G 卡项目仍输出 G 卡版本对比，通用项目不出现 G 卡文案。
- 兼容保留：`import-gcard-artifacts` / `import-gcard-model-artifacts` 保留，只在 help 和文档里标注为 legacy/example。

## Regression Guarantees

- G 卡项目继续支持：
  - `jm project validate`
  - `jm request validate`
  - `jm plan create`
  - `jm run status`
  - `jm run audit`
  - `jm report`
  - `jm run import-gcard-artifacts`
  - `jm run import-gcard-model-artifacts`
- `jm compare --champion gcard_v6` 继续可用。
- G 卡 imported run 的 Excel/Markdown/HTML 报告仍能渲染历史版本对比。
- `run_state.yml`、`artifact_manifest.json`、handoff、retrospective、lesson、audit 语义不变。

## Test Plan

- `pytest tests -q`
- `jm doctor`
- `jm project validate --project projects/2026-05-fujie-gcard-v1`
- `jm request validate --project projects/2026-05-fujie-gcard-v1 --request projects/2026-05-fujie-gcard-v1/requests/model_request_template.md`
- `jm plan create --project projects/2026-05-fujie-gcard-v1 --request projects/2026-05-fujie-gcard-v1/requests/model_request_template.md --output /tmp/gcard_plan.yml`
- `jm run audit --project projects/2026-05-fujie-gcard-v1 --run-id 2026-06-imported-gcard-main-lgbm`
- Add generic project tests proving templates, plan creation, compare, evaluation, and report do not emit `gcard_v*` by default.
- Add G 卡 regression tests proving explicit G 卡 config still emits `gcard_v2/v4/v5/v6` where expected.

## Assumptions

- 不修改 `vendor/feature-select-v2/`。
- 不做 DP 拉数，不覆盖现有 run。
- 不清理 G 卡项目目录中的 G 卡内容。
- 当前 dirty worktree 中已有 `tools/model_request_builder/app.js` 修改和若干未跟踪项目文档；执行时不覆盖这些无关改动。
