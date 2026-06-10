# risk_model_workbench rename 执行准备计划

## Summary

将当前经营建模工作台的包名、文档口径和主命令逐步统一到
`risk_model_workbench` / `rmw`。本轮 Task 1 只落盘执行计划与边界，不改
remote 仓库名称，不改历史项目数据，不改 vendored feature-select 代码。

## Global Worktree Execution

- 本次 rename 采用全局 worktree 执行：每个任务在独立 git worktree / branch
  中完成，最后由集成分支统一检查冲突、回归和兼容性。
- 当前 Task 1 工作树：
  `/Users/guzijun/.config/superpowers/worktrees/jingying_model_agent/codex/rename-risk-model-workbench`。
- 任务之间默认不互相覆盖文件；若发现同一文件已有他人修改，先读 diff 并在现有
  修改上继续，不回退他人变更。
- `vendor/feature-select-v2/` 不属于本次 rename 修改范围。

## Subagent-Driven Development

- 后续实现采用 subagent-driven development：将 rename 拆成文档、Python 包名、
  CLI entrypoint、测试回归、项目模板和发布清理等小任务并行推进。
- 每个 subagent 只处理自己的任务文件集合，并在提交说明中写清兼容性影响。
- 集成前由主协调者统一运行测试、CLI smoke 和必要的 `jm` 兼容验证。

## Rename Targets

- 主 CLI 目标名为 `rmw`，代表 `risk_model_workbench`。
- Python 可复用模块目标命名为 `risk_model_workbench`。
- 用户可见文档应逐步把通用工作台称为 Risk Model Workbench。
- Fujie GCard 仍保留为案例项目、legacy import 示例和回归样例，不成为通用默认。

## Compatibility Boundaries

- `jm` 保持兼容：rename 完成后仍应作为兼容入口或别名继续工作，避免破坏已有
  项目脚本、handoff、文档和用户习惯。
- 远程仓库名称本任务不改：GitHub/origin remote、仓库目录名和外部 repo slug
  不在 Task 1 范围内重命名。
- `jm status` 等既有 legacy alias 的语义不因本计划落盘而改变。
- 已注册 run artifacts、`run_state.yml`、`artifact_manifest.json`、项目 handoff 和
  retrospectives 的历史路径不在 Task 1 中迁移。

## Proposed Task Slices

1. 计划落盘与执行准备：新增本文件，提交独立 docs commit。
2. CLI alias 层：新增 `rmw` entrypoint，保留 `jm` entrypoint 指向同一实现。
3. 包名迁移层：引入 `risk_model_workbench` 包，并提供旧
   `jingying_model_agent` import 兼容桥。
4. 文档与模板口径：README、AGENTS、skills、templates 和 workflow docs 改为
   Risk Model Workbench 优先，GCard 仅作为案例/legacy。
5. 测试与回归：覆盖 `rmw` 主命令、`jm` 兼容入口、GCard imported run 现有 smoke。
6. 集成清理：统一搜索旧命名残留，只清理安全的通用文案和导入路径，不改历史证据。

## Verification Plan

- Docs-only task verification：
  - `git diff -- docs/plans/2026-06-10-rename-risk-model-workbench.md`
  - `git status --short`
- 后续 rename 实现任务的最低验证：
  - `pytest tests -q`
  - `rmw doctor`
  - `jm doctor`
  - `rmw project validate --project projects/2026-05-fujie-gcard-v1`
  - `jm project validate --project projects/2026-05-fujie-gcard-v1`
  - `jm run audit --project projects/2026-05-fujie-gcard-v1 --run-id 2026-06-imported-gcard-main-lgbm`

## Out Of Scope For Task 1

- 不实现 `rmw` entrypoint。
- 不移动或重命名 Python 包目录。
- 不重命名 remote repository、origin URL、GitHub repo slug 或外部发布位置。
- 不修改项目 run artifacts、原始数据、模型二进制、feather 文件或 secrets。
- 不修改 `vendor/feature-select-v2/`。
