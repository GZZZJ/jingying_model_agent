# CLAUDE.md

This repository is the `risk_model_workbench` local business modeling
workbench, also named `风险场景 AI 建模工作台`. The canonical entrypoint is the
`rmw` CLI. `jm` is a long-term compatibility alias.

The workbench is generic business modeling infrastructure. Fujie GCard is the
current active case project and regression example, not the boundary of the
workbench's reusable capability.

## Current Project State

- Active case project: `projects/2026-05-fujie-gcard-v1/`
- Project checkpoint: `projects/2026-05-fujie-gcard-v1/project_state.yml`
- Active run: `2026-06-imported-gcard-main-lgbm`
- Active objective: `复借G卡主模型产物标准化与连续性交接机制建设`
- Active run workflow: `imported_gcard_main_lgbm`
- Active run status: `imported`; current stage: `feature_refine`
- Current audit verdict: `open`
- Open stages in the imported run: `feature_metadata`, `feature_prescreen`,
  `build_wide_sql`

The imported run contains real historical Fujie GCard training, evaluation, and
report artifacts, but it is not proof that the full workflow was rerun locally.
Treat imported or scaffold artifacts as evidence to review, not as local
reproduction evidence.

## Resume Checklist

Before continuing any modeling work:

1. Read `projects/2026-05-fujie-gcard-v1/project_state.yml`.
2. Read `runs/<run_id>/run_state.yml`.
3. Read `runs/<run_id>/audit/artifact_manifest.json`.
4. Review the latest handoff under `handoffs/` and retrospective under
   `retrospectives/` when present.
5. Run `rmw project status --project projects/2026-05-fujie-gcard-v1`.
6. Run `rmw run audit --project projects/2026-05-fujie-gcard-v1 --run-id <run_id>`
   before treating a stage or run as closed.

Avoid relying on conversation memory.

## Operating Rules

- Use `rmw` instead of running project scripts directly unless the user
  explicitly asks for legacy-script inspection.
- Do not overwrite previous runs. Create a new `run_id` or get explicit
  approval.
- Before any DP or `TMLSQLClient` data pull, generate SQL first and require
  explicit approval before running with `--sql-approved`.
- Never modify `vendor/feature-select-v2/scripts/code/` unless explicitly
  asked.
- Never commit raw data, local feather files, model binaries, or secrets.
- New reusable workflow logic belongs under `src/risk_model_workbench/`.
  Project-specific definitions belong in project config, request Markdown, or
  the run workspace.

## Useful Commands

```bash
rmw doctor
rmw project validate --project projects/2026-05-fujie-gcard-v1
rmw project status --project projects/2026-05-fujie-gcard-v1
rmw run status --project projects/2026-05-fujie-gcard-v1 --run-id 2026-06-imported-gcard-main-lgbm
rmw run audit --project projects/2026-05-fujie-gcard-v1 --run-id 2026-06-imported-gcard-main-lgbm
pytest tests -q
```

The same commands can still be run with `jm` for compatibility, but new docs
and handoffs should prefer `rmw`.

For long or noisy tasks, use project subagents:

- sample-auditor
- feature-selector
- modeling-engineer
- evaluation-reviewer
- report-writer
