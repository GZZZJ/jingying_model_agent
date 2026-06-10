# CLAUDE.md

This repository is a local business modeling workbench. The canonical
entrypoint is the `jm` CLI.

## Current Project State

- Active project: `projects/2026-05-fujie-gcard-v1/`
- Project checkpoint: `projects/2026-05-fujie-gcard-v1/project_state.yml`
- Active run: `2026-06-imported-gcard-main-lgbm`
- Active objective: `复借G卡主模型产物标准化与连续性交接机制建设`
- Active run workflow: `imported_gcard_main_lgbm`
- Active run status: `imported`; current stage: `report`
- Current audit verdict: `open`
- Open stages in the imported run: `feature_metadata`, `d01_d02_screening`,
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
5. Run `jm project status --project projects/2026-05-fujie-gcard-v1`.
6. Run `jm run audit --project projects/2026-05-fujie-gcard-v1 --run-id <run_id>`
   before treating a stage or run as closed.

Avoid relying on conversation memory.

## Operating Rules

- Use `jm` instead of running project scripts directly unless the user
  explicitly asks for legacy-script inspection.
- Do not overwrite previous runs. Create a new `run_id` or get explicit
  approval.
- Before any DP or `TMLSQLClient` data pull, generate SQL first and require
  explicit approval before running with `--sql-approved`.
- Never modify `vendor/feature-select-v2/scripts/code/` unless explicitly
  asked.
- Never commit raw data, local feather files, model binaries, or secrets.
- New reusable workflow logic belongs under `src/jingying_model_agent/`.
  Project-specific definitions belong in project config, request Markdown, or
  the run workspace.

## Useful Commands

```bash
jm doctor
jm project validate --project projects/2026-05-fujie-gcard-v1
jm project status --project projects/2026-05-fujie-gcard-v1
jm run status --project projects/2026-05-fujie-gcard-v1 --run-id 2026-06-imported-gcard-main-lgbm
jm run audit --project projects/2026-05-fujie-gcard-v1 --run-id 2026-06-imported-gcard-main-lgbm
pytest tests -q
```

For long or noisy tasks, use project subagents:

- sample-auditor
- feature-selector
- modeling-engineer
- evaluation-reviewer
- report-writer
