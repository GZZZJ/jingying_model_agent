# AGENTS.md

## Project Purpose

This repository is a local business modeling workbench. The canonical entrypoint
is the `jm` CLI.

## Current State

As of 2026-06-09:

- Active project: `projects/2026-05-fujie-gcard-v1/`
- Active run: `2026-06-imported-gcard-main-lgbm`
- Project checkpoint: `projects/2026-05-fujie-gcard-v1/project_state.yml`
- Current objective: `复借G卡主模型产物标准化与连续性交接机制建设`
- Run workflow/status: `imported_gcard_main_lgbm` / `imported`
- Run current stage: `report`
- `jm project status` reports stage counts `done=7, pending=3`.
- `jm run audit` currently reports verdict `open` because
  `feature_metadata`, `d01_d02_screening`, and `build_wide_sql` are pending,
  and completed stages are imported evidence.

The imported run contains real historical Fujie GCard model artifacts, but it is
not local end-to-end rerun evidence.

## Important Directories

- `src/jingying_model_agent/`: reusable modeling workbench code.
- `projects/`: concrete modeling project workspaces.
- `projects/2026-05-fujie-gcard-v1/`: current Fujie GCard modeling project.
- `projects/2026-05-fujie-gcard-v1/project_state.yml`: project-level
  continuity checkpoint.
- `projects/2026-05-fujie-gcard-v1/handoffs/`: explicit session handoffs.
- `projects/2026-05-fujie-gcard-v1/retrospectives/`: explicit session, stage,
  or project retrospectives.
- `projects/2026-05-fujie-gcard-v1/docs/lessons.md`: project lessons that may
  later be promoted into CLI guardrails, tests, or skills.
- `projects/2026-05-fujie-gcard-v1/runs/<run_id>/`: canonical run workspace.
- `projects/2026-05-fujie-gcard-v1/runs/<run_id>/run_state.yml`: run state
  source of truth.
- `projects/2026-05-fujie-gcard-v1/runs/<run_id>/audit/artifact_manifest.json`:
  registered artifact source of truth.
- `workflows/`: reusable workflow definitions.
- `templates/project/`: project workspace template.
- `vendor/feature-select-v2/`: vendored feature selection implementation; treat
  as read-only unless explicitly asked.
- `.agents/skills/`: Codex skills.
- `.claude/skills/` and `.claude/agents/`: Claude Code extensions.

## Commands

- Install editable package: `pip install -e ".[modeling]"`
- Check environment: `jm doctor`
- Validate project: `jm project validate --project projects/2026-05-fujie-gcard-v1`
- Show project status: `jm project status --project projects/2026-05-fujie-gcard-v1`
- Show run state: `jm run status --project projects/2026-05-fujie-gcard-v1 --run-id <run_id>`
- Audit run closure: `jm run audit --project projects/2026-05-fujie-gcard-v1 --run-id <run_id>`
- Write handoff: `jm handoff write --project projects/2026-05-fujie-gcard-v1 --run-id <run_id>`
- Write retrospective: `jm retrospective write --project projects/2026-05-fujie-gcard-v1 --run-id <run_id>`
- Add lesson: `jm lesson add --project projects/2026-05-fujie-gcard-v1 --title <title> --body <body>`
- Run tests: `pytest tests -q`

`jm status` still exists as a legacy alias for run state, but prefer
`jm run status` in documentation and handoffs.

## Workflow Rules

- Use `jm` for workflow actions. Do not run project scripts directly unless the
  user explicitly asks for legacy-script inspection or migration.
- Before resuming work, read `project_state.yml`, `run_state.yml`, and
  `audit/artifact_manifest.json`.
- Treat `runs/<run_id>/run_state.yml` and registered artifacts as source of
  truth for stage status.
- Use `jm run audit` before declaring a stage or run closed.
- `handoff write` and `retrospective write` are explicit checkpoint actions; do
  not infer session completion from conversation state.
- When a modeling request Markdown file is provided, validate it with
  `jm request validate`, create an execution plan with `jm plan create`, and
  bind the request/plan into a new run with `jm run init`.

## Safety Rules

- Never modify `vendor/feature-select-v2/scripts/code/` unless explicitly asked.
- Never commit raw data, local feather files, model binaries, or secrets.
- Before any DP or `TMLSQLClient` data pull, generate SQL first and require
  explicit approval before using `--sql-approved`.
- Do not overwrite previous runs. Create a new `run_id` or require explicit
  approval.
- Imported or scaffold artifacts are not local reproduction evidence.
- Reusable logic belongs under `src/jingying_model_agent/`; project-specific
  definitions belong in project config, request Markdown, or run workspaces.

## Done Means

- Relevant tests pass.
- CLI smoke command passes.
- New or changed workflow writes artifacts into `runs/<run_id>/`.
- `run_state.yml` and `audit/artifact_manifest.json` are updated when workflow
  artifacts change.
- `project_state.yml`, handoff, retrospective, or lessons are updated when the
  task changes project continuity state.
- The final response reports what changed and what was not verified.
