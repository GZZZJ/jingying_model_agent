---
name: jingying-modeling-workbench
description: Use this skill for local business modeling workflows, including sample checks, feature selection, model training, evaluation, champion/challenger comparison, and report generation.
---

# Jingying Modeling Workbench

Use the local `jm` CLI. Do not reimplement modeling logic in chat.

## Source Of Truth

For an existing run, always read:

- `projects/<project>/runs/<run_id>/run_state.yml`
- `projects/<project>/runs/<run_id>/audit/artifact_manifest.json`

## Preferred Commands

- `jm doctor`
- `jm project validate --project <project>`
- `jm request validate --project <project> --request <request.md>`
- `jm plan create --project <project> --request <request.md>`
- `jm run init --project <project> --workflow full_modeling`
- `jm run status --project <project> --run-id <run_id>`
- `jm sample check --project <project> --run-id <run_id>`
- `jm feature d01-d02 --project <project> --run-id <run_id> --dry-run-sql`
- `jm feature refine --project <project> --run-id <run_id> --dry-run-sql`
- `jm train --project <project> --run-id <run_id> --experiment main_lgbm`
- `jm evaluate --project <project> --run-id <run_id>`
- `jm compare --project <project> --run-id <run_id> --champion gcard_v6`
- `jm report --project <project> --run-id <run_id>`

For the imported Fujie GCard real-project baseline, use:

- `jm run import-gcard-model-artifacts --project projects/2026-05-fujie-gcard-v1 --run-id 2026-06-imported-gcard-main-lgbm`
- `jm run status --project projects/2026-05-fujie-gcard-v1 --run-id 2026-06-imported-gcard-main-lgbm`

## Stop Rules

Stop and report before training if sample check artifacts are missing, feature
list is missing, leakage is detected, or SQL approval is required but absent.

If local feather training data or scored predictions are unavailable, the CLI
may create scaffold artifacts. Do not treat scaffold artifacts as real evidence.
When real artifacts exist, read metrics and reports from the run workspace and
artifact manifest, not from chat memory.

## Request-Driven Workflow

When the user provides a modeling request Markdown file, treat it as the task
contract. Validate it first, generate an execution plan, initialize a run, then
execute tasks from the plan. Do not invent tasks that contradict the request.

If the user needs to create a request interactively, direct them to
`tools/model_request_builder/index.html`; the downloaded Markdown becomes the
request contract.

## Hardening Loop

When a project-specific script or notebook is useful across runs, turn the
reusable part into a CLI-backed module under `src/jingying_model_agent/`. Keep
project-specific data paths, business definitions, and one-off assumptions in
the project config, request Markdown, or run workspace. Preserve legacy scripts
under `legacy_scripts/` for traceability.
