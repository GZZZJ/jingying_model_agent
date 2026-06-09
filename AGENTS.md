# AGENTS.md

## Project Purpose

This repository is a local business modeling workbench. The canonical entrypoint
is the `jm` CLI.

## Important Directories

- `src/jingying_model_agent/`: reusable modeling workbench code.
- `projects/`: concrete modeling project workspaces.
- `projects/2026-05-fujie-gcard-v1/`: current Fujie GCard modeling project.
- `workflows/`: reusable workflow definitions.
- `templates/project/`: project workspace template.
- `vendor/feature-select-v2/`: vendored feature selection implementation; treat as read-only unless explicitly asked.
- `.agents/skills/`: Codex skills.
- `.claude/skills/` and `.claude/agents/`: Claude Code extensions.

## Commands

- Install editable package: `pip install -e ".[modeling]"`
- Check environment: `jm doctor`
- Validate project: `jm project validate --project projects/2026-05-fujie-gcard-v1`
- Show status: `jm status --project projects/2026-05-fujie-gcard-v1 --run-id <run_id>`
- Run tests: `pytest tests -q`

## Safety Rules

- Never modify `vendor/feature-select-v2/scripts/code/` unless explicitly asked.
- Never commit raw data, local feather files, model binaries, or secrets.
- Before any DP/TMLSQLClient data pull, generate SQL first and require explicit approval.
- Do not overwrite previous runs. Create a new `run_id` or require explicit approval.
- Treat `runs/<run_id>/run_state.yml` and registered artifacts as the source of truth.

## Done Means

- Relevant tests pass.
- CLI smoke command passes.
- New or changed workflow writes artifacts into `runs/<run_id>/`.
- `run_state.yml` is updated.
- The final response reports what changed and what was not verified.
