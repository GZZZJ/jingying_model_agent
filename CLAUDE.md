# CLAUDE.md

This repository is a local business modeling workbench.

Use the `jm` CLI instead of running project scripts directly.

Before continuing any modeling run:

1. Read `run_state.yml`.
2. Read `audit/artifact_manifest.json`.
3. Check the current workflow stage.
4. Avoid relying on conversation memory.

For long or noisy tasks, use project subagents:

- sample-auditor
- feature-selector
- modeling-engineer
- evaluation-reviewer
- report-writer
