---
name: jingying-modeling-workbench
description: Use this skill for local business modeling workflows.
---

# Jingying Modeling Workbench

Use the `jm` CLI and treat `run_state.yml` plus `audit/artifact_manifest.json`
as the source of truth.

For request-driven work, start with:

- `jm request validate --project <project> --request <request.md>`
- `jm plan create --project <project> --request <request.md>`

The request Markdown is the task contract. The generated execution plan is the
task list to run and audit.

Non-technical users can create that Markdown with
`tools/model_request_builder/index.html`.
