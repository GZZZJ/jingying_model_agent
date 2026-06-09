# Request-Driven Workflow

Users can provide a Markdown model request with YAML front matter. The request
acts as the task contract for Codex or Claude Code.

For non-technical users, use the static request builder:

```bash
open tools/model_request_builder/index.html
```

The page lets users select sample, split, feature-selection, modeling,
evaluation, risk-profile, and report requirements, then download Markdown.

Standard flow:

1. Validate the project config.
2. Validate the model request.
3. Generate `execution_plan.yml`.
4. Initialize a run and copy the request plus plan into the run workspace.
5. Execute tasks under `runs/<run_id>/tasks/`.
6. Register artifacts and decisions.
7. Record missing reusable capabilities in `audit/improvement_candidates.md`.

The Skill explains how the Agent should decide and recover. The CLI provides
stable atomic actions. The run workspace remains the source of truth.

Recommended commands:

```bash
jm project validate --project <project>
jm request validate --project <project> --request <request.md>
jm plan create --project <project> --request <request.md>
jm run init --project <project> --workflow full_modeling --request <request.md> --plan <execution_plan.yml>
jm run status --project <project> --run-id <run_id>
```

After a real project finishes, review its run workspace before changing the
generic workbench:

1. Keep one-off business assumptions in the request or project config.
2. Promote repeated code into a CLI command or shared module.
3. Add or update tests for the promoted behavior.
4. Import historical artifacts into a standard run if they came from an older
   layout.

For the current Fujie GCard baseline, historical outputs can be normalized with:

```bash
jm run import-gcard-model-artifacts \
  --project projects/2026-05-fujie-gcard-v1 \
  --run-id 2026-06-imported-gcard-main-lgbm
```
