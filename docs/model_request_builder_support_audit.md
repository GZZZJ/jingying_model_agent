# Model Request Builder Support Audit

Date: 2026-06-25

Scope:

- UI sources: `tools/model_request_builder/index.html`, `tools/model_request_builder/app.js`
- Request sources: `src/risk_model_workbench/request/`
- Planning sources: `src/risk_model_workbench/planning/`
- Execution sources: `src/risk_model_workbench/cli.py`, `workflows/*.yml`

## Verdict

The request builder can generate Markdown that the workbench parses, validates,
stores in a run, and turns into an execution plan. The workbench does not yet
fully execute every configurable UI choice as an automatic CLI behavior.

Supported means the field is either:

- executable: it changes generated tasks or a concrete CLI action;
- contract-only: it is captured in the request or plan for the Agent/user to
  follow, but does not directly change generic CLI execution;
- planned-only: it is known to the planner and preserved in plan metadata, but
  intentionally excluded from task commands until implemented;
- UI-only: it is local request-builder behavior and has no workbench runtime
  obligation.

## Support Matrix

| Builder area | User-configurable items | Current support |
| --- | --- | --- |
| Builder document actions | New blank request, apply template, save/delete custom template, restore draft, preview/copy/download Markdown | UI-only. These are local browser features and do not need `rmw` support. |
| Basic request contract | `request_id`, `title`, `owner`, `objective`, free-text body sections | Contract-only. Parsed, validated where required, copied into run via `rmw run init --request`, and available as task context. |
| Workflow / task mode | `full_modeling`, `feature_selection`, `train_baseline`, `challenger_evaluation` | Executable. `rmw request validate` checks the workflow exists, and `rmw plan create` now limits generated tasks to that workflow's stages. |
| Business domain / profile | `business_domain`, hidden `scenario_profile` | Executable for planning. Profiles resolve default `stage_steps`, `step_params`, implemented steps, and planned steps. |
| Sample contract | `sample_location`, `target_column`, `id_columns`, `time_column`, `period_column`, `split_column`, DEV/OOS/OOT values, sample definition | Partially executable. Validation checks key fields against `project.yml`; sample checks execute from project config. Request values are not automatic project-config overrides. |
| Feature rounds | metadata, prescreen, refine; Fujie profile includes wide SQL between prescreen and refine | Executable. Plan generation creates `feature_metadata`, `feature_prescreen`, `build_wide_sql`, and `feature_refine` tasks when the workflow includes those stages. |
| Wide SQL approval | SQL review gate, DP/TMLSQLClient execution approval | Partially executable. Dry-run SQL generation and explicit `--sql-approved` execution gate exist. Automated SQL quality review/blocking is planned-only. |
| Feature method switches | availability, constant-value, random-noise importance, null importance, baseline importance | Executable through current feature refinement and artifact registration, with details controlled by project config. Request-level parameter overrides are retained in plan metadata but not yet applied to the stage config. |
| Feature method switches | missing-rate, IV, correlation de-dup thresholds/method | Partial/planned. The legacy/pipeline configs contain missing, IV, and correlation behavior, and feature refinement has preprocessing/global correlation logic; the request-level switches are still marked planned or metadata-only in the generic planner. |
| Modeling experiments | experiment name, method, segment, description | Partially executable. Plan creates one `rmw train --experiment <name>` task per experiment. Method/segment/description are retained as contract metadata; generic training still reads project train config. |
| Candidate targets / sample variants | `candidate_targets`, `sample_variants` | Contract-only. Captured in Markdown; no generic CLI branching yet. |
| Evaluation metrics | AUC, KS, decile lift, ranking inversion, PSI, business risk | Partially executable. Standard evaluator/reporting supports core metrics from project/run artifacts. Request metric choices do not yet directly reconfigure `rmw evaluate`. |
| Evaluation dimensions | champions, comparison dimensions, risk profile dimensions | Partially executable. Champion comparison is wired to `rmw compare`; other dimensions are request/report context unless the project evaluation artifacts already provide them. |
| Report requirements | selected sections and requested output filenames | Partially executable. `rmw report` generates standard report/model-card/executive-summary outputs and optional Excel/HTML sidecars when artifacts exist. Arbitrary requested sections/filenames are not yet command-configurable. |
| Advanced parameters | monthly sample minimum, score PSI warning, monthly KS std cap, SQL high-risk blocking | Contract/planned. Values are preserved in `step_params`; most are not currently applied by stage commands. |

## Closed Gaps In Current Worktree

- Builder workflow options are now checked against real workflow files.
- Execution plans now honor workflow stage scope instead of always generating a
  full modeling chain.
- Feature plans insert `build_wide_sql` between prescreen and refine when a
  request asks for prescreen plus refine.
- Wide SQL generation registers run artifacts and can execute reviewed SQL only
  with explicit approval.
- Feature refinement registers standard run artifacts including
  `feature_selection/stage_summary.json` and `feature_selection/final_features.txt`.

## Remaining Gaps

1. Request values are not automatically materialized into `project.yml` or
   stage config files. The current execution commands still treat project config
   as source of truth.
2. `step_params` are planner metadata, not a general config override system for
   sample, feature, evaluation, or report commands.
3. Several UI-visible capabilities are intentionally planned-only: automated SQL
   review blocking, some domain-specific sample/evaluation slices, and some
   feature filters as independent generic steps.
4. Evaluation metrics and report section/output selections are not yet direct
   CLI switches.

Bottom line: the workbench can accept and plan the Markdown request, and it can
execute the implemented stages without fabricating unsupported commands. It
cannot yet claim that every UI-configurable requirement is automatically
satisfied end-to-end by generic `rmw` commands.
