# Resource-Aware Feature Selection Intake Implementation Plan

> **For agentic workers:** REQUIRED: follow TDD where practical. Do not implement before this plan is approved. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resource-aware data intake gate so a request can choose either a remote DP source or a local feather file, then feature prescreening and refinement can profile the selected source, estimate local memory capacity, choose full-table uniform random sampling when needed, batch excessive features, persist all SQL/evidence artifacts, and release memory after each batch.

**Architecture:** Extend the request-builder HTML and Markdown contract with an explicit data source mode, materialize that request into run-scoped runtime configs, then add reusable planning/profiling modules and integrate them into existing `rmw feature prescreen`, `rmw build-wide-sql`, and `rmw feature refine` flows. Keep `sh_dp_mcp` as select-only profiler for remote sources. Keep local feather mode as an independent existing-file data source, not a DP pull engine. For remote feature-selection pulls, detect the platform and record the DP pull policy: Windows/macOS do not auto-select a remote pull engine, while Linux/other platforms default to `TMLSQLClient` unless explicitly overridden. Keep CTAS execution as a separate reviewed execution path.

**Tech Stack:** Static HTML/JS request builder, Python, pandas/pyarrow, pytest, existing `rmw` CLI/state/artifact helpers, existing `TMLSQLClient` wrapper, `sh_dp_mcp` adapter/fake for tests.

---

## Chunk 1: Request Builder and Markdown Data Source Contract

### Task 1: Add Explicit Local Feather Source Mode To The Request Builder

**Goal:** Let users choose local feather as a first-class data source in the HTML builder and preserve the choice in exported Markdown.

**Files:**
- Modify: `tools/model_request_builder/index.html`
- Modify: `tools/model_request_builder/app.js`
- Modify: `tools/model_request_builder/styles.css` if layout needs adjustment
- Modify: `tools/model_request_builder/README.md`
- Test manually in browser or add lightweight DOM/static tests if the project has a browser-test harness

**Changes:**
- Add a data source mode control in the "样本与切分" section:
  - `remote_table` / DP 表或 SQL 来源
  - `local_feather` / 本地 feather 文件
- Keep `sample_location`, but make the placeholder and helper text mode-aware:
  - remote table mode: `ads_app_off_feature.some_sample_table`
  - local feather mode: `data/raw/model.feather`
- Store the mode in builder state as `data_source_mode`.
- Export Markdown front matter with:
  - `data_source_mode`
  - `sample_location`
- Preserve backward compatibility:
  - if old Markdown has no `data_source_mode`, infer `local_feather` only when `sample_location` ends with `.feather`; otherwise default to `remote_table`.
- Update preview/summary text so users can see whether the request will use remote DP data or a local feather file.

**Acceptance:**
- User can explicitly select local feather in the HTML.
- Downloaded Markdown includes `data_source_mode: local_feather`.
- Existing templates still load and export valid Markdown.
- The UI does not imply that local feather will be committed or uploaded.

**Verification:**
- [ ] Open `tools/model_request_builder/index.html`.
- [ ] Select local feather, enter `data/raw/model.feather`, preview Markdown, and verify `data_source_mode: local_feather`.
- [ ] Select remote table, enter a table name, preview Markdown, and verify `data_source_mode: remote_table`.

## Chunk 2: Request Validation and Runtime Materialization

### Task 2: Carry Local Feather Mode Through `rmw request` And `rmw run init`

**Goal:** Make the Markdown request contract executable by the backend workflow.

**Files:**
- Modify: `src/risk_model_workbench/request/validate.py`
- Modify: `src/risk_model_workbench/request/materialize.py`
- Modify: `src/risk_model_workbench/planning/execution_plan.py` if task scope must change for local feather mode
- Modify: `tests/test_request_materialize.py`
- Modify: `tests/test_request_plan.py`
- Possibly add: `tests/test_request_data_source_mode.py`

**Changes:**
- Accept `data_source_mode` values:
  - `remote_table`
  - `local_feather`
- Backward-compatible inference:
  - missing mode + `.feather` sample location -> local feather warning/info
  - missing mode + table-like sample location -> remote table
- Validate local feather mode:
  - `sample_location` is present
  - path suffix is `.feather`
  - path is project-relative or explicitly absolute
  - path points to an ignored/raw-data location when practical
- Materialize runtime project config:
  - local feather -> `data.raw_path`
  - remote table -> `data.source_table`
  - always preserve `request.data_source_mode`
- Write `configs_runtime/request_runtime.yaml` with the resolved mode.
- For local feather mode, plan should not require remote feature metadata/prescreen/build-wide-SQL unless the request explicitly asks for remote feature tables.

**Acceptance:**
- `rmw request validate` accepts local feather requests.
- `rmw run init --request` writes runtime configs with `data.raw_path`.
- Existing remote-table requests keep working.
- Local feather mode can drive sample check, feature selection from local columns, training, evaluation, and report generation.

**Verification:**
- [ ] Run `pytest tests/test_request_materialize.py tests/test_request_plan.py -q`.

## Chunk 3: Resource Planning Core

### Task 3: Add Memory Capacity Estimation

**Goal:** Compute safe row capacity from current environment memory, feature count, and peak multiplier.

**Files:**
- Create: `src/risk_model_workbench/resource_planning.py`
- Create: `tests/test_resource_planning.py`

**Changes:**
- Add dataclasses or typed dict helpers for:
  - memory snapshot
  - capacity request
  - capacity estimate
- Implement memory probing with platform fallbacks:
  - macOS/Linux/local fallback from Python standard library where possible
  - manual override accepted by function arguments for tests and remote environments
- Implement:
  - `estimate_max_rows(...)`
  - `choose_uniform_sampling_ratio(...)`
  - `build_resource_plan_payload(...)`
- Default knobs:
  - `memory_budget_fraction=0.8`
  - prescreen `peak_multiplier=3.0`
  - refine `peak_multiplier=4.0`
  - `bytes_per_numeric_value=8`
- Support both remote table estimates and local feather file estimates.

**Acceptance:**
- Given 16GB available memory, 96 features, 20 non-feature columns, and multiplier 4, output row capacity is conservative and deterministic.
- Given 15028 features, output row capacity is small enough to force sampling/batching.
- Formula details are included in returned payload.
- Local feather mode includes file size and estimated in-memory expansion in the resource payload when available.

**Verification:**
- [ ] Run `pytest tests/test_resource_planning.py -q`.

## Chunk 4: Runtime Environment and Data Pull Engine

### Task 4: Record DP Data-Pull Policy By Platform

**Goal:** Detect the execution environment at the beginning of remote-source feature selection and avoid implicit desktop DP pulls.

**Files:**
- Create: `src/risk_model_workbench/data/pull_engine.py`
- Create: `tests/test_data_pull_engine.py`
- Modify: `src/risk_model_workbench/dp_feather.py`
- Modify: `src/risk_model_workbench/cli.py`

**Changes:**
- Add platform normalization:
  - `Windows` -> `windows`
  - `Darwin` -> `macos`
  - `Linux` -> `linux`
  - any other value -> `other`
- Add default policy mapping:
  - `windows` -> no auto-selected remote pull engine
  - `macos` -> no auto-selected remote pull engine
  - `linux` -> `tmlsqlclient`
  - `other` -> `tmlsqlclient`
- Add optional explicit override for controlled tests and emergency operations:
  - CLI/config key: `data_pull_engine`
  - accepted values: `auto`, `dp_cli`, `tmlsqlclient`
- Add a data-pull interface used by DP select-query extraction:
  - `pull_query_to_dataframe(sql, engine=...)`
  - `pull_query_to_feather(sql, feather_path, metadata_path, engine=...)`
- Implement fakeable adapters:
  - `TMLSQLClient` adapter for Linux/other pulls
- Keep `sh_dp_mcp` exploration outside this engine; it remains the profiler.
- Bypass this engine entirely when `data_source_mode=local_feather`.
- Persist:
  - `feature_selection/execution_environment.json`
  - selected platform
  - selected data-pull engine
  - override source, if any
  - availability check result

**Acceptance:**
- On fake Windows/macOS, no remote pull engine is auto-selected.
- On fake Linux/other, the selected data-pull engine is `tmlsqlclient`.
- `sh_dp_mcp` profiler tests are unaffected by the data-pull engine choice.
- Local feather mode does not select `dp_cli` or `TMLSQLClient` for data pull.
- SQL approval behavior remains identical for both data-pull engines.
- If the selected engine is unavailable, the stage fails before pulling data and writes a clear failure message.

**Verification:**
- [ ] Run `pytest tests/test_data_pull_engine.py -q`.

## Chunk 5: SQL Evidence Registry

### Task 5: Persist User SQL, Generated SQL, and SQL Metadata

**Goal:** Ensure all SQL inputs and generated SQL are saved in tracked locations with hashes and purpose metadata.

**Files:**
- Create: `src/risk_model_workbench/data/sql_evidence.py`
- Modify: `src/risk_model_workbench/cli.py`
- Test: `tests/test_sql_evidence.py`

**Changes:**
- Add helper to write SQL under:
  - `runs/<run_id>/queries/user_sql/`
  - `runs/<run_id>/queries/generated/`
- Add or update `runs/<run_id>/queries/sql_evidence_manifest.json`.
- Include:
  - `source`
  - `purpose`
  - `stage`
  - `created_at`
  - `sql_sha256`
  - `path`
- Register evidence artifacts in run manifest when called from a run.

**Acceptance:**
- User-provided SQL and generated SQL are written before any execution.
- Manifest remains JSON, not `.pkl` or log-only.
- Evidence paths are not excluded by current `.gitignore`.
- Local feather mode can have no SQL evidence and still pass when `data_source_contract.json` records the source file.

**Verification:**
- [ ] Run `pytest tests/test_sql_evidence.py -q`.

## Chunk 6: Source Profiling Adapters

### Task 6: Add Remote Table And Local Feather Profiling

**Goal:** Profile the selected data source without moving unnecessary data into memory.

**Files:**
- Create: `src/risk_model_workbench/data/table_profile.py`
- Create: `src/risk_model_workbench/data/local_feather_profile.py`
- Create: `tests/test_table_profile.py`
- Create: `tests/test_local_feather_profile.py`
- Modify: `src/risk_model_workbench/cli.py`

**Changes:**
- Define a profiler interface that can be backed by:
  - `sh_dp_mcp` in live runs
  - fake query client in tests
- Add a local feather profiler that captures:
  - file path
  - existence/readability
  - size bytes
  - row count
  - column count
  - required field availability
  - split distribution
  - label-valid counts
  - candidate feature count after excluding base columns
- Generate bounded select SQL for:
  - total row count
  - split distribution
  - label-valid row count
  - random column min/max/null/count bucket checks
  - bounded column preview if metadata source is unavailable
- Persist profile results under:
  - `runs/<run_id>/feature_selection/profiles/*.json`
- Persist local feather profile under:
  - `runs/<run_id>/feature_selection/profiles/local_feather_profile.json`
- Record query SQL and query IDs when available.

**Acceptance:**
- The adapter does not run CTAS or non-select SQL.
- The adapter does not switch to `dp_cli` on Windows/macOS; profiling stays on `sh_dp_mcp`.
- Local feather profiling does not call `sh_dp_mcp`, `dp_cli`, or `TMLSQLClient`.
- Local feather profiling never registers or copies the feather payload as a tracked artifact.
- Profile outputs include enough information to justify sampling and row-capacity choices.
- Failed profile queries produce clear failure codes.

**Verification:**
- [ ] Run `pytest tests/test_table_profile.py tests/test_local_feather_profile.py -q`.

## Chunk 7: Sampling and Batch Plan Generation

### Task 7: Build Uniform Random Sampling and Feature Batch Planner

**Goal:** Convert memory and table profile evidence into executable sampling and feature-batch plans.

**Files:**
- Create: `src/risk_model_workbench/feature_selection/intake_plan.py`
- Create: `tests/test_feature_intake_plan.py`
- Modify: `src/risk_model_workbench/config.py` if config normalization is needed

**Changes:**
- Implement `build_sampling_plan(...)`:
  - default full-table uniform random sampling
  - prefer configured random columns such as `rand_flag0`
  - compute ratio from `max_rows / total_rows`
  - clamp ratio to configured min/max if supplied
  - add `limit` only when still needed
  - in local feather mode, produce a local row-selection plan instead of SQL predicates
- Implement `build_feature_batch_plan(...)`:
  - default `max_features_per_batch=1000`
  - include required base columns in every batch
  - write stable deterministic batch IDs
- Persist:
  - `feature_selection/sampling_plan.json`
  - `feature_selection/batch_plan.json`
  - `feature_selection/batches/batch_###_plan.json`

**Acceptance:**
- For full-table row count 10M and capacity 1M, planner selects approximately 0.1 sampling ratio.
- For local feather row count 10M and capacity 1M, planner selects an equivalent local sample fraction or max-row cap.
- For 15028 features and batch size 1000, planner emits 16 batches.
- Required non-feature columns are included in every batch and not counted as feature candidates.

**Verification:**
- [ ] Run `pytest tests/test_feature_intake_plan.py -q`.

## Chunk 8: Prescreen Integration

### Task 8: Make `feature prescreen` Resource-Aware

**Goal:** Apply resource planning before D01/D02 prescreening and persist per-table/per-batch evidence.

**Files:**
- Modify: `src/risk_model_workbench/batch_feature_select.py`
- Modify: `src/risk_model_workbench/cli.py`
- Modify: `workflows/full_modeling.yml`
- Modify: `workflows/feature_selection.yml`
- Test: `tests/test_feature_pipeline_flow.py`
- Test: `tests/test_resource_planning.py`

**Changes:**
- Add CLI/config knobs:
  - `--auto-sample`
  - `--data-pull-engine`
  - `--memory-budget-fraction`
  - `--peak-memory-multiplier`
  - `--max-features-per-batch`
  - optional manual memory override for tests or managed environments
- Before fetching table data:
  - resolve `data_source_mode`
  - detect execution environment and selected data-pull engine
  - profile remote table or local feather source if profile evidence is missing or refresh requested
  - estimate capacity
  - build sampling plan
  - build per-table/per-batch plan
- Modify generated sample SQL to include the uniform random predicate.
- Pull sampled prescreen data through an explicitly selected or default remote data-pull engine:
  - Windows/macOS: no auto remote pull; use `local_feather` existing-file mode or explicit reviewed override
  - Linux/other: `TMLSQLClient`
- In local feather mode:
  - read only required columns when possible
  - sample rows locally according to the sampling plan
  - treat the feather as a wide candidate-feature frame
  - skip remote feature-table pulls
- If feature count exceeds batch size, process feature batches and aggregate results.
- Write JSON/CSV per-batch results in addition to existing `.pkl` checkpoint.
- Release dataframes and intermediate screening objects after each batch.

**Acceptance:**
- Existing fixed `sample_where` remains supported for backward compatibility.
- New auto mode writes resource, sampling, profile, and batch artifacts.
- Local feather mode writes `data_source_contract.json` and `local_feather_profile.json`.
- Execution environment and data-pull engine evidence are registered.
- Per-batch JSON/CSV results are tracked evidence; `.pkl` remains cache only.
- Stage state and artifact manifest include the new evidence files.

**Verification:**
- [ ] Run `pytest tests/test_feature_pipeline_flow.py -q`.
- [ ] Run `rmw feature prescreen --project projects/2026-05-fujie-gcard-v1 --run-id <test_run> --dry-run-sql` on a non-production/smoke run if fixtures allow it.

## Chunk 9: Wide Table Execution and Post-Create Profile

### Task 9: Profile the Wide Table After CTAS Execution

**Goal:** After reviewed CTAS execution creates the wide table, immediately validate the created table through select-only profiling.

**Files:**
- Modify: `src/risk_model_workbench/cli.py`
- Modify: `src/risk_model_workbench/dp_feather.py` if execution metadata needs extension
- Test: `tests/test_feature_pipeline_flow.py`
- Test: `tests/test_table_profile.py`

**Changes:**
- Keep current static SQL review and `--sql-approved` gate.
- Keep CTAS execution separate from the platform-selected data-pull engine unless a future implementation proves the local `dp_cli` path can satisfy the same DDL safety contract.
- In local feather mode without remote feature tables, mark `build_wide_sql` as skipped/not applicable with a tracked reason artifact instead of generating CTAS SQL.
- After successful CTAS execution:
  - run table profile on the output wide table
  - persist `feature_selection/profiles/wide_table_profile.json`
  - register profile artifact
- Capture:
  - row count
  - split distribution
  - label-valid counts
  - column count or bounded column probe
  - random column availability

**Acceptance:**
- CTAS still refuses to run without approval.
- High-risk SQL still blocks even when approved.
- Successful execution has both `wide_table_execution.json` and `wide_table_profile.json`.
- Local feather mode has a tracked `wide_table_skipped.json` or equivalent decision artifact.

**Verification:**
- [ ] Run `pytest tests/test_feature_pipeline_flow.py::test_build_wide_sql_execute_registers_artifacts -q`.
- [ ] Run all table-profile focused tests.

## Chunk 10: Refine Integration

### Task 10: Make `feature refine` Resource-Aware and Batch-Capable

**Goal:** Apply the same memory and sampling gate to refinement, with a final global convergence pass when features are batched.

**Files:**
- Modify: `src/risk_model_workbench/feature_refine.py`
- Modify: `src/risk_model_workbench/cli.py`
- Test: `tests/test_feature_pipeline_flow.py`
- Test: `tests/test_feature_refine_d03.py`
- Possibly create: `tests/test_feature_refine_batching.py`

**Changes:**
- Add auto-planning knobs equivalent to prescreen.
- Resolve `data_source_mode`.
- Detect execution environment and selected data-pull engine before pulling refine samples.
- Build refine sampling SQL from `sampling_plan.json`.
- Pull sampled refine data through an explicitly selected or default remote data-pull engine:
  - Windows/macOS: no auto remote pull; use `local_feather` existing-file mode or explicit reviewed override
  - Linux/other: `TMLSQLClient`
- In local feather mode:
  - read the local feather directly
  - use local row-selection and feature-batch plans
  - skip DP pull approval because no remote pull occurs
  - still write local profile/resource/sampling evidence
- If remaining features fit memory:
  - keep existing one-shot flow.
- If remaining features exceed capacity:
  - process feature batches first.
  - write batch-level refine outputs.
  - aggregate candidates.
  - run final global correlation / D03 / D04 / D05 on candidate pool if it fits.
- Add explicit cleanup after each batch and after model-importance stages.
- Preserve existing `--sample-max-rows` behavior as a manual override.

**Acceptance:**
- Existing dry-run SQL behavior remains unchanged unless auto-planning is enabled.
- Refine stage records execution environment and data-pull engine evidence.
- Local feather mode records local feather profile and data-source contract evidence.
- Batch mode writes per-batch evidence and final aggregate evidence.
- Final feature list remains registered as `feature_selection/final_features.txt`.
- The run summary states whether the refine result came from one-shot or batch-plus-convergence mode.

**Verification:**
- [ ] Run `pytest tests/test_feature_refine_d03.py tests/test_feature_pipeline_flow.py -q`.

## Chunk 11: Workflow Contracts, Rules, and Documentation

### Task 11: Register New Evidence and Guardrails

**Goal:** Make the new resource-aware behavior visible in workflow audit and project documentation.

**Files:**
- Modify: `tools/model_request_builder/README.md`
- Modify: `workflows/full_modeling.yml`
- Modify: `workflows/feature_selection.yml`
- Modify: `docs/workbench_rules.yml`
- Modify: `docs/feature_selection_standard.md`
- Modify: `AGENTS.md` if command guidance needs an update
- Test: `tests/test_workflow_state.py`
- Test: `tests/test_harness_hardening.py`

**Changes:**
- Add accepted artifact sets for:
  - `feature_selection/data_source_contract.json`
  - `feature_selection/execution_environment.json`
  - `feature_selection/resource_plan.json`
  - `feature_selection/sampling_plan.json`
  - `feature_selection/batch_plan.json`
  - `feature_selection/profiles/*.json`
  - `feature_selection/profiles/local_feather_profile.json`
  - `queries/sql_evidence_manifest.json`
- Add rules:
  - Request Markdown must preserve explicit `data_source_mode` when generated by the HTML builder.
  - Local feather files may be used as full workflow data sources when required fields are present.
  - Local feather payloads must stay ignored and must not be registered as tracked artifacts.
  - Feature selection must detect platform and data-pull engine before pulling DP data.
  - Windows/macOS feature-selection does not auto-select a remote pull engine.
  - Linux/other feature-selection data pulls use `TMLSQLClient` by default.
  - `sh_dp_mcp` remains the profiler on every platform.
  - DP pulls require resource and sampling plans.
  - SQL and planning evidence must be tracked.
  - Data files and heavy caches must stay ignored.
  - `.pkl` and `.feather` are not sufficient audit evidence.
  - large intermediate objects must be released after batch use.

**Acceptance:**
- `rmw run audit` can see the new evidence where required.
- Documentation clearly distinguishes tracked evidence from ignored data.
- Documentation explains the difference between remote table mode and local feather mode.

**Verification:**
- [ ] Run `pytest tests/test_workflow_state.py tests/test_harness_hardening.py -q`.
- [ ] Run `rmw workflow validate --workflow workflows/full_modeling.yml`.

## Chunk 12: End-to-End Verification

### Task 12: Final Checks

**Goal:** Verify the implementation without relying on real DP data pulls.

**Files:**
- All files changed by this plan.

**Changes:**
- Run focused tests first.
- Run full test suite if focused tests pass.
- Run workflow and project validation.
- Review git diff for unintended changes.

**Acceptance:**
- Tests pass.
- Workflow validation passes.
- No raw data, feather, pickle, model binary, or secret is staged.
- Final response reports any unverified DP-live behavior separately.

**Verification:**
- [ ] Run `pytest tests/test_request_materialize.py tests/test_request_plan.py tests/test_resource_planning.py tests/test_data_pull_engine.py tests/test_sql_evidence.py tests/test_table_profile.py tests/test_local_feather_profile.py tests/test_feature_intake_plan.py tests/test_feature_pipeline_flow.py -q`.
- [ ] Run `pytest tests -q`.
- [ ] Run `rmw workflow validate --workflow workflows/full_modeling.yml`.
- [ ] Run `rmw project validate --project projects/2026-05-fujie-gcard-v1`.
- [ ] Run `git status --short`.
