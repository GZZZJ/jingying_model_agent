# Resource-Aware Feature Selection Intake Implementation Plan

> **For agentic workers:** REQUIRED: follow TDD where practical. Do not implement before this plan is approved. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resource-aware data intake gate so feature prescreening and refinement automatically profile remote tables, estimate local memory capacity, choose full-table uniform random sampling, batch excessive features, persist all SQL/evidence artifacts, and release memory after each batch.

**Architecture:** Add small reusable planning/profiling modules, then integrate them into existing `rmw feature prescreen`, `rmw build-wide-sql`, and `rmw feature refine` flows. Keep `sh_dp_mcp` as select-only profiler and `TMLSQLClient` as the approved execution engine.

**Tech Stack:** Python, pandas, pytest, existing `rmw` CLI/state/artifact helpers, existing `TMLSQLClient` wrapper, `sh_dp_mcp` adapter/fake for tests.

---

## Chunk 1: Resource Planning Core

### Task 1: Add Memory Capacity Estimation

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

**Acceptance:**
- Given 16GB available memory, 96 features, 20 non-feature columns, and multiplier 4, output row capacity is conservative and deterministic.
- Given 15028 features, output row capacity is small enough to force sampling/batching.
- Formula details are included in returned payload.

**Verification:**
- [ ] Run `pytest tests/test_resource_planning.py -q`.

## Chunk 2: SQL Evidence Registry

### Task 2: Persist User SQL, Generated SQL, and SQL Metadata

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

**Verification:**
- [ ] Run `pytest tests/test_sql_evidence.py -q`.

## Chunk 3: Remote Table Profiling Adapter

### Task 3: Add Select-Only DP Profiling Layer

**Goal:** Use `sh_dp_mcp`-style select queries to profile table scale and random sampling fields without moving bulk data.

**Files:**
- Create: `src/risk_model_workbench/data/table_profile.py`
- Create: `tests/test_table_profile.py`
- Modify: `src/risk_model_workbench/cli.py`

**Changes:**
- Define a profiler interface that can be backed by:
  - `sh_dp_mcp` in live runs
  - fake query client in tests
- Generate bounded select SQL for:
  - total row count
  - split distribution
  - label-valid row count
  - random column min/max/null/count bucket checks
  - bounded column preview if metadata source is unavailable
- Persist profile results under:
  - `runs/<run_id>/feature_selection/profiles/*.json`
- Record query SQL and query IDs when available.

**Acceptance:**
- The adapter does not run CTAS or non-select SQL.
- Profile outputs include enough information to justify sampling and row-capacity choices.
- Failed profile queries produce clear failure codes.

**Verification:**
- [ ] Run `pytest tests/test_table_profile.py -q`.

## Chunk 4: Sampling and Batch Plan Generation

### Task 4: Build Uniform Random Sampling and Feature Batch Planner

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
- For 15028 features and batch size 1000, planner emits 16 batches.
- Required non-feature columns are included in every batch and not counted as feature candidates.

**Verification:**
- [ ] Run `pytest tests/test_feature_intake_plan.py -q`.

## Chunk 5: Prescreen Integration

### Task 5: Make `feature prescreen` Resource-Aware

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
  - `--memory-budget-fraction`
  - `--peak-memory-multiplier`
  - `--max-features-per-batch`
  - optional manual memory override for tests or managed environments
- Before fetching table data:
  - profile table if profile evidence is missing or refresh requested
  - estimate capacity
  - build sampling plan
  - build per-table/per-batch plan
- Modify generated sample SQL to include the uniform random predicate.
- If feature count exceeds batch size, process feature batches and aggregate results.
- Write JSON/CSV per-batch results in addition to existing `.pkl` checkpoint.
- Release dataframes and intermediate screening objects after each batch.

**Acceptance:**
- Existing fixed `sample_where` remains supported for backward compatibility.
- New auto mode writes resource, sampling, profile, and batch artifacts.
- Per-batch JSON/CSV results are tracked evidence; `.pkl` remains cache only.
- Stage state and artifact manifest include the new evidence files.

**Verification:**
- [ ] Run `pytest tests/test_feature_pipeline_flow.py -q`.
- [ ] Run `rmw feature prescreen --project projects/2026-05-fujie-gcard-v1 --run-id <test_run> --dry-run-sql` on a non-production/smoke run if fixtures allow it.

## Chunk 6: Wide Table Execution and Post-Create Profile

### Task 6: Profile the Wide Table After CTAS Execution

**Goal:** After `TMLSQLClient` creates the wide table, immediately validate the created table through select-only profiling.

**Files:**
- Modify: `src/risk_model_workbench/cli.py`
- Modify: `src/risk_model_workbench/dp_feather.py` if execution metadata needs extension
- Test: `tests/test_feature_pipeline_flow.py`
- Test: `tests/test_table_profile.py`

**Changes:**
- Keep current static SQL review and `--sql-approved` gate.
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

**Verification:**
- [ ] Run `pytest tests/test_feature_pipeline_flow.py::test_build_wide_sql_execute_registers_artifacts -q`.
- [ ] Run all table-profile focused tests.

## Chunk 7: Refine Integration

### Task 7: Make `feature refine` Resource-Aware and Batch-Capable

**Goal:** Apply the same memory and sampling gate to refinement, with a final global convergence pass when features are batched.

**Files:**
- Modify: `src/risk_model_workbench/feature_refine.py`
- Modify: `src/risk_model_workbench/cli.py`
- Test: `tests/test_feature_pipeline_flow.py`
- Test: `tests/test_feature_refine_d03.py`
- Possibly create: `tests/test_feature_refine_batching.py`

**Changes:**
- Add auto-planning knobs equivalent to prescreen.
- Build refine sampling SQL from `sampling_plan.json`.
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
- Batch mode writes per-batch evidence and final aggregate evidence.
- Final feature list remains registered as `feature_selection/final_features.txt`.
- The run summary states whether the refine result came from one-shot or batch-plus-convergence mode.

**Verification:**
- [ ] Run `pytest tests/test_feature_refine_d03.py tests/test_feature_pipeline_flow.py -q`.

## Chunk 8: Workflow Contracts, Rules, and Documentation

### Task 8: Register New Evidence and Guardrails

**Goal:** Make the new resource-aware behavior visible in workflow audit and project documentation.

**Files:**
- Modify: `workflows/full_modeling.yml`
- Modify: `workflows/feature_selection.yml`
- Modify: `docs/workbench_rules.yml`
- Modify: `docs/feature_selection_standard.md`
- Modify: `AGENTS.md` if command guidance needs an update
- Test: `tests/test_workflow_state.py`
- Test: `tests/test_harness_hardening.py`

**Changes:**
- Add accepted artifact sets for:
  - `feature_selection/resource_plan.json`
  - `feature_selection/sampling_plan.json`
  - `feature_selection/batch_plan.json`
  - `feature_selection/profiles/*.json`
  - `queries/sql_evidence_manifest.json`
- Add rules:
  - DP pulls require resource and sampling plans.
  - SQL and planning evidence must be tracked.
  - Data files and heavy caches must stay ignored.
  - `.pkl` and `.feather` are not sufficient audit evidence.
  - large intermediate objects must be released after batch use.

**Acceptance:**
- `rmw run audit` can see the new evidence where required.
- Documentation clearly distinguishes tracked evidence from ignored data.

**Verification:**
- [ ] Run `pytest tests/test_workflow_state.py tests/test_harness_hardening.py -q`.
- [ ] Run `rmw workflow validate --workflow workflows/full_modeling.yml`.

## Chunk 9: End-to-End Verification

### Task 9: Final Checks

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
- [ ] Run `pytest tests/test_resource_planning.py tests/test_sql_evidence.py tests/test_table_profile.py tests/test_feature_intake_plan.py tests/test_feature_pipeline_flow.py -q`.
- [ ] Run `pytest tests -q`.
- [ ] Run `rmw workflow validate --workflow workflows/full_modeling.yml`.
- [ ] Run `rmw project validate --project projects/2026-05-fujie-gcard-v1`.
- [ ] Run `git status --short`.

