# Resource-Aware Feature Selection Intake Design

Date: 2026-06-25

## Purpose

Add a resource-aware data intake gate to the risk modeling workbench so feature prescreening and refinement can adapt to the memory capacity of the current execution environment before pulling DP data into memory.

The goal is not to make local machines handle unlimited data. The goal is to make each run explicit about data source mode, table scale or local file scale, memory budget, sampling decisions, feature batching, SQL lineage, and intermediate screening evidence.

This design applies to the reusable `rmw` workbench. Fujie GCard is the active case and regression example, but the implementation must stay generic.

## Confirmed Decisions

- Use full-table uniform random sampling by default, not DEV/OOT stratified sampling.
- The request-builder HTML must let the user explicitly choose the modeling data source mode:
  - remote DP table or SQL-backed source
  - local feather file
- The exported Markdown must preserve that choice with a stable field such as `data_source_mode: local_feather`.
- A local feather file can be used directly as the data source for the full workflow when it contains the required sample, split, label, and feature columns.
- Use the current environment's available memory as the sizing basis.
- Default readable memory budget is 80% of currently available memory.
- Apply a peak-memory multiplier before deciding row capacity:
  - feature prescreening: default `3.0`
  - feature refinement and model-importance stages: default `4.0`
- At the beginning of feature selection, detect the current execution platform and record the DP data-pull policy.
- Use `sh_dp_mcp` for remote select-only exploration on every platform: counts, split distribution, schema-like probes, random-field checks, and created-table validation.
- Do not auto-select `dp_cli` for Windows/macOS. Desktop runs should use `local_feather` mode for an already-downloaded local file, or an explicit reviewed remote-pull override.
- Use `TMLSQLClient` by default for DP data extraction on Linux and other environments.
- Keep create-table execution as a separate reviewed execution path.
- If row count is too large, reduce the random sampling ratio and/or add a `limit`.
- If feature count is too large, process features in batches. The default batch size is 1000 feature columns.
- Release no-longer-needed in-memory tables, feature frames, model objects, and LightGBM datasets throughout the flow.
- Persist all SQL, generated SQL, exploration results, sampling plans, batch plans, per-batch results, and aggregate screening results as local files that are not excluded by `.gitignore`.
- Persist actual data files and heavy binary/model/cache artifacts only in ignored locations.

## Non-Goals

- Do not remove the existing SQL approval gate.
- Do not make `sh_dp_mcp` responsible for bulk data transfer or table creation.
- Do not use `dp_cli` for metadata exploration when `sh_dp_mcp` is available.
- Do not treat a local feather file as a data-pull engine; it is an existing-file data source independent from DP pull.
- Do not upload, copy into Git, or register raw local feather data as a tracked artifact.
- Do not run remote table profiling or DP pulling when the request explicitly selects local feather mode and no remote table is configured.
- Do not store raw row-level data, feather files, pickle caches, model binaries, secrets, or local credentials in Git.
- Do not treat `.pkl` checkpoints or `.feather` files as the only audit evidence.
- Do not require DEV/OOT balanced sampling unless a future request explicitly changes the default.

## Current System Context

The existing workbench already has these reusable pieces:

- `src/risk_model_workbench/batch_feature_select.py`: per-table feature prescreening, DP feather cache, D01/D02 screening, and parallel table processing.
- `src/risk_model_workbench/feature_refine.py`: wide-table sampling SQL, `--sample-max-rows`, DP feather cache, D03/D04/D05 refinement.
- `src/risk_model_workbench/wide_sql.py`: wide-table CTAS SQL generation.
- `src/risk_model_workbench/dp_feather.py`: DP query execution, local feather writing, create-table execution, and SQL approval prompts. It currently uses `TMLSQLClient` and should gain an environment-selected data-pull engine.
- `src/risk_model_workbench/cli.py`: run-stage orchestration and artifact registration.
- `workflows/full_modeling.yml`: stage contracts for feature metadata, prescreening, wide-table build, refinement, training, evaluation, comparison, and reporting.

The missing piece is a shared gate that decides which data source mode is active, how much data to pull or read, and how to split feature batches before the heavy work starts.

## Architecture

Add a shared resource-aware intake layer used by both feature prescreening and feature refinement.

```text
request-builder HTML
  -> Markdown request with data_source_mode
  -> request validation and runtime config materialization
  -> data source resolver
  -> execution environment detection
  -> SQL evidence registry when remote SQL/table mode is active
  -> remote table profiling through sh_dp_mcp or local feather profiling
  -> local memory probe
  -> capacity estimate
  -> uniform random sampling plan
  -> feature batch plan
  -> reviewed SQL generation
  -> configured remote data pull or local feather read
  -> feature screening/refinement
  -> per-batch evidence and aggregate result
```

## Component Boundaries

### Request Builder Data Source Contract

Responsibilities:

- Add an explicit data source selector in `tools/model_request_builder/index.html`.
- Store the selector in the builder state in `tools/model_request_builder/app.js`.
- Export the selector to Markdown front matter.
- Keep `sample_location` as the location field, but validate it according to mode:
  - remote table mode: table name or SQL reference
  - local feather mode: local `.feather` path, usually under `data/raw/` or another local ignored data directory
- Show local feather mode as a first-class user choice, not a hidden convention inferred only from file suffix.

Recommended Markdown fields:

```yaml
data_source_mode: local_feather
sample_location: data/raw/model.feather
```

For remote sources:

```yaml
data_source_mode: remote_table
sample_location: ads_app_off_feature.some_sample_table
```

### Data Source Resolver

Responsibilities:

- Resolve request metadata into one canonical data contract for the run.
- For `local_feather`, write runtime configs with `data.raw_path` and no required DP pull.
- For `remote_table`, write runtime configs with `data.source_table` and keep the remote profiling/pull path active.
- Persist the resolved mode and reason.
- Fail early when local feather mode is selected but the path is missing, not a feather file, or does not contain required fields.

Output:

- `runs/<run_id>/feature_selection/data_source_contract.json`
- `runs/<run_id>/configs_runtime/project.yml`

### Environment Capacity Probe

Responsibilities:

- Read total memory and currently available memory for the running environment.
- Record the platform name used by the data-pull engine selector.
- Compute the usable memory budget: `available_memory_bytes * memory_budget_fraction`.
- Estimate matrix capacity after peak multiplier.
- Record the environment snapshot and formulas used.

Output:

- `runs/<run_id>/feature_selection/resource_plan.json`

### Execution Environment Detector

Responsibilities:

- Detect the current platform before feature selection data access begins.
- Normalize platform values into `windows`, `macos`, `linux`, or `other`.
- Persist the selected data-pull engine and reason.
- Fail early when the selected engine is unavailable and no explicit override is configured.

Default engine mapping:

| Platform | Data-pull engine | Reason |
| --- | --- | --- |
| Windows | none by default | Use `local_feather` for an existing local file or set an explicit reviewed remote-pull override. |
| macOS | none by default | Use `local_feather` for an existing local file or set an explicit reviewed remote-pull override. |
| Linux | `TMLSQLClient` | Server/notebook environment keeps existing behavior. |
| Other | `TMLSQLClient` | Conservative fallback to existing behavior. |

Output:

- `runs/<run_id>/feature_selection/execution_environment.json`

### Data Pull Engine

Responsibilities:

- Expose one interface for reviewed DP select queries that return local DataFrames or feather files.
- Do not auto-select a remote DP pull engine on Windows/macOS.
- Use `TMLSQLClient` on Linux/other platforms.
- Keep SQL approval behavior identical regardless of selected engine.
- Persist engine, platform, SQL hash, row count, column count, and local data path in metadata.

This boundary applies to remote data extraction for prescreening and refinement. It does not replace `sh_dp_mcp` profiling, it does not automatically replace CTAS execution, and it is separate from local feather mode.

### Remote Table Profiler

Responsibilities:

- Use `sh_dp_mcp` select-only queries to explore remote tables.
- Capture total row count.
- Capture split distribution, especially DEV/OOT counts when a split column exists.
- Capture label-valid counts when a target column exists.
- Capture available random columns and their usable value ranges.
- Capture a bounded schema/column probe.
- Profile newly created wide tables after reviewed CTAS execution.

Output examples:

- `runs/<run_id>/feature_selection/profiles/source_table_profile.json`
- `runs/<run_id>/feature_selection/profiles/wide_table_profile.json`
- `runs/<run_id>/feature_selection/profiles/random_column_profile.json`

### Local Feather Profiler

Responsibilities:

- Inspect a local feather file without treating it as a tracked artifact.
- Capture file path, existence, size, row count, column count, required-field availability, split distribution, label-valid counts, and candidate feature count.
- Reuse the same memory capacity estimate before reading large local files into pandas.
- Persist only metadata and profile summaries, never the feather payload.

Output examples:

- `runs/<run_id>/feature_selection/profiles/local_feather_profile.json`

### SQL Evidence Registry

Responsibilities:

- Save every user-provided SQL script before modification.
- Save every system-generated SQL script.
- Save the source, purpose, generated timestamp, and SQL hash.
- Keep SQL artifacts in tracked locations.

Output examples:

- `runs/<run_id>/queries/user_sql/*.sql`
- `runs/<run_id>/queries/generated/*.sql`
- `runs/<run_id>/queries/sql_evidence_manifest.json`

### Sampling Planner

Responsibilities:

- Use full-table uniform random sampling by default.
- Prefer explicit random columns from request/project config.
- Fall back only when the workflow can prove a safe random expression exists.
- Generate sampling predicates such as `rand_flag0 < 0.1`.
- Add `limit <max_rows>` when needed to keep memory under budget.
- For local feather mode, generate an equivalent local row-selection plan rather than SQL predicates.
- Preserve the user-visible rationale.

Output:

- `runs/<run_id>/feature_selection/sampling_plan.json`

### Feature Batch Planner

Responsibilities:

- Decide whether features must be split into batches.
- Default to 1000 feature columns per batch.
- Keep required non-feature columns in every batch: ids, time, split, target, random columns, champion scores, and configured base columns.
- Persist batch membership and SQL fragments for every batch.

Output:

- `runs/<run_id>/feature_selection/batch_plan.json`
- `runs/<run_id>/feature_selection/batches/batch_001_plan.json`
- `runs/<run_id>/queries/generated/batch_001_*.sql`

### Memory Lifecycle Guard

Responsibilities:

- Explicitly release dataframes and temporary objects once later steps no longer need them.
- Call `gc.collect()` after batch completion and after large model objects are no longer needed.
- Keep small summary payloads and persisted artifacts, not full dataframes.

Minimum release points:

- after each prescreen batch finishes
- after each refine batch finishes
- after D01/D02 intermediate frames are summarized
- after D03/D04/D05 model objects are summarized
- after local feather is read and converted into the next smaller frame
- after final aggregate artifacts are written

## Capacity Estimate

The baseline estimate assumes float64 feature storage:

```text
row_width_bytes = (feature_column_count + required_non_feature_column_count) * 8
matrix_budget_bytes = available_memory_bytes * memory_budget_fraction / peak_multiplier
max_rows = floor(matrix_budget_bytes / row_width_bytes)
```

The estimate must be stored with:

- total memory
- available memory
- memory budget fraction
- peak multiplier
- required non-feature column count
- feature column count
- row width
- max readable rows
- selected random sampling ratio
- selected limit, if any

The planner should be conservative. A run may still lower capacity manually by config.

## Uniform Random Sampling

Default behavior:

- Use a random column such as `rand_flag0`.
- Select rows by threshold, for example `rand_flag0 < 0.02`.
- Apply the predicate uniformly to the full table.
- Preserve DEV/OOT distribution as a natural consequence of full-table sampling.

If the random column distribution is suspect, the run should fail with a clear message instead of silently using a biased fallback.

## Feature Batching

Prescreening:

- Existing logic already screens per feature table.
- Add row-capacity planning and optional intra-table feature batches when a single feature table has too many columns.
- Persist per-table and per-batch evidence as JSON/CSV, not only `.pkl` checkpoints.
- In local feather mode, if the file already contains a wide candidate-feature frame, prescreen/refine should operate directly on the local columns and skip remote feature-table pulls.
- In local feather mode, `build_wide_sql` should be skipped or marked not applicable unless the request also provides remote feature tables.

Refinement:

- If the remaining feature set fits memory, run the existing one-shot refinement.
- If it does not fit, run batch refinement first, then do a final global convergence pass on the candidate pool.
- The final convergence pass is necessary because cross-batch correlation and importance are not comparable if only per-batch winners are merged.

## Artifact Layout

Canonical tracked evidence:

```text
runs/<run_id>/
  queries/
    user_sql/
    generated/
    sql_evidence_manifest.json
  feature_selection/
    data_source_contract.json
    execution_environment.json
    resource_plan.json
    sampling_plan.json
    batch_plan.json
    profiles/
      local_feather_profile.json
      source_table_profile.json
      random_column_profile.json
      wide_table_profile.json
    batches/
      batch_001_plan.json
      batch_001_result.json
      batch_001_feature_stats.csv
      ...
    prescreen_run_summary.json
    prescreen_final_remain_features.json
    wide_sql_summary.json
    wide_table_execution.json
    feature_stage_summary.json
    final_features.txt
```

Ignored data/cache artifacts:

```text
projects/*/data/raw/*
projects/*/data/local/**
projects/*/runs/**/*.feather
projects/*/runs/**/*.pkl
projects/*/runs/**/*.joblib
projects/*/runs/**/*.bin
projects/*/runs/**/*.model
```

The current repository `.gitignore` already excludes these data and binary artifacts while allowing JSON, CSV, and SQL evidence to be tracked.

## SQL Governance

- Every generated SQL must be written before execution.
- Create-table SQL still goes through static SQL review.
- `--sql-approved` remains required before any configured remote data-pull engine runs data pulls.
- `--sql-approved` remains required before the reviewed CTAS execution path runs create-table statements.
- `sh_dp_mcp` exploration queries are select-only and bounded by the MCP result limit.
- High-risk SQL review findings block execution even when approval is present.
- Local feather mode does not require SQL approval for reading the local file, but it still requires local file profiling, memory planning, and artifact evidence before heavy processing.

## Failure Handling

Fail fast when:

- memory cannot be probed and no manual memory budget is provided
- platform detection fails and no explicit pull-engine override is provided
- the selected data-pull engine is unavailable in the current environment
- local feather mode is selected but the file is missing, unreadable, not `.feather`, or lacks required fields
- the source table cannot be profiled
- no usable random column or sampling expression exists
- sampled rows are estimated to exceed memory budget
- DEV or OOT disappears after uniform sampling and the stage requires both
- generated SQL fails static review
- newly created wide table cannot be profiled
- required evidence artifacts cannot be written

Partial batch failures should not be hidden. The aggregate summary must report completed, skipped, and failed batches separately.

## Testing Strategy

Focused unit tests:

- memory capacity calculation
- uniform random ratio selection
- `limit` fallback behavior
- feature batch planning
- SQL evidence manifest writing
- `.gitignore`-compatible artifact path decisions
- memory cleanup hooks called after batch completion

Integration-style tests with fakes:

- `sh_dp_mcp` table profile adapter using fake query results
- data-pull engine selection for Windows/macOS/Linux/other platforms
- request-builder export of `data_source_mode`
- local feather request validation and runtime materialization
- local feather profiling with fake feather fixtures
- `dp_cli` data-pull execution using fake clients
- `TMLSQLClient` data-pull execution using fake clients
- CTAS execution using fake clients
- prescreen run writes tracked JSON/CSV/SQL evidence
- refine dry run writes sampling and batch plans
- wide-table execution writes profile evidence after CTAS

Smoke commands:

- `pytest tests/test_resource_planning.py tests/test_feature_pipeline_flow.py -q`
- `pytest tests -q`
- `rmw workflow validate --workflow workflows/full_modeling.yml`
- `rmw project validate --project projects/2026-05-fujie-gcard-v1`
