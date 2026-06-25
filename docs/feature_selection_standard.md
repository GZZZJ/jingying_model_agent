# Feature Selection Standard

Use `vendor/feature-select-v2/` through the workbench integration. Do not modify
vendored core algorithms unless explicitly requested.

## Workbench Concepts

The reusable workbench uses generic feature-selection stages:

1. `feature_prescreen`: coarse feature prescreening. Use sampling when the
   candidate feature count is too large, then apply low-cost quality and
   stability checks such as missing rate, constant or near-constant value share,
   and PSI. This stage reduces the feature set enough to build a manageable
   wide-table SQL.
2. `build_wide_sql`: assemble the prescreened features into a wide-table SQL so
   later selection can use broader or fuller data.
3. `feature_refine`: feature refinement/convergence on the wide-table sample,
   including executable-feature filtering, global correlation deduplication,
   random-noise importance, null importance, and baseline-model importance.

`feature_prescreen` plus `feature_refine` is the complete workbench feature
selection flow. Project-specific method names from legacy feature-selection
libraries must stay as implementation details or compatibility aliases.

## Resource-Aware Intake

Every new request should make the modeling data source explicit:

- `data_source_mode: remote_table` uses a DP table or SQL-backed remote source.
- `data_source_mode: local_feather` uses a local `.feather` file as the runtime
  sample and feature frame.

Remote-source feature selection must keep `sh_dp_mcp` as the select-only
profiler. Reviewed remote bulk pulls are a separate DP-pull chain: Linux/other
platforms default to `TMLSQLClient`, while Windows/macOS do not auto-select a
remote pull engine. Desktop runs should either use explicit `local_feather`
mode for an already-downloaded file, or set an explicit remote pull engine
override after review. The SQL approval gate still applies before any remote
pull or CTAS execution.

Local feather mode is an independent data-source chain, not a DP extraction
engine. It bypasses DP profiling and bulk pull engines. The workflow may profile
the feather file and persist JSON summaries, but must not copy the feather
payload into tracked artifacts.

Resource-aware stages should persist these tracked evidence files before or
alongside heavy feature-selection work:

- `feature_selection/data_source_contract.json`
- `feature_selection/execution_environment.json`
- `feature_selection/resource_plan.json`
- `feature_selection/sampling_plan.json`
- `feature_selection/batch_plan.json`
- `feature_selection/profiles/*.json`
- `queries/sql_evidence_manifest.json`

`.pkl`, `.feather`, model binaries, and local caches are never sufficient audit
evidence by themselves.
