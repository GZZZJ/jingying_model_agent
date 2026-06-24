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
