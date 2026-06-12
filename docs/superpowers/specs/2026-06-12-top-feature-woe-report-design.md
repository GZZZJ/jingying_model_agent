# Top Feature WOE Report Design

Date: 2026-06-12

## Purpose

Add WOE charts for the model's top variables to the standard report. The charts help reviewers see whether each important feature has a sensible risk direction, stable split behavior, and acceptable population distribution.

This design applies to the reusable `rmw` workbench. Fujie GCard is the current example run, but the implementation must stay generic.

## Confirmed Decisions

- Generate one WOE chart for each Top 20 feature.
- Rank Top features by `feature_importance.csv` using descending `gain`.
- Use `DEV` as the base split for bin boundary creation.
- Apply the same bins to all available splits.
- Use the chart style "WOE line plus population percentage bars".
- Add a new Excel sheet named `Top变量WOE`.
- Save chart images and the underlying aggregate table as registered run artifacts.

## Current Constraint

The current imported run `2026-06-imported-gcard-main-lgbm` cannot generate real WOE charts from its registered artifacts alone. It has model outputs, feature importance, scores, and evaluation summaries, but it does not contain row-level Top feature values.

The workbench must therefore compute WOE artifacts while local feature data is available, normally during or immediately after training. The report stage must embed existing WOE artifacts. It must not fabricate WOE charts when raw feature values are missing.

## Scope

In scope:

- Compute Top 20 WOE tables from local feature data, label, and split columns.
- Render one PNG chart per Top feature.
- Register the WOE summary CSV and PNG files in the run artifact manifest.
- Add `Top变量WOE` to `model_report.xlsx`.
- Add a short WOE section or pointer to `model_report.md` and `model_report.html`.
- Update the missing-results document when WOE artifacts cannot be generated.

Out of scope:

- WOE-based model training.
- Manual bin editing.
- Interactive chart dashboards.
- Historical GCard score WOE charts.
- Business Chinese variable descriptions. Those still require a feature dictionary.

## Artifact Layout

Canonical computed artifacts:

```text
runs/<run_id>/modeling/<experiment>/woe_top_features/
  woe_top20_summary.csv
  images/
    001_<safe_feature_name>_WOE.png
    ...
    020_<safe_feature_name>_WOE.png
```

Report-facing artifacts:

```text
runs/<run_id>/reports/woe_top_features/
  woe_top20_summary.csv
  images/
    001_<safe_feature_name>_WOE.png
    ...
    020_<safe_feature_name>_WOE.png
```

The modeling directory is the source artifact because WOE depends on row-level feature values. The report directory may copy those files for convenient delivery with `model_report.xlsx`.

## WOE Calculation

For each Top feature:

1. Read `feature`, `gain`, and `split` from `modeling/<experiment>/feature_importance.csv`.
2. Select the Top 20 rows by `gain`.
3. Read the feature value, label column, and split column from the local training feature table.
4. Replace configured missing sentinels and nulls with a dedicated missing bucket.
5. Use rows where `split_col == "DEV"` to create numeric bin boundaries.
6. Prefer equal-frequency bins with `qcut`, target `n_bins = 10`, and `duplicates = "drop"`.
7. Fall back to equal-width bins if equal-frequency binning fails.
8. If a feature cannot produce at least two non-missing bins, skip chart rendering for that feature and record the reason in the summary CSV.
9. Apply the final bins to all available splits, including `DEV`, `OOT`, `DEV-OOS`, and `OOT-OOS` when present.

Missing values get a separate `Missing` bucket. The chart places this bucket first.

WOE uses the risk-oriented form:

```text
WOE = ln(bad_distribution / good_distribution)
```

where `bad` is label `1` and `good` is label `0`. Each bin uses additive smoothing of `0.5` for good and bad counts to avoid infinite WOE values.

The summary CSV uses long format:

```text
feature, rank, gain, split_importance, bin_order, bin_label,
lower_bound, upper_bound, is_missing_bin, split_value,
good, bad, total, bad_rate, pop_pct, woe, iv_component,
status, skip_reason
```

`iv_component` is:

```text
(bad_distribution - good_distribution) * WOE
```

## Chart Design

Each PNG contains one feature.

The x-axis shows ordered bins. The missing bin appears first, followed by numeric bins in ascending order.

The left y-axis shows WOE. Each split gets one line.

The right y-axis shows population percentage. Each split gets one grouped bar per bin.

The title includes feature rank and feature name. A subtitle includes gain, total IV, missing rate, and base split. The legend includes both WOE lines and population bars.

The default image settings are:

- Format: PNG
- DPI: 180 or higher
- Figure size: about 15 by 7 inches
- Matplotlib backend: non-interactive `Agg`

Feature names must be sanitized for filenames.

## Report Integration

`model_report.xlsx` adds a sheet named `Top变量WOE`.

The sheet layout:

- One feature block per Top variable.
- Each block starts with rank, feature name, gain, IV, and missing rate.
- The chart image appears below the block header.
- Blocks are stacked vertically.

If WOE artifacts are missing, the sheet still exists and states that WOE charts require row-level feature values. The missing-results document must list the missing input instead of silently omitting the section.

`model_report.md` and `model_report.html` add a short section with:

- The Top 20 feature list.
- The WOE artifact directory.
- A note that full charts are in Excel and PNG files.

## Components

Add a focused WOE module:

```text
src/risk_model_workbench/explainability/woe.py
```

Responsibilities:

- Select Top features from feature importance.
- Compute bin boundaries from the base split.
- Compute per-split WOE and population percentages.
- Write `woe_top20_summary.csv`.
- Render PNG charts.

This module should expose pure functions where practical:

- `select_top_features(importance, top_n)`
- `build_feature_bins(series, base_mask, n_bins, missing_values)`
- `compute_woe_table(df, feature, label_col, split_col, bins, missing_values)`
- `plot_woe_chart(table, feature_meta, output_path)`
- `generate_top_feature_woe(...)`

Keep Excel embedding inside:

```text
src/risk_model_workbench/reporting/excel_report.py
```

The report module should read existing WOE artifacts and insert images. It should not recompute WOE from raw row-level data.

## Workflow Integration

Training integration:

- `train_lightgbm_from_feather` already has the raw feature table, label column, split column, and feature importance.
- After writing `feature_importance.csv`, call the WOE generator when enabled.
- Write WOE artifacts under `modeling/<experiment>/woe_top_features/`.
- Register `woe_top_features/woe_top20_summary.csv` and each PNG under `train_baseline`.

Report integration:

- `rmw report` looks for `modeling/<experiment>/woe_top_features/`.
- If found, copy or mirror the WOE directory to `reports/woe_top_features/`.
- Register report-facing WOE artifacts under `report`.
- Add `Top变量WOE` to the workbook.

Configuration defaults:

```yaml
explainability:
  top_feature_woe:
    enabled: true
    top_n: 20
    n_bins: 10
    base_split_value: DEV
    missing_sentinels: [-999, -998]
```

The generator should also work if this config block is absent by using the defaults above.

## Error Handling

The generator must continue when one feature fails. It should record the failure in the summary CSV and continue with the remaining Top features.

Expected non-fatal cases:

- Feature missing from the local feature table.
- Feature has too few non-missing unique values.
- Base split has no valid rows.
- A split has only good or only bad labels.
- Matplotlib cannot render a specific chart.

The command should fail only for run-level problems, such as missing label column, missing split column, unreadable input feather, or a malformed feature importance file.

## Data Safety

WOE artifacts must contain only aggregated counts, rates, WOE values, and charts. They must not write row-level feature values, identifiers, raw feather files, model binaries, or secrets.

## Tests

Add focused tests for:

- Top feature selection by gain.
- DEV-based bin reuse across all splits.
- Missing bucket ordering.
- WOE formula and smoothing behavior.
- Skip behavior for constant or all-missing features.
- Filename sanitization.
- Report behavior when WOE artifacts exist.
- Report behavior when WOE artifacts are absent.

Add a CLI smoke test that runs the WOE generation on a small synthetic dataset and verifies:

- `woe_top20_summary.csv` exists.
- At least one PNG exists.
- `rmw report` creates `Top变量WOE`.

## Acceptance Criteria

- `rmw train` on a local feature feather writes Top20 WOE artifacts when data is available.
- `rmw report` embeds the Top20 WOE images in `model_report.xlsx`.
- `rmw report` registers the WOE CSV and PNG files in `artifact_manifest.json`.
- The report explains missing WOE artifacts instead of fabricating charts.
- Existing report generation still works for imported runs without raw feature values.
- Relevant unit tests and CLI smoke tests pass.
