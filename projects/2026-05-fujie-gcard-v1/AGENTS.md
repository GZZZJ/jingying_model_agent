# 复借 G 卡项目规则

## Project

Project path: `projects/2026-05-fujie-gcard-v1`

This is the current Fujie GCard modeling project.

## Data

Target:
- `ftr_30d_ord_flag`

Keys:
- `uid`
- `mdl_dte`

Split column:
- `final_flag`

Champion score columns:
- `gcard_v2`
- `gcard_v4`
- `gcard_v5`
- `gcard_v6`

## Main Segments

- all
- e2e3
- b2
- e2
- e3

## Rules

- Do not change project definitions unless the user explicitly requests it.
- Preserve existing feature selection artifacts by importing them into a standard run.
- Use `runs/2026-06-imported-gcard-main-lgbm/` as the imported real-project baseline for main LightGBM training, evaluation, and report artifacts.
- Any report for this project must include sample positive rate, feature screening process, final selected features, model experiment summary, comparison with historical GCard scores, segment-level evaluation, month-level evaluation, and decile lift evaluation.
- MOB1/MOB3 historical risk definitions are not fully confirmed. Do not present current risk observation tables as exact historical MOB1/MOB3 metrics without confirmation.
