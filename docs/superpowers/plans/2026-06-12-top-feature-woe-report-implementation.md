# Top Feature WOE Report Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate Top20 feature WOE charts during training and embed them in the standard model report.

**Architecture:** Add a focused explainability module for WOE table and chart generation. Training writes canonical WOE artifacts while row-level feature values are available; report generation copies and embeds those artifacts without recomputing from raw data.

**Tech Stack:** Python, pandas, numpy, matplotlib Agg, openpyxl, pytest, existing `rmw` CLI/state helpers.

---

## Chunk 1: WOE Computation

### Task 1: Add WOE Unit Tests

**Files:**
- Create: `tests/test_woe_explainability.py`
- Create: `src/risk_model_workbench/explainability/woe.py`

- [ ] Write tests for Top feature selection by `gain`, DEV-based bin reuse, missing bucket ordering, WOE smoothing, skip behavior, filename sanitization, and PNG/CSV generation on synthetic data.
- [ ] Run `pytest tests/test_woe_explainability.py -q` and verify the tests fail because the module does not exist.
- [ ] Implement `src/risk_model_workbench/explainability/woe.py`.
- [ ] Re-run `pytest tests/test_woe_explainability.py -q` and verify it passes.

## Chunk 2: Training Integration

### Task 2: Generate WOE Artifacts From Local Feature Data

**Files:**
- Modify: `src/risk_model_workbench/modeling/train_lgb.py`
- Modify: `src/risk_model_workbench/cli.py`
- Test: `tests/test_woe_explainability.py`

- [ ] Add a test that training-style generation writes `modeling/<experiment>/woe_top_features/woe_top20_summary.csv` and PNG images from a synthetic raw frame.
- [ ] Run the focused test and verify failure.
- [ ] Call the WOE generator after `feature_importance.csv` is written, using config defaults when `explainability.top_feature_woe` is absent.
- [ ] Register the WOE CSV and PNG files under `train_baseline`.
- [ ] Re-run focused tests.

## Chunk 3: Report Integration

### Task 3: Embed Existing WOE Artifacts In Reports

**Files:**
- Modify: `src/risk_model_workbench/reporting/excel_report.py`
- Modify: `src/risk_model_workbench/cli.py`
- Test: `tests/test_report_render.py`

- [ ] Add a report test that supplies existing WOE PNG/CSV artifacts and verifies workbook sheet `Top变量WOE`.
- [ ] Add a report test that verifies imported/no-WOE runs still produce a clear missing-artifact note.
- [ ] Run the focused report tests and verify failure.
- [ ] Add `Top变量WOE` to `REPORT_SHEETS`.
- [ ] Copy or mirror WOE artifacts from modeling to reports.
- [ ] Insert WOE images in the new worksheet when present; write a missing note otherwise.
- [ ] Add a short WOE section to Markdown/HTML.
- [ ] Register report-facing WOE CSV and PNG artifacts.
- [ ] Re-run focused report tests.

## Chunk 4: Verification

### Task 4: Final Checks

**Files:**
- All files changed by this plan.

- [ ] Run `pytest tests/test_woe_explainability.py tests/test_report_render.py -q`.
- [ ] Run `pytest tests -q`.
- [ ] Run `rmw project validate --project projects/2026-05-fujie-gcard-v1`.
- [ ] Run `rmw run audit --project projects/2026-05-fujie-gcard-v1 --run-id 2026-06-imported-gcard-main-lgbm`.
- [ ] Review `git diff` and confirm no unrelated dirty files were modified.
