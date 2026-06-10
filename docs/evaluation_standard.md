# Evaluation Standard

Evaluation outputs must be generated from scored predictions, not from manually
typed metrics. The scored table should include the label, split column, time
column, model score, and any benchmark scores that need comparison.

`jm evaluate` writes standard artifacts when a score feather is available:

- `evaluation/evaluation_summary.json`
- `evaluation/overall_metrics.csv`
- `evaluation/monthly_metrics.csv`
- `evaluation/segment_metrics.csv`
- `evaluation/benchmark_uplift.csv`
- `evaluation/decile_lift_*.csv`
- `evaluation/intent_zc_distribution.csv`
- `evaluation/intent_zc_ftr_rate.csv`
- `evaluation/intent_zc_amount_risk.csv`
- `evaluation/intent_zc_headcount_risk.csv`
- `evaluation/score_psi_by_month.csv`

The standard metrics are AUC, KS, bad rate, decile lift, benchmark uplift,
score PSI, segment-level effects, and configured risk-profile cuts. Benchmark
score columns are project configuration, not workbench defaults. The Fujie GCard
case config evaluates `model_score` against `gcard_v2`, `gcard_v4`, `gcard_v5`,
and `gcard_v6` across `DEV`, `DEV-OOS`, `OOT`, and `OOT-OOS`.

If prediction data is unavailable, the command may write a scaffold summary.
Scaffold summaries must clearly state missing inputs and must not fabricate
model performance.
