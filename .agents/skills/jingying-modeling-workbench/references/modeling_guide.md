# Modeling Guide

Training commands may scaffold when local training data is unavailable. A
scaffold artifact is only a trace that the stage was planned or attempted.

When local feather data and a feature list exist, prefer `jm train` over
project-local ad hoc scripts. The reusable LightGBM path writes model pickle,
train/valid metrics, feature importance, preprocessing detail, run config, and
optional all-split scores.

For evaluation, prefer `jm evaluate` with a scored feather table. It writes
overall/monthly/segment metrics, decile lift, benchmark uplift, risk-profile
tables, and score PSI.

For reports, prefer `jm report`. If standard train and evaluation artifacts
exist, it generates `reports/model_report.xlsx`; otherwise it must clearly mark
the report as scaffold.
