# Report Standard

Reports must be generated from registered artifacts. Missing data must be shown
as missing or scaffold, not inferred from chat context.

`rmw report` always registers lightweight Markdown report placeholders for
traceability. When standard train and evaluation artifacts exist, it also writes
`reports/model_report.xlsx` and marks the report stage as done.

The Excel report should contain at least:

- summary and training setup
- overall, monthly, and benchmark model effects
- segment-level effects
- decile lift tables
- intent and asset-rating cross tables
- risk observation tables
- score stability
- top features and dropped-feature detail
- source-data summaries

Project-specific historical risk definitions belong in project config or
project docs. For the Fujie GCard case, MOB1/MOB3 historical risk definitions
are still not fully confirmed. Current amount-risk and headcount-risk tables are
useful diagnostic views, but they should not be described as exact historical
MOB1/MOB3 metrics until the business definition is confirmed.
