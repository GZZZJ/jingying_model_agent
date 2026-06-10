"""Score stability calculations."""

from __future__ import annotations

import pandas as pd

from risk_model_workbench.evaluation.metrics import psi_score


def compute_score_psi(df: pd.DataFrame, score_col: str, time_col: str, n_bins: int = 10) -> pd.DataFrame:
    """Compute score-distribution PSI by month using the first month as baseline."""
    sub = df[[score_col, time_col]].dropna(subset=[score_col]).copy()
    if len(sub) < 20:
        return pd.DataFrame()

    sub["_month"] = pd.to_datetime(sub[time_col], errors="coerce").dt.to_period("M")
    months = sorted(sub["_month"].dropna().unique())
    if len(months) < 2:
        return pd.DataFrame()

    sub["_bin"] = pd.qcut(sub[score_col], n_bins, labels=False, duplicates="drop")
    base_dist = sub[sub["_month"] == months[0]]["_bin"].value_counts(normalize=True).sort_index()

    rows = []
    for month in months:
        current = sub[sub["_month"] == month]
        current_dist = current["_bin"].value_counts(normalize=True).sort_index()
        bins = sorted(set(base_dist.index) | set(current_dist.index))
        rows.append(
            {
                "month": str(month),
                "psi": psi_score([base_dist.get(b, 0) for b in bins], [current_dist.get(b, 0) for b in bins]),
                "n_samples": len(current),
            }
        )
    return pd.DataFrame(rows)
