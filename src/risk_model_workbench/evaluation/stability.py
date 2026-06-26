"""Score stability calculations."""

from __future__ import annotations

import pandas as pd

from risk_model_workbench.evaluation.metrics import psi_score


def compute_score_psi(df: pd.DataFrame, score_col: str, time_col: str, n_bins: int = 10):
    """Compute score-distribution PSI by month using the first month as baseline.

    Returns ``(monthly_df, bin_detail_df)``:
    - monthly_df: ``month, psi, n_samples`` (unchanged values/contract).
    - bin_detail_df: ``month, bin, base_prop, current_prop, psi_component`` — the
      per-bin distributions + PSI component that aggregate into ``psi``, so bin-level
      stability detail is produced rather than left as a report stub.
    """
    empty = pd.DataFrame(), pd.DataFrame()
    sub = df[[score_col, time_col]].dropna(subset=[score_col]).copy()
    if len(sub) < 20:
        return empty

    sub["_month"] = pd.to_datetime(sub[time_col], errors="coerce").dt.to_period("M")
    months = sorted(sub["_month"].dropna().unique())
    if len(months) < 2:
        return empty

    sub["_bin"] = pd.qcut(sub[score_col], n_bins, labels=False, duplicates="drop")
    base_dist = sub[sub["_month"] == months[0]]["_bin"].value_counts(normalize=True).sort_index()

    rows = []
    bin_rows = []
    for month in months:
        current = sub[sub["_month"] == month]
        current_dist = current["_bin"].value_counts(normalize=True).sort_index()
        bins = sorted(set(base_dist.index) | set(current_dist.index))
        base_arr = [base_dist.get(b, 0) for b in bins]
        curr_arr = [current_dist.get(b, 0) for b in bins]
        rows.append(
            {
                "month": str(month),
                "psi": psi_score(base_arr, curr_arr),
                "n_samples": len(current),
            }
        )
        components = _psi_components(base_arr, curr_arr)
        for b, base_p, curr_p, comp in zip(bins, base_arr, curr_arr, components):
            bin_rows.append(
                {
                    "month": str(month),
                    "bin": int(b),
                    "base_prop": float(base_p),
                    "current_prop": float(curr_p),
                    "psi_component": comp,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(bin_rows)


def _psi_components(expected: list[float], actual: list[float]) -> list[float | None]:
    """Per-bin PSI components using the same eps+normalize convention as metrics.psi_score."""
    import numpy as np

    eps = 1e-10
    e = np.asarray(expected, dtype=float) + eps
    a = np.asarray(actual, dtype=float) + eps
    if e.sum() <= 0 or a.sum() <= 0:
        return [None] * len(expected)
    e = e / e.sum()
    a = a / a.sum()
    return [None if (ei <= 0 or ai <= 0) else float((ai - ei) * np.log(ai / ei)) for ei, ai in zip(e, a)]

