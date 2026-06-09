"""Decile lift calculations."""

from __future__ import annotations

import pandas as pd


def compute_decile_lift(df: pd.DataFrame, score_col: str, label_col: str) -> pd.DataFrame:
    """Compute equal-frequency decile lift for a score column."""
    sub = df[[score_col, label_col]].dropna(subset=[score_col]).copy()
    if len(sub) < 20:
        return pd.DataFrame()

    sub["decile"] = pd.qcut(sub[score_col], 10, labels=False, duplicates="drop")
    if sub["decile"].nunique() < 2:
        return pd.DataFrame()

    bad_rate_total = sub[label_col].mean()
    rows = []
    for decile in sorted(sub["decile"].dropna().unique()):
        group = sub[sub["decile"] == decile]
        n = len(group)
        bad = int(group[label_col].sum())
        cum = sub[sub["decile"] <= decile]
        remaining = sub[sub["decile"] > decile]
        cum_n = len(cum)
        remaining_n = len(remaining)
        cum_bad = int(cum[label_col].sum())
        remaining_bad = int(remaining[label_col].sum())
        cum_bad_rate = cum_bad / cum_n if cum_n else 0
        remaining_bad_rate = remaining_bad / remaining_n if remaining_n else 0
        rows.append(
            {
                "decile": int(decile) + 1,
                "n_samples": n,
                "pct": n / len(sub),
                "bad": bad,
                "bad_rate": group[label_col].mean(),
                "cum_bad": cum_bad,
                "cum_bad_rate": cum_bad_rate,
                "cum_lift": cum_bad_rate / bad_rate_total if bad_rate_total > 0 else 0,
                "remaining_bad": remaining_bad,
                "remaining_bad_rate": remaining_bad_rate,
                "remaining_lift": remaining_bad_rate / bad_rate_total if bad_rate_total > 0 else 0,
            }
        )
    return pd.DataFrame(rows)
