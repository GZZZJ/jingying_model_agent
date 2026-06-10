"""Stable evaluation metric helpers."""

from __future__ import annotations

from typing import Any


def auc_score(y_true: Any, y_score: Any) -> float | None:
    """Return AUC, or None when the slice cannot be evaluated."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    yt = np.asarray(y_true, dtype=float)
    ys = np.asarray(y_score, dtype=float)
    mask = ~np.isnan(yt) & ~np.isnan(ys) & np.isin(yt, [0, 1])
    if mask.sum() < 2 or len(np.unique(yt[mask])) < 2:
        return None
    return float(roc_auc_score(yt[mask].astype(int), ys[mask]))


def ks_score(y_true: Any, y_score: Any) -> float | None:
    """Return two-sample KS, or None when the slice cannot be evaluated."""
    import numpy as np
    from scipy.stats import ks_2samp

    yt = np.asarray(y_true, dtype=float)
    ys = np.asarray(y_score, dtype=float)
    mask = ~np.isnan(yt) & ~np.isnan(ys) & np.isin(yt, [0, 1])
    if mask.sum() < 2 or len(np.unique(yt[mask])) < 2:
        return None
    pos = ys[mask][yt[mask] == 1]
    neg = ys[mask][yt[mask] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(ks_2samp(pos, neg).statistic)


def psi_score(expected: Any, actual: Any) -> float | None:
    """Return PSI between two distributions."""
    import numpy as np

    eps = 1e-10
    e = np.asarray(expected, dtype=float) + eps
    a = np.asarray(actual, dtype=float) + eps
    if e.sum() <= 0 or a.sum() <= 0:
        return None
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))
