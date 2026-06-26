"""Guardrail tests: evaluate must emit bin-level PSI detail and segmented intent
matrices when the inputs allow, so these are never silently skipped.

Locks the behavior added in enhance/eval-bin-psi-segmented-intent:
- score_psi_bin_detail.csv (per month x per bin base/current prop + psi component)
- intent_zc_*_{e2e3,b2}.csv (intent x zc matrices per blue_customer_flag segment)
"""

import pandas as pd
import pytest

from risk_model_workbench.evaluation.run import _write_intent_risk_outputs, _write_psi_outputs


@pytest.fixture()
def synthetic_scores() -> pd.DataFrame:
    n = 400
    scores = [(i % 100) / 100 for i in range(n)]
    flags = ["E2", "E3", "B2", "E3"]
    months = ["2025-06-15", "2025-07-15", "2025-08-15", "2026-01-15"]
    return pd.DataFrame(
        {
            "model_score": scores,
            "ftr_30d_ord_flag": [1 if s > 0.6 else 0 for s in scores],
            "zc_level": [str((i % 7) + 1) for i in range(n)],
            "blue_customer_flag": [flags[i % 4] for i in range(n)],
            "prc_amt_xz_30d_3m": [float(i) for i in range(n)],
            "ovd_amt_xz_30d_3m": [float(i % 5) for i in range(n)],
            "mdl_dte": [months[i % 4] for i in range(n)],
        }
    )


def test_psi_outputs_include_bin_detail(tmp_path, synthetic_scores):
    _write_psi_outputs(synthetic_scores, tmp_path, ["model_score"], "mdl_dte")

    monthly = pd.read_csv(tmp_path / "score_psi_by_month.csv", encoding="utf-8-sig")
    assert "psi" in monthly.columns and not monthly.empty

    bin_detail = pd.read_csv(tmp_path / "score_psi_bin_detail.csv", encoding="utf-8-sig")
    assert set(["month", "bin", "base_prop", "current_prop", "psi_component"]).issubset(bin_detail.columns)
    # baseline month components are ~0; later months carry the drift signal
    assert bin_detail["month"].nunique() >= 2
    assert (bin_detail["psi_component"].fillna(0) >= 0).mean() > 0.5


def test_intent_outputs_include_segments(tmp_path, synthetic_scores):
    segment_filters = {
        "e2e3": synthetic_scores["blue_customer_flag"].isin(["E2", "E3"]),
        "b2": synthetic_scores["blue_customer_flag"] == "B2",
    }
    _write_intent_risk_outputs(synthetic_scores, tmp_path, "ftr_30d_ord_flag", segment_filters=segment_filters)

    # cohort files still produced (unchanged contract)
    assert (tmp_path / "intent_zc_distribution.csv").exists()
    # per-segment files now produced (no longer "missing")
    for seg in ["e2e3", "b2"]:
        dist = pd.read_csv(tmp_path / f"intent_zc_distribution_{seg}.csv", encoding="utf-8-sig")
        assert {"intent_level", "zc_level", "n_samples", "bad_rate"}.issubset(dist.columns)
        assert not dist.empty
        assert (tmp_path / f"intent_zc_headcount_risk_{seg}.csv").exists()
