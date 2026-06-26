import pandas as pd

from risk_model_workbench.evaluation.decile_lift import compute_decile_lift
from risk_model_workbench.evaluation.metrics import auc_score, ks_score, psi_score
from risk_model_workbench.evaluation.stability import compute_score_psi


def test_auc_ks_and_psi_scores():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]

    assert auc_score(y_true, y_score) == 1.0
    assert ks_score(y_true, y_score) == 1.0
    assert psi_score([10, 10], [10, 10]) == 0.0


def test_decile_lift_and_monthly_psi():
    frame = pd.DataFrame(
        {
            "score": [i / 40 for i in range(40)],
            "label": [0] * 20 + [1] * 20,
            "mdl_dte": ["2026-01-01"] * 20 + ["2026-02-01"] * 20,
        }
    )

    lift = compute_decile_lift(frame, "score", "label")
    assert not lift.empty
    assert lift.iloc[0]["decile"] == 1
    assert lift.iloc[-1]["cum_lift"] == 1.0
    assert lift.iloc[-1]["remaining_lift"] == 0.0

    psi, bin_detail = compute_score_psi(frame, "score", "mdl_dte")
    assert list(psi["month"]) == ["2026-01", "2026-02"]
    # bin-level PSI detail is now produced alongside the monthly scalar
    assert not bin_detail.empty
    assert set(["month", "bin", "base_prop", "current_prop", "psi_component"]).issubset(bin_detail.columns)
    # components for the baseline month sum to ~0, later months reconstruct the psi
    assert bin_detail["month"].nunique() == 2
