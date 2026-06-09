import pandas as pd

from jingying_model_agent.evaluation.decile_lift import compute_decile_lift
from jingying_model_agent.evaluation.metrics import auc_score, ks_score, psi_score
from jingying_model_agent.evaluation.stability import compute_score_psi


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

    psi = compute_score_psi(frame, "score", "mdl_dte")
    assert list(psi["month"]) == ["2026-01", "2026-02"]
