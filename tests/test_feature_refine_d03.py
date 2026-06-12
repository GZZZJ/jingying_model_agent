import pandas as pd

from risk_model_workbench import feature_refine
from risk_model_workbench.feature_refine import DatasetParts, d03_random_importance, select_feature_select_v2_drops


def test_feature_select_v2_drop_rules_split_gain_zero_and_tail():
    importance = pd.DataFrame(
        {
            "feature": ["strong", "medium", "weak", "zero", "random_col"],
            "split": [10, 5, 1, 0, 2],
            "gain": [100, 30, 6, 0, 5],
        }
    )

    drops, detail = select_feature_select_v2_drops(
        importance,
        "random_col",
        thresholds=0.8,
        importance_types=["split", "gain"],
    )

    assert drops["random"] == {"weak"}
    assert drops["zero"] == {"zero"}
    assert drops["thresholds"] == {"medium", "random_col"}
    survives = set(detail.loc[detail["survives"], "feature"])
    assert survives == {"strong"}
    assert "random_col" not in set(detail["feature"])


def test_feature_select_v2_threshold_cumsum_includes_random_column():
    importance = pd.DataFrame(
        {
            "feature": ["strong", "medium", "random_col"],
            "split": [10, 5, 1],
            "gain": [80, 15, 5],
        }
    )

    drops, detail = select_feature_select_v2_drops(
        importance,
        "random_col",
        thresholds=0.8,
        importance_types=["gain"],
    )

    assert drops["thresholds"] == {"medium", "random_col"}
    assert set(detail.loc[detail["dropped"], "feature"]) == {"medium"}


def test_d03_feature_select_v2_uses_union_of_bagging_drops(monkeypatch):
    class FakeModel:
        def __init__(self, features):
            self.features = features

        def feature_importance(self, importance_type):
            values = {
                "strong": {"split": 10, "gain": 100},
                "weak": {"split": 1, "gain": 6},
                "random_col": {"split": 2, "gain": 5},
            }
            return [values[feature][importance_type] for feature in self.features]

    def fake_train(train_x, train_y, features, cfg, seed):
        return FakeModel(features), 0.5

    monkeypatch.setattr(feature_refine, "train_feature_select_v2_model", fake_train)
    parts = DatasetParts(
        train_x=pd.DataFrame({"strong": [1, 2, 3, 4], "weak": [4, 3, 2, 1]}),
        train_y=pd.Series([0, 1, 0, 1]),
        valid_x=pd.DataFrame({"strong": [1, 2], "weak": [2, 1]}),
        valid_y=pd.Series([0, 1]),
    )
    cfg = {
        "random_seed": 0,
        "d03_random_importance": {
            "enabled": True,
            "mode": "feature_select_v2",
            "bagging_rounds": 2,
            "bagging_fraction": 1.0,
            "thresholds": None,
            "importance_types": ["split", "gain"],
        },
    }

    kept, detail = d03_random_importance(parts, ["strong", "weak"], cfg)

    assert kept == ["strong"]
    assert set(detail["feature"]) == {"strong", "weak"}
    assert set(detail.loc[detail["dropped"], "feature"]) == {"weak"}
