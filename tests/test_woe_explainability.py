from pathlib import Path
import importlib.util

import pandas as pd

from risk_model_workbench.explainability.woe import (
    compute_woe_table,
    generate_top_feature_woe,
    sanitize_feature_filename,
    select_top_features,
)


def test_select_top_features_ranks_by_gain():
    importance = pd.DataFrame(
        {
            "feature": ["low", "top", "middle"],
            "gain": [1.0, 9.0, 4.0],
            "split": [2, 10, 5],
        }
    )

    result = select_top_features(importance, top_n=2)

    assert result["feature"].tolist() == ["top", "middle"]
    assert result["rank"].tolist() == [1, 2]


def test_sanitize_feature_filename_removes_unsafe_characters():
    assert sanitize_feature_filename('a/b:c*"d?<e>|') == "a_b_c__d__e__"


def test_compute_woe_table_uses_dev_bins_and_missing_bucket_first():
    frame = pd.DataFrame(
        {
            "feature_a": [-999, 1, 2, 3, 4, 5, 6, 7, -999, 10, 20, 30],
            "label": [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
            "final_flag": ["DEV"] * 8 + ["OOT"] * 4,
        }
    )

    table = compute_woe_table(
        frame,
        feature="feature_a",
        rank=1,
        gain=10.0,
        split_importance=3,
        label_col="label",
        split_col="final_flag",
        base_split_value="DEV",
        n_bins=3,
        missing_values=[-999],
    )

    assert table["status"].eq("ok").all()
    assert table["bin_label"].iloc[0] == "Missing"
    assert table["split_value"].drop_duplicates().tolist() == ["DEV", "OOT"]

    oot_bins = table.loc[table["split_value"] == "OOT", "bin_label"].tolist()
    assert oot_bins == table.loc[table["split_value"] == "DEV", "bin_label"].tolist()
    assert table["woe"].notna().all()


def test_compute_woe_table_skips_constant_features():
    frame = pd.DataFrame(
        {
            "constant_feature": [1, 1, 1, 1],
            "label": [0, 1, 0, 1],
            "final_flag": ["DEV", "DEV", "OOT", "OOT"],
        }
    )

    table = compute_woe_table(
        frame,
        feature="constant_feature",
        rank=1,
        gain=1.0,
        split_importance=1,
        label_col="label",
        split_col="final_flag",
        base_split_value="DEV",
        n_bins=10,
    )

    assert table.shape[0] == 1
    assert table.iloc[0]["status"] == "skipped"
    assert "too_few_bins" in table.iloc[0]["skip_reason"]


def test_generate_top_feature_woe_writes_summary_and_pngs(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "feature_a": list(range(1, 41)),
            "feature_b": [-999, *range(2, 41)],
            "feature_c": [1] * 40,
            "label": [0, 0, 0, 1, 1] * 8,
            "final_flag": ["DEV"] * 20 + ["OOT"] * 20,
        }
    )
    importance = pd.DataFrame(
        {
            "feature": ["feature_a", "feature_b", "feature_c"],
            "gain": [30.0, 20.0, 10.0],
            "split": [5, 4, 3],
        }
    )

    result = generate_top_feature_woe(
        frame,
        importance,
        output_dir=tmp_path / "woe_top_features",
        label_col="label",
        split_col="final_flag",
        top_n=3,
        n_bins=4,
        base_split_value="DEV",
        missing_values=[-999],
    )

    assert result.summary_path.exists()
    if importlib.util.find_spec("matplotlib"):
        assert len(result.image_paths) == 2
        assert all(path.exists() and path.suffix == ".png" for path in result.image_paths)
    else:
        assert result.image_paths == []

    summary = pd.read_csv(result.summary_path)
    assert {"feature", "rank", "bin_label", "split_value", "woe", "iv_component", "status", "skip_reason"}.issubset(summary.columns)
    assert set(summary.loc[summary["status"] == "ok", "feature"]) == {"feature_a", "feature_b"}
    assert summary.loc[summary["feature"] == "feature_b", "bin_label"].iloc[0] == "Missing"
    assert summary.loc[summary["feature"] == "feature_c", "status"].iloc[0] == "skipped"
