import json

import pytest

from risk_model_workbench.data.local_feather_profile import (
    LocalFeatherProfileError,
    profile_local_feather,
    write_local_feather_profile,
)


pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def test_profile_local_feather_captures_metadata_without_copying_payload(tmp_path):
    feather_path = tmp_path / "data" / "raw" / "model.feather"
    feather_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "uid": [1, 2, 3, 4],
            "final_flag": ["DEV", "DEV", "OOT", "OOT"],
            "target": [1, 0, None, 1],
            "feat_a": [0.1, 0.2, 0.3, 0.4],
            "feat_b": [10, 11, 12, 13],
        }
    ).to_feather(feather_path)

    profile = profile_local_feather(
        feather_path,
        required_columns=["uid", "final_flag", "target"],
        split_column="final_flag",
        target_column="target",
        feature_exclude_columns=["uid", "final_flag", "target"],
    )

    assert profile["status"] == "ok"
    assert profile["path"] == str(feather_path)
    assert profile["exists"] is True
    assert profile["suffix"] == ".feather"
    assert profile["row_count"] == 4
    assert profile["column_count"] == 5
    assert profile["required_columns"]["missing"] == []
    assert profile["split_distribution"] == {"DEV": 2, "OOT": 2}
    assert profile["label_valid_count"] == 3
    assert profile["candidate_feature_count"] == 2
    assert "payload_copied" not in profile


def test_profile_local_feather_raises_for_missing_required_columns(tmp_path):
    feather_path = tmp_path / "model.feather"
    pd.DataFrame({"uid": [1], "feat_a": [0.1]}).to_feather(feather_path)

    with pytest.raises(LocalFeatherProfileError, match="missing required columns"):
        profile_local_feather(
            feather_path,
            required_columns=["uid", "target"],
            target_column="target",
        )


def test_profile_local_feather_rejects_non_feather_suffix(tmp_path):
    csv_path = tmp_path / "model.csv"
    csv_path.write_text("uid,target\n1,0\n", encoding="utf-8")

    with pytest.raises(LocalFeatherProfileError, match="must be a .feather file"):
        profile_local_feather(csv_path, required_columns=["uid"])


def test_write_local_feather_profile_writes_json_summary_only(tmp_path):
    output_path = tmp_path / "run" / "feature_selection" / "profiles" / "local_feather_profile.json"
    profile = {"status": "ok", "path": "/tmp/model.feather", "row_count": 1}

    written = write_local_feather_profile(profile, output_path)
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert written == output_path
    assert payload == profile
