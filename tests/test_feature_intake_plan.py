import json

import pytest

from risk_model_workbench.feature_selection.intake_plan import (
    IntakePlanError,
    build_feature_batch_plan,
    build_sampling_plan,
    persist_intake_plan,
)


def test_remote_sampling_plan_uses_uniform_random_column():
    plan = build_sampling_plan(
        data_source_mode="remote_table",
        total_rows=10_000_000,
        max_rows=1_000_000,
        random_columns=["rand_flag0"],
    )

    assert plan["data_source_mode"] == "remote_table"
    assert plan["sampling_required"] is True
    assert plan["method"] == "full_table_uniform_random"
    assert plan["ratio"] == 0.1
    assert plan["random_column"] == "rand_flag0"
    assert plan["sql_predicate"] == "rand_flag0 < 0.1"
    assert plan["limit"] is None


def test_local_feather_sampling_plan_uses_local_row_selection():
    plan = build_sampling_plan(
        data_source_mode="local_feather",
        total_rows=10_000_000,
        max_rows=1_000_000,
        random_columns=["rand_flag0"],
    )

    assert plan["method"] == "local_uniform_random"
    assert plan["ratio"] == 0.1
    assert plan["sql_predicate"] is None
    assert plan["local_row_selection"] == {
        "method": "uniform_random_fraction",
        "fraction": 0.1,
        "max_rows": 1_000_000,
    }


def test_remote_sampling_plan_requires_random_column_when_sampling_needed():
    with pytest.raises(IntakePlanError, match="random column"):
        build_sampling_plan(
            data_source_mode="remote_table",
            total_rows=10_000_000,
            max_rows=1_000_000,
            random_columns=[],
        )


def test_feature_batch_plan_emits_stable_batches_and_keeps_required_columns():
    features = [f"feat_{idx:05d}" for idx in range(15028)]
    plan = build_feature_batch_plan(
        feature_columns=features,
        required_columns=["uid", "target", "final_flag"],
        max_features_per_batch=1000,
    )

    assert plan["total_feature_count"] == 15028
    assert plan["batch_count"] == 16
    assert plan["batches"][0]["batch_id"] == "batch_001"
    assert plan["batches"][0]["feature_count"] == 1000
    assert plan["batches"][0]["select_columns"][:3] == ["uid", "target", "final_flag"]
    assert plan["batches"][-1]["batch_id"] == "batch_016"
    assert plan["batches"][-1]["feature_count"] == 28
    assert "uid" not in plan["batches"][0]["feature_columns"]


def test_feature_batch_plan_removes_required_columns_from_feature_candidates():
    plan = build_feature_batch_plan(
        feature_columns=["uid", "feat_a", "target", "feat_b"],
        required_columns=["uid", "target"],
        max_features_per_batch=1,
    )

    assert plan["total_feature_count"] == 2
    assert [batch["feature_columns"] for batch in plan["batches"]] == [["feat_a"], ["feat_b"]]
    assert all(batch["select_columns"][0:2] == ["uid", "target"] for batch in plan["batches"])


def test_persist_intake_plan_writes_summary_and_per_batch_json(tmp_path):
    run_dir = tmp_path / "run"
    sampling_plan = build_sampling_plan(
        data_source_mode="local_feather",
        total_rows=100,
        max_rows=50,
    )
    batch_plan = build_feature_batch_plan(
        feature_columns=["feat_a", "feat_b"],
        required_columns=["uid", "target"],
        max_features_per_batch=1,
    )

    written = persist_intake_plan(run_dir, sampling_plan=sampling_plan, batch_plan=batch_plan)

    assert written["sampling_plan"] == run_dir / "feature_selection" / "sampling_plan.json"
    assert written["batch_plan"] == run_dir / "feature_selection" / "batch_plan.json"
    assert written["batch_files"] == [
        run_dir / "feature_selection" / "batches" / "batch_001_plan.json",
        run_dir / "feature_selection" / "batches" / "batch_002_plan.json",
    ]
    assert json.loads(written["batch_files"][0].read_text(encoding="utf-8"))["batch_id"] == "batch_001"
