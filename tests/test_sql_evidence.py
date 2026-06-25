import json

from risk_model_workbench.data.sql_evidence import (
    is_tracked_sql_evidence_path,
    load_sql_evidence_manifest,
    write_sql_evidence,
)


def test_write_generated_sql_evidence_and_manifest(tmp_path):
    run_dir = tmp_path / "run"

    entry = write_sql_evidence(
        run_dir,
        "select * from mart.base\n",
        source="system",
        purpose="build prescreen sample",
        stage="feature_prescreen",
        sql_kind="generated",
        name="prescreen_sample",
    )

    sql_path = run_dir / entry["path"]
    manifest_path = run_dir / "queries" / "sql_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert sql_path == run_dir / "queries" / "generated" / "prescreen_sample.sql"
    assert sql_path.read_text(encoding="utf-8") == "select * from mart.base\n"
    assert entry["sql_sha256"] == manifest["entries"][0]["sql_sha256"]
    assert entry["purpose"] == "build prescreen sample"
    assert entry["stage"] == "feature_prescreen"


def test_write_user_and_generated_sql_evidence_append_without_overwriting(tmp_path):
    run_dir = tmp_path / "run"

    first = write_sql_evidence(
        run_dir,
        "select uid from mart.base\n",
        source="request",
        purpose="user supplied source query",
        stage="request",
        sql_kind="user_sql",
        name="source_query",
    )
    second = write_sql_evidence(
        run_dir,
        "select uid from mart.base where rand_flag0 < 0.1\n",
        source="planner",
        purpose="sampled source query",
        stage="feature_refine",
        sql_kind="generated",
        name="sampled_source_query",
    )

    manifest = load_sql_evidence_manifest(run_dir)
    paths = [entry["path"] for entry in manifest["entries"]]

    assert first["path"] == "queries/user_sql/source_query.sql"
    assert second["path"] == "queries/generated/sampled_source_query.sql"
    assert paths == [first["path"], second["path"]]


def test_sql_evidence_paths_are_tracked_text_artifacts():
    assert is_tracked_sql_evidence_path("queries/user_sql/source_query.sql") is True
    assert is_tracked_sql_evidence_path("queries/generated/sample.sql") is True
    assert is_tracked_sql_evidence_path("queries/sql_evidence_manifest.json") is True
    assert is_tracked_sql_evidence_path("data/raw/sample.feather") is False
    assert is_tracked_sql_evidence_path("queries/generated/cache.pkl") is False
