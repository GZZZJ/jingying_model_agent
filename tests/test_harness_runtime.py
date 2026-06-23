from pathlib import Path

import yaml

from risk_model_workbench.cli import main
from risk_model_workbench.harness.runtime import (
    classify_exception_message,
    register_action_artifact,
    run_with_retry,
    should_retry_failure,
    stage_action_done,
    stage_action_failed,
)
from risk_model_workbench.progress import load_progress_events
from risk_model_workbench.state import load_run_state


def test_stage_action_metadata_is_written_by_cli_stage(tmp_path):
    project = _make_project(tmp_path)
    run_path = _init_run(project, "run1")

    state = load_run_state(run_path)
    validate_config = state["stages"]["validate_config"]
    assert validate_config["status"] == "done"
    assert validate_config["action"]["id"] == "validate_config"
    assert validate_config["last_result"]["status"] == "done"

    assert main(["sample", "check", "--project", str(project), "--run-id", "run1"]) == 0

    state = load_run_state(run_path)
    sample = state["stages"]["sample_check"]
    assert sample["status"] == "scaffold"
    assert sample["action"]["id"] == "sample_check"
    assert sample["action"]["expected_inputs"]
    assert sample["action"]["artifact_rules"]
    assert sample["last_result"]["status"] == "scaffold"
    assert sample["last_result"]["failure_code"] == "scaffold_only"
    assert sample["failure_code"] == "scaffold_only"


def test_stage_action_failed_records_classified_failure(tmp_path):
    project = _make_project(tmp_path)
    run_path = _init_run(project, "run1")

    stage_action_failed(run_path, "build_wide_sql", "required feature list missing")

    state = load_run_state(run_path)
    stage = state["stages"]["build_wide_sql"]
    assert stage["status"] == "failed"
    assert stage["action"]["id"] == "build_wide_sql"
    assert stage["last_result"]["status"] == "failed"
    assert stage["last_result"]["failure_code"] == "data_missing"
    assert stage["failure_code"] == "data_missing"


def test_stage_action_result_tracks_artifacts_and_latest_decision(tmp_path):
    project = _make_project(tmp_path)
    run_path = _init_run(project, "run1")
    artifact = run_path / "sample_check" / "sample_summary.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"status": "scaffold"}\n', encoding="utf-8")

    entry = register_action_artifact(run_path, "sample_check", "sample_check/sample_summary.json")
    assert entry["path"] == "sample_check/sample_summary.json"

    from risk_model_workbench.state import append_decision

    append_decision(run_path, stage="sample_check", decision="scaffold", reason="local data missing")
    stage_action_done(run_path, "sample_check", scaffold=True, message="local data missing", retry_count=1)

    state = load_run_state(run_path)
    result = state["stages"]["sample_check"]["last_result"]
    assert result["retry_count"] == 1
    assert result["artifacts"][0]["path"] == "sample_check/sample_summary.json"
    assert result["artifacts"][0]["rule_matched"] is True
    assert result["decision"]["decision"] == "scaffold"
    assert result["decision"]["reason"] == "local data missing"

    done_events = [
        event
        for event in load_progress_events(run_path)
        if event["stage"] == "sample_check" and event["step"] == "action_done"
    ]
    assert done_events[-1]["metrics"]["failure_code"] == "scaffold_only"
    assert done_events[-1]["metrics"]["retry_count"] == 1
    assert done_events[-1]["metrics"]["artifact_count"] == 1


def test_stage_action_progress_records_failure_code_and_retry_count(tmp_path):
    project = _make_project(tmp_path)
    run_path = _init_run(project, "run1")

    stage_action_failed(run_path, "build_wide_sql", "temporary IO timeout", retry_count=2)

    failed_events = [
        event
        for event in load_progress_events(run_path)
        if event["stage"] == "build_wide_sql" and event["status"] == "failed"
    ]
    assert failed_events[-1]["step"] == "action_failed"
    assert failed_events[-1]["metrics"]["action_id"] == "build_wide_sql"
    assert failed_events[-1]["metrics"]["failure_code"] == "transient_io"
    assert failed_events[-1]["metrics"]["retry_count"] == 2


def test_failure_classification_and_retry_boundary():
    assert classify_exception_message("Refusing to query DP without SQL approval") == "sql_approval_required"
    assert classify_exception_message("No module named lightgbm") == "dependency_missing"
    assert classify_exception_message("input feather not found") == "data_missing"
    assert classify_exception_message("temporary IO timeout") == "transient_io"
    assert should_retry_failure("build_wide_sql", "transient_io", attempt=1) is False
    assert should_retry_failure("run_audit", "transient_io", attempt=1) is True
    assert should_retry_failure("run_audit", "sql_approval_required", attempt=1) is False


def test_run_with_retry_only_retries_safe_transient_io():
    attempts = {"count": 0}

    def flaky_read():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("temporary IO timeout")
        return "ok"

    result, retry_count = run_with_retry("run_audit", flaky_read)
    assert result == "ok"
    assert retry_count == 1
    assert attempts["count"] == 2

    unsafe_attempts = {"count": 0}

    def unsafe_write():
        unsafe_attempts["count"] += 1
        raise TimeoutError("temporary IO timeout")

    try:
        run_with_retry("build_wide_sql", unsafe_write)
    except TimeoutError:
        pass
    else:
        raise AssertionError("write-stage transient IO should not be retried by default")
    assert unsafe_attempts["count"] == 1


def _init_run(project: Path, run_id: str) -> Path:
    assert main(["run", "init", "--project", str(project), "--workflow", "full_modeling", "--run-id", run_id]) == 0
    return project / "runs" / run_id


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "demo_project"
    for directory in ["configs", "queries", "runs", "reports", "docs"]:
        (project / directory).mkdir(parents=True, exist_ok=True)
    (project / "project.yml").write_text(
        yaml.safe_dump(
            {
                "project": {"name": "demo_project", "display_name": "Demo Project"},
                "data": {
                    "source_table": "demo.sample",
                    "id_columns": ["uid"],
                    "target_column": "label",
                    "time_column": "event_time",
                    "period_column": "ds",
                },
                "segments": [{"name": "all", "display_name": "All", "filter": None}],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return project
