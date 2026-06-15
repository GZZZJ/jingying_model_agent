import json
from pathlib import Path

import yaml

from risk_model_workbench.cli import main
from risk_model_workbench.progress import ProgressReporter, load_progress_events, load_progress_summary
from risk_model_workbench.state import create_run_state, save_run_state


def test_progress_reporter_writes_chinese_message_and_run_state(tmp_path):
    run_path = tmp_path / "project" / "runs" / "run1"
    state = create_run_state(tmp_path / "project", run_id="run1", workflow="full_modeling")
    save_run_state(run_path, state)

    reporter = ProgressReporter(run_path, "d01_d02_screening", emit_terminal=False)
    reporter.emit(
        step="d01_done",
        message="表 9/50：D01 完成，保留 86/120 个变量",
        current=9,
        total=50,
        metrics={"table": "demo.table", "d01_remain": 86},
    )

    events = load_progress_events(run_path)
    assert events[-1]["message"] == "表 9/50：D01 完成，保留 86/120 个变量"
    assert events[-1]["percent"] == 18.0
    assert events[-1]["metrics"]["d01_remain"] == 86

    summary = load_progress_summary(run_path)
    assert summary["stage_label"] == "特征筛选-D01/D02"
    assert summary["latest_event"]["message"].startswith("表 9/50")

    updated = yaml.safe_load((run_path / "run_state.yml").read_text(encoding="utf-8"))
    progress = updated["stages"]["d01_d02_screening"]["progress"]
    assert progress["message"] == "表 9/50：D01 完成，保留 86/120 个变量"
    assert progress["percent"] == 18.0


def test_run_status_progress_outputs_chinese_summary(tmp_path, capsys):
    project = tmp_path / "project"
    run_path = project / "runs" / "run1"
    state = create_run_state(project, run_id="run1", workflow="full_modeling")
    state["current_stage"] = "feature_refine"
    save_run_state(run_path, state)
    ProgressReporter(run_path, "feature_refine", emit_terminal=False).emit(
        step="global_corr_done",
        message="全局相关性筛选完成，保留 732 个，剔除 118 个",
        percent=45,
    )

    assert main(["run", "status", "--project", str(project), "--run-id", "run1", "--progress"]) == 0
    output = capsys.readouterr().out
    assert "current_stage: 特征精筛" in output
    assert "progress: 45%" in output
    assert "全局相关性筛选完成" in output


def test_run_watch_once_outputs_recent_events(tmp_path, capsys):
    project = tmp_path / "project"
    run_path = project / "runs" / "run1"
    state = create_run_state(project, run_id="run1", workflow="full_modeling")
    save_run_state(run_path, state)
    ProgressReporter(run_path, "feature_metadata", emit_terminal=False).emit(
        step="table_metadata_done",
        message="表 1/3：元数据完成，候选字段 20 个",
        current=1,
        total=3,
    )

    assert main(["run", "watch", "--project", str(project), "--run-id", "run1", "--once"]) == 0
    output = capsys.readouterr().out
    assert "特征元数据" in output
    assert "表 1/3：元数据完成" in output


def test_sample_check_emits_stage_progress(tmp_path, capsys):
    project = _make_project(tmp_path)
    assert main(["run", "init", "--project", str(project), "--workflow", "full_modeling", "--run-id", "run1"]) == 0
    capsys.readouterr()

    assert main(["sample", "check", "--project", str(project), "--run-id", "run1"]) == 0
    output = capsys.readouterr().out
    assert "[RMW] 样本检查" in output
    assert "样本检查完成" in output

    events_path = project / "runs" / "run1" / "audit" / "progress_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert any(event["stage"] == "sample_check" and event["status"] == "scaffold" for event in events)


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "demo_project"
    for directory in ["configs", "queries", "runs", "reports", "docs"]:
        (project / directory).mkdir(parents=True, exist_ok=True)
    (project / "project.yml").write_text(
        "\n".join(
            [
                "project:",
                "  name: demo_project",
                "  display_name: Demo Project",
                "data:",
                "  source_table: demo.sample",
                "  id_columns:",
                "    - uid",
                "  target_column: label",
                "  time_column: event_time",
                "  period_column: ds",
                "segments:",
                "  - name: all",
                "    display_name: All",
                "    filter: null",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return project
