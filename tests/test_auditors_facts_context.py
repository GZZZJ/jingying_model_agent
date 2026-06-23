import json
from pathlib import Path

import yaml

from risk_model_workbench.cli import main


def test_auditors_are_read_only_and_report_json(tmp_path, capsys):
    project = _make_project(tmp_path)
    run_path = _init_run(project, "run1")
    before_state = (run_path / "run_state.yml").read_text(encoding="utf-8")
    before_manifest = (run_path / "audit" / "artifact_manifest.json").read_text(encoding="utf-8")

    capsys.readouterr()
    assert main(["auditor", "list", "--json"]) == 0
    auditors = json.loads(capsys.readouterr().out)
    assert {item["name"] for item in auditors} >= {"sql_review", "artifact_consistency", "report_gap_scan", "config_risk"}

    assert main(["auditor", "run", "artifact_consistency", "--project", str(project), "--run-id", "run1", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["auditor"] == "artifact_consistency"
    assert result["read_only"] is True

    assert (run_path / "run_state.yml").read_text(encoding="utf-8") == before_state
    assert (run_path / "audit" / "artifact_manifest.json").read_text(encoding="utf-8") == before_manifest


def test_fact_store_requires_source_and_lists_json(tmp_path, capsys):
    project = _make_project(tmp_path)

    assert (
        main(
            [
                "fact",
                "add",
                "--project",
                str(project),
                "--category",
                "business_definition",
                "--statement",
                "样本定义来自 project.yml 的 data.source_table。",
                "--source-path",
                "project.yml",
            ]
        )
        == 0
    )
    assert (project / "project_facts.yml").exists()

    capsys.readouterr()
    assert main(["fact", "list", "--project", str(project), "--category", "business_definition", "--json"]) == 0
    facts = json.loads(capsys.readouterr().out)
    assert facts[0]["id"] == "fact_0001"
    assert facts[0]["source_path"] == "project.yml"

    assert (
        main(
            [
                "fact",
                "add",
                "--project",
                str(project),
                "--category",
                "lesson",
                "--statement",
                "missing source should fail",
                "--source-path",
                "missing.md",
            ]
        )
        == 1
    )


def test_context_snapshot_and_handoff_reference(tmp_path):
    project = _make_project(tmp_path)
    _init_run(project, "run1")
    assert (
        main(
            [
                "fact",
                "add",
                "--project",
                str(project),
                "--category",
                "label_definition",
                "--statement",
                "标签字段为 label。",
                "--source-path",
                "project.yml",
            ]
        )
        == 0
    )

    assert main(["context", "snapshot", "--project", str(project), "--run-id", "run1", "--markdown"]) == 0
    snapshot_path = project / "runs" / "run1" / "audit" / "context_snapshot.json"
    markdown_path = project / "runs" / "run1" / "audit" / "context_snapshot.md"
    assert snapshot_path.exists()
    assert markdown_path.exists()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["run_id"] == "run1"
    assert snapshot["workflow"]["stage_contracts"]
    assert snapshot["artifact_manifest"]["artifact_count"] >= 1
    assert snapshot["facts"][0]["statement"] == "标签字段为 label。"

    assert (
        main(
            [
                "handoff",
                "write",
                "--project",
                str(project),
                "--run-id",
                "run1",
                "--output",
                "handoffs/manual.md",
                "--context-snapshot",
            ]
        )
        == 0
    )
    handoff = (project / "handoffs" / "manual.md").read_text(encoding="utf-8")
    assert "runs/run1/audit/context_snapshot.json" in handoff


def _init_run(project: Path, run_id: str) -> Path:
    assert main(["run", "init", "--project", str(project), "--workflow", "full_modeling", "--run-id", run_id]) == 0
    return project / "runs" / run_id


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "demo_project"
    for directory in ["configs", "queries", "runs", "reports", "docs", "handoffs"]:
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
