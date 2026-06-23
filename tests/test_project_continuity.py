from pathlib import Path

import yaml

from risk_model_workbench.cli import main
from risk_model_workbench.project_state import audit_run
from risk_model_workbench.state import mark_stage_done, register_artifact


def test_project_status_writes_project_state(tmp_path):
    project = _make_project(tmp_path)

    assert main(["run", "init", "--project", str(project), "--workflow", "full_modeling", "--run-id", "run1"]) == 0
    assert main(["project", "status", "--project", str(project), "--write-state"]) == 0

    state = yaml.safe_load((project / "project_state.yml").read_text(encoding="utf-8"))
    assert state["active_run_id"] == "run1"
    assert state["last_verified_commands"] == [f"rmw project status --project {project}"]
    assert state["next_actions"]


def test_project_update_state_appends_resume_metadata(tmp_path):
    project = _make_project(tmp_path)

    assert (
        main(
            [
                "project",
                "update-state",
                "--project",
                str(project),
                "--active-run-id",
                "run1",
                "--objective",
                "standardize run handoff",
                "--next-action",
                "write a handoff before switching sessions",
                "--risk",
                "imported artifacts are not local rerun evidence",
            ]
        )
        == 0
    )

    state = yaml.safe_load((project / "project_state.yml").read_text(encoding="utf-8"))
    assert state["active_run_id"] == "run1"
    assert state["current_objective"] == "standardize run handoff"
    assert state["next_actions"] == ["write a handoff before switching sessions"]
    assert state["risks"] == ["imported artifacts are not local rerun evidence"]


def test_handoff_write_records_source_of_truth(tmp_path):
    project = _make_project(tmp_path)

    assert main(["run", "init", "--project", str(project), "--workflow", "full_modeling", "--run-id", "run1"]) == 0
    assert main(["handoff", "write", "--project", str(project), "--run-id", "run1", "--output", "handoffs/manual.md"]) == 0

    handoff = project / "handoffs" / "manual.md"
    text = handoff.read_text(encoding="utf-8")
    assert "runs/run1/run_state.yml" in text
    assert "runs/run1/audit/artifact_manifest.json" in text
    state = yaml.safe_load((project / "project_state.yml").read_text(encoding="utf-8"))
    assert state["last_handoff"] == str(handoff)
    assert state["next_actions"]


def test_lesson_add_project_scope(tmp_path):
    project = _make_project(tmp_path)

    assert (
        main(
            [
                "lesson",
                "add",
                "--project",
                str(project),
                "--title",
                "SQL approval gate",
                "--kind",
                "guardrail",
                "--body",
                "Dry-run SQL must be reviewed before DP pulls.",
                "--tag",
                "dp",
            ]
        )
        == 0
    )

    lessons = (project / "docs" / "lessons.md").read_text(encoding="utf-8")
    assert "## SQL approval gate" in lessons
    assert "Dry-run SQL must be reviewed before DP pulls." in lessons
    assert "- tags: dp" in lessons


def test_run_audit_classifies_completed_stage(tmp_path):
    project = _make_project(tmp_path)

    assert main(["run", "init", "--project", str(project), "--workflow", "full_modeling", "--run-id", "run1"]) == 0
    run_path = project / "runs" / "run1"
    artifact = run_path / "sample_check" / "sample_summary.json"
    report = run_path / "sample_check" / "sample_check_report.md"
    artifact.write_text('{"status": "done"}\n', encoding="utf-8")
    report.write_text("# Sample Check\n\nstatus: done\n", encoding="utf-8")
    register_artifact(run_path, "sample_check", "sample_check/sample_summary.json")
    register_artifact(run_path, "sample_check", "sample_check/sample_check_report.md")
    mark_stage_done(run_path, "sample_check")

    audit = audit_run(project, "run1", stage="sample_check")

    assert audit["verdict"] == "complete"
    assert audit["stages"][0]["verdict"] == "complete"
    assert main(["run", "audit", "--project", str(project), "--run-id", "run1", "--stage", "sample_check"]) == 0


def test_retrospective_write_is_explicit_checkpoint(tmp_path):
    project = _make_project(tmp_path)

    assert main(["run", "init", "--project", str(project), "--workflow", "full_modeling", "--run-id", "run1"]) == 0
    assert (
        main(
            [
                "retrospective",
                "write",
                "--project",
                str(project),
                "--run-id",
                "run1",
                "--scope",
                "session",
                "--note",
                "Completed continuity checkpoint work.",
                "--lesson",
                "Do not infer session end; write explicit checkpoints.",
                "--output",
                "retrospectives/session.md",
            ]
        )
        == 0
    )

    retrospective = project / "retrospectives" / "session.md"
    text = retrospective.read_text(encoding="utf-8")
    assert "- trigger: explicit" in text
    assert "Do not infer session end; write explicit checkpoints." in text
    state = yaml.safe_load((project / "project_state.yml").read_text(encoding="utf-8"))
    assert state["last_retrospective"] == str(retrospective)


def test_stage_retrospective_requires_stage(tmp_path):
    project = _make_project(tmp_path)

    assert main(["retrospective", "write", "--project", str(project), "--scope", "stage"]) == 1


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
