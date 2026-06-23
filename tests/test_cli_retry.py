from pathlib import Path

import yaml

import risk_model_workbench.cli as cli_module
from risk_model_workbench.cli import main
from risk_model_workbench.state import create_run_state, save_run_state


def test_run_audit_retries_transient_io_for_read_only_cli(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    project.mkdir()
    attempts = {"count": 0}

    def flaky_audit(project_dir: Path, run_id: str, *, stage: str | None = None):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("temporary IO timeout")
        return {
            "run_id": run_id,
            "workflow": "full_modeling",
            "run_status": "done",
            "verdict": "complete",
            "contract_source": "",
            "stages": [],
        }

    monkeypatch.setattr(cli_module, "audit_run", flaky_audit)

    assert main(["run", "audit", "--project", str(project), "--run-id", "run1"]) == 0
    assert attempts["count"] == 2
    assert "verdict: complete" in capsys.readouterr().out


def test_project_status_retries_transient_io_when_read_only(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    project.mkdir()
    attempts = {"count": 0}

    def flaky_summary(project_dir: Path, run_id: str | None = None):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("temporary IO timeout")
        return {
            "project": str(project_dir),
            "project_name": "demo",
            "display_name": "Demo",
            "status": "running",
            "active_run_id": run_id or "",
            "rules": {},
            "run": None,
        }

    monkeypatch.setattr(cli_module, "summarize_project", flaky_summary)

    assert main(["project", "status", "--project", str(project)]) == 0
    assert attempts["count"] == 2
    assert "project: Demo" in capsys.readouterr().out


def test_workflow_validate_retries_transient_io_for_read_only_cli(monkeypatch, capsys):
    original_load_yaml = cli_module.load_yaml
    attempts = {"count": 0}

    def flaky_load_yaml(path: Path):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("temporary IO timeout")
        return original_load_yaml(path)

    monkeypatch.setattr(cli_module, "load_yaml", flaky_load_yaml)

    assert main(["workflow", "validate", "--workflow", "full_modeling"]) == 0
    assert attempts["count"] == 2
    assert "workflow validation ok" in capsys.readouterr().out


def test_run_status_retries_transient_io_for_read_only_cli(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    run_path = project / "runs" / "run1"
    save_run_state(run_path, create_run_state(project, run_id="run1", workflow="full_modeling"))
    original_load_run_state = cli_module.load_run_state
    attempts = {"count": 0}

    def flaky_load_run_state(path: Path):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("temporary IO timeout")
        return original_load_run_state(path)

    monkeypatch.setattr(cli_module, "load_run_state", flaky_load_run_state)

    assert main(["run", "status", "--project", str(project), "--run-id", "run1"]) == 0
    assert attempts["count"] == 2
    assert "run_id: run1" in capsys.readouterr().out


def test_write_stage_cli_is_not_wrapped_in_safe_retry(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    assert main(["run", "init", "--project", str(project), "--workflow", "full_modeling", "--run-id", "run1"]) == 0

    def forbidden_retry(*args, **kwargs):
        raise AssertionError("write-stage commands must not use read-only retry")

    monkeypatch.setattr(cli_module, "run_with_retry", forbidden_retry, raising=False)

    assert main(["sample", "check", "--project", str(project), "--run-id", "run1"]) == 0


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
