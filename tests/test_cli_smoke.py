import shutil
from pathlib import Path

from risk_model_workbench.cli import main


def test_cli_doctor():
    assert main(["doctor"]) == 0


def test_workflow_validate():
    assert main(["workflow", "validate", "--workflow", "full_modeling"]) == 0


def test_project_validate():
    assert main(["project", "validate", "--project", "projects/2026-05-fujie-gcard-v1"]) == 0


def test_run_status_subcommand():
    project = "projects/2026-05-fujie-gcard-v1"
    run_id = "pytest_run_status"
    run_dir = Path(project) / "runs" / run_id
    try:
        assert main(["run", "init", "--project", project, "--workflow", "full_modeling", "--run-id", run_id, "--force"]) == 0
        assert main(["run", "status", "--project", project, "--run-id", run_id]) == 0
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
