from pathlib import Path
import shutil

from risk_model_workbench.cli import main


def test_sample_check_scaffold(tmp_path):
    project = "projects/2026-05-fujie-gcard-v1"
    run_id = "pytest_sample_scaffold"
    run_dir = Path(project) / "runs" / run_id
    try:
        main(["run", "init", "--project", project, "--workflow", "full_modeling", "--run-id", run_id, "--force"])
        assert main(["sample", "check", "--project", project, "--run-id", run_id]) == 0
        summary = run_dir / "sample_check" / "sample_summary.json"
        assert summary.exists()
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_sample_check_scaffold_when_remote_request_has_null_raw_path(tmp_path):
    project = "projects/2026-05-fujie-gcard-v1"
    request = "projects/2026-05-fujie-gcard-v1/requests/2026-06-25-resource-aware-gcard-wide-smoke.md"
    run_id = "pytest_sample_remote_scaffold"
    run_dir = Path(project) / "runs" / run_id
    try:
        assert main(["run", "init", "--project", project, "--workflow", "full_modeling", "--run-id", run_id, "--request", request, "--force"]) == 0
        assert main(["sample", "check", "--project", project, "--run-id", run_id]) == 0
        summary = run_dir / "sample_check" / "sample_summary.json"
        assert summary.exists()
        assert "local data not available" in summary.read_text(encoding="utf-8")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
