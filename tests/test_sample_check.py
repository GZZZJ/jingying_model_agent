from pathlib import Path
import shutil

from jingying_model_agent.cli import main


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
