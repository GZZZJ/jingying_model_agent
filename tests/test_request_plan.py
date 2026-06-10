from pathlib import Path

from risk_model_workbench.cli import main
from risk_model_workbench.planning import create_execution_plan
from risk_model_workbench.request import parse_model_request, validate_model_request


def test_parse_and_validate_gcard_request_template():
    request_path = Path("projects/2026-05-fujie-gcard-v1/requests/model_request_template.md")
    request_doc = parse_model_request(request_path)
    result = validate_model_request(request_doc, Path("projects/2026-05-fujie-gcard-v1"))
    assert result["status"] == "ok"
    assert request_doc["metadata"]["request_id"] == "2026-06-fujie-gcard-baseline"


def test_create_execution_plan_from_request():
    request_path = Path("projects/2026-05-fujie-gcard-v1/requests/model_request_template.md")
    request_doc = parse_model_request(request_path)
    plan = create_execution_plan(request_doc, "projects/2026-05-fujie-gcard-v1")
    task_ids = [task["task_id"] for task in plan["tasks"]]
    assert "sample_check_profile" in task_ids
    assert "feature_d01_d02" in task_ids
    assert "train_baseline_all" in task_ids
    assert task_ids[-1] == "report_final"


def test_cli_request_validate_and_plan_create(tmp_path):
    output = tmp_path / "execution_plan.yml"
    assert main(
        [
            "request",
            "validate",
            "--project",
            "projects/2026-05-fujie-gcard-v1",
            "--request",
            "projects/2026-05-fujie-gcard-v1/requests/model_request_template.md",
        ]
    ) == 0
    assert main(
        [
            "plan",
            "create",
            "--project",
            "projects/2026-05-fujie-gcard-v1",
            "--request",
            "projects/2026-05-fujie-gcard-v1/requests/model_request_template.md",
            "--output",
            str(output),
        ]
    ) == 0
    assert output.exists()
