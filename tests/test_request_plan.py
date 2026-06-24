from pathlib import Path

from risk_model_workbench.cli import main
from risk_model_workbench.planning import create_execution_plan
from risk_model_workbench.request import parse_model_request, validate_model_request


def _request_doc(**overrides):
    metadata = {
        "request_id": "pytest-request",
        "project": "pytest-project",
        "workflow": "full_modeling",
        "target_column": "target",
        "id_columns": ["uid", "sample_date"],
        "split_column": "final_flag",
        "experiments": [{"name": "baseline_all"}],
        "evaluation": {"metrics": ["auc", "ks"], "champions": []},
        "reports": {"outputs": ["model_report.md"]},
    }
    metadata.update(overrides)
    return {"path": "/tmp/pytest-request.md", "metadata": metadata, "body": ""}


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
    assert "feature_prescreen" in task_ids
    assert "train_baseline_all" in task_ids
    assert task_ids[-1] == "report_final"
    assert plan["scenario_profile"] == "fujie_gcard_main_lgbm"
    assert "sql_review_gate" in plan["stage_steps"]["build_wide_sql"]
    assert "feature_availability_filter" in plan["stage_steps"]["feature_refine"]
    assert "constant_value_filter" in plan["stage_steps"]["feature_refine"]
    assert "random_noise_importance" in plan["stage_steps"]["feature_refine"]
    assert "null_importance_filter" in plan["stage_steps"]["feature_refine"]
    assert "baseline_importance_filter" in plan["stage_steps"]["feature_refine"]
    assert plan["step_params"]["constant_value_filter"]["max_unique_values"] == 1
    assert "hier_ranknet_training" not in {step for steps in plan["stage_steps"].values() for step in steps}


def test_profile_defaults_are_resolved_into_plan_metadata():
    request_doc = _request_doc(scenario_profile="inloan_behavior_card")

    plan = create_execution_plan(request_doc, "projects/2026-05-fujie-gcard-v1")
    sample_task = next(task for task in plan["tasks"] if task["type"] == "sample_check")

    assert plan["scenario_profile"] == "inloan_behavior_card"
    assert "account_status_distribution" in plan["stage_steps"]["sample_check"]
    assert "field_contract" in sample_task["step_ids"]
    assert plan["step_params"]["psi_filter"]["max_psi"] == 0.25


def test_business_domain_defaults_are_resolved_into_profile_metadata():
    acquisition_plan = create_execution_plan(_request_doc(business_domain="acquisition"), "projects/2026-05-fujie-gcard-v1")
    operation_plan = create_execution_plan(
        _request_doc(business_domain="inloan_operation"),
        "projects/2026-05-fujie-gcard-v1",
    )

    assert acquisition_plan["scenario_profile"] == "acquisition"
    assert "channel_distribution" in acquisition_plan["stage_steps"]["sample_check"]
    assert "hier_ranknet_training" in {step["id"] for step in acquisition_plan["planned_steps"]}
    assert operation_plan["scenario_profile"] == "inloan_operation"
    assert "segment_metrics" in operation_plan["stage_steps"]["evaluate"]
    assert "fujie_gcard_main_lgbm" != operation_plan["scenario_profile"]


def test_request_stage_steps_and_step_params_override_profile_defaults():
    request_doc = _request_doc(
        scenario_profile="inloan_behavior_card",
        stage_steps={"sample_check": ["field_contract"]},
        step_params={"psi_filter": {"max_psi": 0.3}},
    )

    plan = create_execution_plan(request_doc, "projects/2026-05-fujie-gcard-v1")
    sample_task = next(task for task in plan["tasks"] if task["type"] == "sample_check")

    assert plan["stage_steps"]["sample_check"] == ["field_contract"]
    assert sample_task["step_ids"] == ["field_contract"]
    assert plan["step_params"]["psi_filter"]["max_psi"] == 0.3


def test_unknown_profile_and_step_fail_validation():
    unknown_profile = validate_model_request(_request_doc(scenario_profile="unknown_profile"))
    unknown_domain = validate_model_request(_request_doc(business_domain="unknown_domain"))
    unknown_step = validate_model_request(_request_doc(stage_steps={"sample_check": ["unknown_step"]}))

    assert unknown_profile["status"] == "failed"
    assert "unknown scenario_profile" in unknown_profile["errors"][0]
    assert unknown_domain["status"] == "failed"
    assert "unknown business_domain" in unknown_domain["errors"][0]
    assert unknown_step["status"] == "failed"
    assert "unknown step id" in unknown_step["errors"][0]


def test_planned_steps_do_not_generate_task_commands():
    request_doc = _request_doc(scenario_profile="acquisition_conversion")

    plan = create_execution_plan(request_doc, "projects/2026-05-fujie-gcard-v1")
    planned_ids = {step["id"] for step in plan["planned_steps"]}

    assert "hier_ranknet_training" in planned_ids
    assert all("hier_ranknet_training" not in task["step_ids"] for task in plan["tasks"])
    assert all("hier_ranknet" not in " ".join(task["command"]["args"]) for task in plan["tasks"])


def test_request_without_id_columns_can_use_project_contract():
    request_doc = _request_doc(id_columns=[])

    result = validate_model_request(request_doc, Path("projects/2026-05-fujie-gcard-v1"))

    assert result["status"] == "ok"
    assert "request id_columns omitted; using project.yml data.id_columns" in result["warnings"]


def test_request_without_id_columns_fails_without_project_contract():
    request_doc = _request_doc(id_columns=[])

    result = validate_model_request(request_doc)

    assert result["status"] == "failed"
    assert "missing required field: id_columns" in result["errors"][-1]


def test_task_mode_label_is_plan_compatible():
    request_doc = _request_doc(task_mode="完整建模")

    plan = create_execution_plan(request_doc, "projects/2026-05-fujie-gcard-v1")

    assert plan["workflow"] == "full_modeling"
    assert plan["request_id"] == "pytest-request"


def test_experiment_description_derives_baseline_experiment():
    request_doc = _request_doc(experiments=[], experiment_description="先跑一个全客群 baseline。")

    validation = validate_model_request(request_doc, Path("projects/2026-05-fujie-gcard-v1"))
    plan = create_execution_plan(request_doc, "projects/2026-05-fujie-gcard-v1")
    task_ids = [task["task_id"] for task in plan["tasks"]]

    assert validation["status"] == "ok"
    assert "train_baseline_from_description" in task_ids


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
