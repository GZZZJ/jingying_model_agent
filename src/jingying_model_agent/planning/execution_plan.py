"""Create execution plans from model request documents."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _as_list(value: Any, default: list[Any] | None = None) -> list[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return value
    return [value]


def _load_project_yaml(project_path: str | Path) -> dict[str, Any]:
    project_dir = Path(project_path)
    for name in ["project.yml", "project.yaml"]:
        path = project_dir / name
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return data if isinstance(data, dict) else {}
    return {}


def _project_champions(project_path: str | Path) -> list[Any]:
    config = _load_project_yaml(project_path)
    champions = config.get("champions") or {}
    if isinstance(champions, dict):
        return _as_list(champions.get("score_columns"))
    return _as_list(champions)


def _task(
    *,
    task_id: str,
    task_type: str,
    depends_on: list[str],
    args: list[str],
    outputs: list[str],
    workspace: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "type": task_type,
        "status": "pending",
        "workspace": workspace or f"tasks/{task_id}",
        "depends_on": depends_on,
        "command": {
            "executable": "jm",
            "args": args,
        },
        "outputs": outputs,
    }


def create_execution_plan(request_doc: dict[str, Any], project_path: str | Path) -> dict[str, Any]:
    """Create a deterministic execution plan from a parsed model request."""
    metadata = request_doc["metadata"]
    project = str(project_path)
    request_id = metadata["request_id"]
    workflow = metadata.get("workflow", "full_modeling")
    run_arg = "<run_id>"
    tasks: list[dict[str, Any]] = []

    sample_checks = _as_list(metadata.get("sample_checks"), default=["sample_check_001"])
    sample_task_ids: list[str] = []
    for index, item in enumerate(sample_checks, start=1):
        name = item.get("name") if isinstance(item, dict) else str(item)
        task_id = name or f"sample_check_{index:03d}"
        sample_task_ids.append(task_id)
        tasks.append(
            _task(
                task_id=task_id,
                task_type="sample_check",
                depends_on=[],
                args=["sample", "check", "--project", project, "--run-id", run_arg],
                outputs=["sample_check/sample_summary.json", "sample_check/sample_check_report.md"],
            )
        )

    feature_cfg = metadata.get("feature_selection") or {}
    feature_rounds = _as_list(feature_cfg.get("rounds"), default=["metadata", "d01_d02", "refine"])
    feature_task_ids: list[str] = []
    feature_dependency = sample_task_ids[-1:] if sample_task_ids else []
    for item in feature_rounds:
        name = item.get("name") if isinstance(item, dict) else str(item)
        normalized = name.replace("-", "_")
        if normalized == "metadata":
            task_id = "feature_metadata"
            args = ["feature", "metadata", "--project", project, "--run-id", run_arg]
            outputs = ["feature_selection/feature_table_summary.csv", "feature_selection/feature_columns.csv"]
        elif normalized in {"d01_d02", "d01d02"}:
            task_id = "feature_d01_d02"
            args = ["feature", "d01-d02", "--project", project, "--run-id", run_arg, "--dry-run-sql"]
            outputs = ["feature_selection/d01_d02_run_summary.json", "feature_selection/d01_d02_final_remain_features.json"]
        elif normalized == "refine":
            task_id = "feature_refine"
            args = ["feature", "refine", "--project", project, "--run-id", run_arg, "--dry-run-sql"]
            outputs = ["feature_selection/stage_summary.json", "feature_selection/final_features.txt"]
        else:
            task_id = f"feature_{normalized}"
            args = ["feature", normalized, "--project", project, "--run-id", run_arg]
            outputs = []
        tasks.append(
            _task(
                task_id=task_id,
                task_type="feature_selection",
                depends_on=feature_dependency,
                args=args,
                outputs=outputs,
            )
        )
        feature_dependency = [task_id]
        feature_task_ids.append(task_id)

    train_dep = feature_task_ids[-1:] or sample_task_ids[-1:]
    train_task_ids: list[str] = []
    for experiment in metadata.get("experiments", []):
        name = experiment.get("name") if isinstance(experiment, dict) else str(experiment)
        task_id = f"train_{name}"
        train_task_ids.append(task_id)
        tasks.append(
            _task(
                task_id=task_id,
                task_type="train",
                depends_on=train_dep,
                args=["train", "--project", project, "--run-id", run_arg, "--experiment", name],
                outputs=[
                    f"modeling/{name}/model.pkl",
                    f"modeling/{name}/prediction.parquet",
                    f"modeling/{name}/train_metrics.json",
                ],
            )
        )

    tasks.append(
        _task(
            task_id="evaluate_final",
            task_type="evaluate",
            depends_on=train_task_ids,
            args=["evaluate", "--project", project, "--run-id", run_arg],
            outputs=["evaluation/evaluation_summary.json"],
        )
    )

    champions = _as_list((metadata.get("evaluation") or {}).get("champions"))
    if not champions:
        champions = _project_champions(project_path)
    champion = champions[-1] if champions else None
    compare_args = ["compare", "--project", project, "--run-id", run_arg]
    if champion:
        compare_args.extend(["--champion", str(champion)])
    tasks.append(
        _task(
            task_id="compare_final",
            task_type="compare",
            depends_on=["evaluate_final"],
            args=compare_args,
            outputs=["evaluation/champion_challenger.json"],
        )
    )

    tasks.append(
        _task(
            task_id="report_final",
            task_type="report",
            depends_on=["compare_final"],
            args=["report", "--project", project, "--run-id", run_arg],
            outputs=["reports/model_report.md", "reports/model_card.md", "reports/executive_summary.md"],
        )
    )

    return {
        "version": 1,
        "plan_id": f"{request_id}_plan",
        "request_id": request_id,
        "request_path": request_doc["path"],
        "project": project,
        "workflow": workflow,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id_placeholder": run_arg,
        "tasks": tasks,
        "improvement_log": "audit/improvement_candidates.md",
    }


def save_execution_plan(plan: dict[str, Any], path: str | Path) -> Path:
    """Write an execution plan YAML file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.dump(plan, handle, Dumper=_NoAliasDumper, allow_unicode=True, sort_keys=False)
    return output_path
