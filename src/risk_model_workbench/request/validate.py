"""Validate model request documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from risk_model_workbench.config import load_yaml
from risk_model_workbench.paths import project_config_path


REQUIRED_FIELDS = [
    "request_id",
    "project",
    "target_column",
    "id_columns",
    "split_column",
    "experiments",
    "evaluation",
    "reports",
]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def validate_model_request(request_doc: dict[str, Any], project_dir: str | Path | None = None) -> dict[str, Any]:
    """Return validation errors and warnings for a parsed model request."""
    metadata = request_doc.get("metadata", {})
    errors: list[str] = []
    warnings: list[str] = []

    for field in REQUIRED_FIELDS:
        if field not in metadata or metadata.get(field) in (None, "", []):
            errors.append(f"missing required field: {field}")

    if "experiments" in metadata and not isinstance(metadata.get("experiments"), list):
        errors.append("experiments must be a list")
    if "id_columns" in metadata and not isinstance(metadata.get("id_columns"), list):
        errors.append("id_columns must be a list")

    evaluation = metadata.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        errors.append("evaluation must be a mapping")
    elif not _as_list(evaluation.get("metrics")):
        warnings.append("evaluation.metrics is empty")

    reports = metadata.get("reports") or {}
    if not isinstance(reports, dict):
        errors.append("reports must be a mapping")
    elif not _as_list(reports.get("outputs")):
        warnings.append("reports.outputs is empty")

    if project_dir:
        project_path = Path(project_dir).resolve()
        config_path = project_config_path(project_path)
        if not config_path.exists():
            errors.append(f"missing project config: {config_path}")
        else:
            project_config = load_yaml(config_path)
            configured_project = project_config.get("project", {}).get("name")
            configured_data = project_config.get("data", {})
            if configured_project and metadata.get("project") and metadata["project"] != configured_project:
                warnings.append(f"request project differs from project.yml: {metadata['project']} != {configured_project}")
            if configured_data.get("target_column") and metadata.get("target_column") != configured_data.get("target_column"):
                warnings.append(
                    f"request target_column differs from project.yml: {metadata.get('target_column')} != {configured_data.get('target_column')}"
                )
            configured_ids = configured_data.get("id_columns") or []
            if configured_ids and metadata.get("id_columns") and metadata.get("id_columns") != configured_ids:
                warnings.append(f"request id_columns differs from project.yml: {metadata.get('id_columns')} != {configured_ids}")

    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
    }
