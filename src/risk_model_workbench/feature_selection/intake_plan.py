"""Sampling and feature-batch plan generation for feature-selection intake."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from risk_model_workbench.resource_planning import choose_uniform_sampling_ratio


DEFAULT_MAX_FEATURES_PER_BATCH = 1000


class IntakePlanError(RuntimeError):
    """Raised when an intake plan cannot be built safely."""


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _format_ratio(ratio: float) -> str:
    return f"{ratio:.12g}"


def _select_random_column(random_columns: Iterable[str] | None, preferred_random_column: str | None) -> str | None:
    columns = list(random_columns or [])
    if preferred_random_column:
        if preferred_random_column in columns or not columns:
            return preferred_random_column
    return columns[0] if columns else None


def build_sampling_plan(
    *,
    data_source_mode: str,
    total_rows: int,
    max_rows: int,
    random_columns: Iterable[str] | None = None,
    preferred_random_column: str | None = None,
    min_ratio: float | None = None,
    max_ratio: float = 1.0,
) -> dict[str, Any]:
    """Build a remote SQL or local row-selection sampling plan."""
    if data_source_mode not in {"remote_table", "local_feather"}:
        raise IntakePlanError(f"unsupported data_source_mode: {data_source_mode}")

    decision = choose_uniform_sampling_ratio(
        total_rows=total_rows,
        max_rows=max_rows,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
    )

    random_column = _select_random_column(random_columns, preferred_random_column)
    if data_source_mode == "remote_table" and decision.sampling_required and not random_column:
        raise IntakePlanError("remote uniform sampling requires a usable random column")

    ratio_text = _format_ratio(decision.ratio)
    if data_source_mode == "remote_table":
        method = "full_table_uniform_random" if decision.sampling_required else "full_table"
        sql_predicate = f"{random_column} < {ratio_text}" if decision.sampling_required else None
        local_row_selection = None
    else:
        method = "local_uniform_random" if decision.sampling_required else "local_full_scan"
        sql_predicate = None
        local_row_selection = {
            "method": "uniform_random_fraction" if decision.sampling_required else "all_rows",
            "fraction": decision.ratio,
            "max_rows": max_rows,
        }

    return {
        "data_source_mode": data_source_mode,
        "total_rows": total_rows,
        "max_rows": max_rows,
        "sampling_required": decision.sampling_required,
        "method": method,
        "ratio": decision.ratio,
        "estimated_rows": decision.estimated_rows,
        "limit": decision.limit,
        "random_column": random_column,
        "sql_predicate": sql_predicate,
        "local_row_selection": local_row_selection,
        "reason": decision.reason,
    }


def build_feature_batch_plan(
    *,
    feature_columns: Iterable[str],
    required_columns: Iterable[str] | None = None,
    max_features_per_batch: int = DEFAULT_MAX_FEATURES_PER_BATCH,
) -> dict[str, Any]:
    """Split candidate features into deterministic batches."""
    if max_features_per_batch <= 0:
        raise IntakePlanError("max_features_per_batch must be positive")

    required = _unique(required_columns or [])
    required_set = set(required)
    candidates = [column for column in _unique(feature_columns) if column not in required_set]
    batches: list[dict[str, Any]] = []
    for start in range(0, len(candidates), max_features_per_batch):
        batch_features = candidates[start : start + max_features_per_batch]
        batch_number = len(batches) + 1
        batches.append(
            {
                "batch_id": f"batch_{batch_number:03d}",
                "batch_index": batch_number,
                "feature_columns": batch_features,
                "feature_count": len(batch_features),
                "required_columns": required,
                "select_columns": _unique([*required, *batch_features]),
            }
        )

    return {
        "max_features_per_batch": max_features_per_batch,
        "required_columns": required,
        "total_feature_count": len(candidates),
        "batch_count": len(batches),
        "batches": batches,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def persist_intake_plan(
    run_dir: str | Path,
    *,
    sampling_plan: dict[str, Any] | None = None,
    batch_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist sampling, aggregate batch, and per-batch plan JSON files."""
    run_path = Path(run_dir)
    written: dict[str, Any] = {}

    if sampling_plan is not None:
        written["sampling_plan"] = _write_json(run_path / "feature_selection" / "sampling_plan.json", sampling_plan)

    if batch_plan is not None:
        written["batch_plan"] = _write_json(run_path / "feature_selection" / "batch_plan.json", batch_plan)
        batch_files: list[Path] = []
        for batch in batch_plan.get("batches", []):
            batch_path = run_path / "feature_selection" / "batches" / f"{batch['batch_id']}_plan.json"
            batch_files.append(_write_json(batch_path, batch))
        written["batch_files"] = batch_files

    return written
