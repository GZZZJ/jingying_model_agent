"""Local feather metadata profiling for resource-aware feature intake."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


class LocalFeatherProfileError(RuntimeError):
    """Raised when a local feather source cannot satisfy the intake contract."""


def _as_list(values: Iterable[str] | None) -> list[str]:
    return list(values or [])


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise LocalFeatherProfileError("pandas is required to profile local feather files") from exc
    return pd


def _json_key(value: Any) -> str:
    if value is None:
        return "null"
    try:
        if value != value:  # NaN
            return "null"
    except Exception:
        pass
    return str(value)


def profile_local_feather(
    feather_path: str | Path,
    *,
    required_columns: Iterable[str] | None = None,
    split_column: str | None = None,
    target_column: str | None = None,
    feature_exclude_columns: Iterable[str] | None = None,
    feature_columns: Iterable[str] | None = None,
    max_distribution_values: int = 50,
) -> dict[str, Any]:
    """Return a JSON-serializable profile summary for a local feather file."""
    path = Path(feather_path)
    if path.suffix.lower() != ".feather":
        raise LocalFeatherProfileError("local feather source must be a .feather file")
    if not path.exists():
        raise LocalFeatherProfileError(f"local feather file does not exist: {path}")
    if not path.is_file():
        raise LocalFeatherProfileError(f"local feather path is not a file: {path}")

    pd = _import_pandas()
    try:
        dataframe = pd.read_feather(path)
    except Exception as exc:
        raise LocalFeatherProfileError(f"failed to read local feather file: {path}") from exc

    columns = [str(column) for column in dataframe.columns]
    required = _as_list(required_columns)
    missing = [column for column in required if column not in columns]
    if missing:
        raise LocalFeatherProfileError(f"missing required columns: {missing}")

    split_distribution: dict[str, int] | None = None
    if split_column and split_column in dataframe.columns:
        counts = dataframe[split_column].value_counts(dropna=False).head(max_distribution_values)
        split_distribution = {_json_key(index): int(value) for index, value in counts.items()}

    label_valid_count: int | None = None
    if target_column and target_column in dataframe.columns:
        label_valid_count = int(dataframe[target_column].notna().sum())

    excluded = set(_as_list(feature_exclude_columns)) | set(required)
    if feature_columns is None:
        candidate_feature_columns = [column for column in columns if column not in excluded]
    else:
        candidate_feature_columns = [column for column in feature_columns if column in columns and column not in excluded]

    stat = path.stat()
    return {
        "status": "ok",
        "path": str(path),
        "exists": True,
        "suffix": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "row_count": int(len(dataframe)),
        "column_count": int(len(columns)),
        "columns": columns,
        "required_columns": {
            "requested": required,
            "present": [column for column in required if column in columns],
            "missing": missing,
        },
        "split_column": split_column,
        "split_distribution": split_distribution or {},
        "target_column": target_column,
        "label_valid_count": label_valid_count,
        "candidate_feature_count": len(candidate_feature_columns),
        "candidate_feature_columns": candidate_feature_columns,
        "profile_type": "local_feather_summary",
    }


def write_local_feather_profile(profile: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path
