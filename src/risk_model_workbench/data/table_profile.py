"""Select-only remote table profiling helpers for feature-selection intake."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol


class SelectQueryClient(Protocol):
    def query(self, sql: str) -> Any:
        """Run a select-only SQL statement and return rows or a dataframe-like object."""


class TableProfileError(RuntimeError):
    """Raised when select-only table profiling cannot produce required evidence."""


@dataclass(frozen=True)
class ProfileQuery:
    purpose: str
    sql: str


def _first_row(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if hasattr(result, "to_dict"):
        try:
            rows = result.to_dict("records")
            return dict(rows[0]) if rows else {}
        except TypeError:
            pass
    if isinstance(result, list):
        return dict(result[0]) if result else {}
    if isinstance(result, tuple):
        return dict(result[0]) if result and isinstance(result[0], dict) else {}
    if isinstance(result, dict):
        if "rows" in result and isinstance(result["rows"], list):
            return dict(result["rows"][0]) if result["rows"] else {}
        return dict(result)
    return {}


def _rows(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    if hasattr(result, "to_dict"):
        try:
            return [dict(row) for row in result.to_dict("records")]
        except TypeError:
            pass
    if isinstance(result, list):
        return [dict(row) for row in result if isinstance(row, dict)]
    if isinstance(result, dict) and isinstance(result.get("rows"), list):
        return [dict(row) for row in result["rows"] if isinstance(row, dict)]
    row = _first_row(result)
    return [row] if row else []


def _int_value(row: dict[str, Any], *names: str) -> int | None:
    for name in names:
        if name in row and row[name] is not None:
            try:
                return int(row[name])
            except (TypeError, ValueError):
                return None
    return None


def _quote(value: str) -> str:
    return value.replace("'", "''")


def build_profile_queries(
    table_name: str,
    *,
    split_column: str | None = None,
    target_column: str | None = None,
    random_columns: list[str] | None = None,
) -> list[ProfileQuery]:
    queries = [
        ProfileQuery("row_count", f"select count(*) as row_count from {table_name}"),
    ]
    if split_column:
        queries.append(
            ProfileQuery(
                "split_distribution",
                f"select {split_column} as split_value, count(*) as row_count from {table_name} group by {split_column}",
            )
        )
    if target_column:
        queries.append(
            ProfileQuery(
                "label_valid_count",
                f"select count(*) as label_valid_count from {table_name} where {target_column} in (0, 1)",
            )
        )
    for column in random_columns or []:
        queries.append(
            ProfileQuery(
                f"random_column:{column}",
                (
                    f"select '{_quote(column)}' as column_name, min({column}) as min_value, "
                    f"max({column}) as max_value, sum(case when {column} is null then 1 else 0 end) as null_count, "
                    f"count(*) as row_count from {table_name}"
                ),
            )
        )
    return queries


def profile_remote_table(
    table_name: str,
    *,
    query_client: SelectQueryClient,
    split_column: str | None = None,
    target_column: str | None = None,
    random_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Profile a remote table using bounded select-only queries."""
    query_results: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "status": "ok",
        "profile_type": "remote_table_select_only",
        "table": table_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": None,
        "split_column": split_column,
        "split_distribution": {},
        "target_column": target_column,
        "label_valid_count": None,
        "random_columns": [],
        "queries": query_results,
    }

    for query in build_profile_queries(
        table_name,
        split_column=split_column,
        target_column=target_column,
        random_columns=random_columns,
    ):
        if not query.sql.lstrip().lower().startswith("select"):
            raise TableProfileError(f"profile query must be select-only: {query.purpose}")
        result = query_client.query(query.sql)
        query_results.append({"purpose": query.purpose, "sql": query.sql})
        if query.purpose == "row_count":
            payload["row_count"] = _int_value(_first_row(result), "row_count", "cnt", "count")
        elif query.purpose == "split_distribution":
            payload["split_distribution"] = {
                str(row.get("split_value")): int(row.get("row_count", 0))
                for row in _rows(result)
            }
        elif query.purpose == "label_valid_count":
            payload["label_valid_count"] = _int_value(_first_row(result), "label_valid_count", "row_count")
        elif query.purpose.startswith("random_column:"):
            row = _first_row(result)
            payload["random_columns"].append(
                {
                    "column": str(row.get("column_name") or query.purpose.split(":", 1)[1]),
                    "min_value": row.get("min_value"),
                    "max_value": row.get("max_value"),
                    "null_count": _int_value(row, "null_count"),
                    "row_count": _int_value(row, "row_count"),
                }
            )

    if payload["row_count"] is None:
        raise TableProfileError(f"row_count profile failed for {table_name}")
    return payload


def build_static_table_profile(
    table_name: str,
    *,
    row_count: int | None = None,
    column_count: int | None = None,
    feature_count: int | None = None,
    source: str = "static_metadata",
    status: str = "metadata_only",
    note: str = "",
) -> dict[str, Any]:
    """Build profile evidence when live select profiling is unavailable."""
    payload: dict[str, Any] = {
        "status": status,
        "profile_type": "remote_table_static_metadata",
        "table": table_name,
        "source": source,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": row_count,
        "column_count": column_count,
        "candidate_feature_count": feature_count,
    }
    if note:
        payload["note"] = note
    return payload


def write_table_profile(profile: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(profile, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path
