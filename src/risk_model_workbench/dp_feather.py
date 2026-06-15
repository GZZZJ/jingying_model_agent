"""DP query caching helpers for local feather datasets."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def resolve_project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def relative_display(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def default_dataset_paths(
    project_dir: Path,
    *,
    dataset_id: str,
    data_dir: str | Path = "data/local/dp_feather",
    metadata_dir: str | Path = "data/profile/dp_feather_datasets",
) -> tuple[Path, Path]:
    feather_path = resolve_project_path(project_dir, data_dir) / f"{dataset_id}.feather"
    metadata_path = resolve_project_path(project_dir, metadata_dir) / f"{dataset_id}.json"
    return feather_path, metadata_path


def write_dataset_metadata(
    *,
    project_dir: Path,
    metadata_path: Path,
    feather_path: Path,
    dataset_id: str,
    description: str,
    sql: str,
    status: str,
    row_count: int | None = None,
    column_count: int | None = None,
    columns: list[str] | None = None,
    note: str = "",
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset_id": dataset_id,
        "description": description,
        "status": status,
        "created_or_updated_at": datetime.now().isoformat(timespec="seconds"),
        "storage": {
            "feather_path": relative_display(feather_path, project_dir),
            "gitignored": True,
        },
        "dimensions": {
            "rows": row_count,
            "columns": column_count,
        },
        "sql": sql,
        "sql_sha256": sha256_text(sql),
    }
    if columns is not None:
        payload["columns"] = columns
    if note:
        payload["note"] = note
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_sql_review(
    *,
    dataset_id: str,
    description: str,
    feather_path: Path,
    metadata_path: Path,
    sql: str,
) -> None:
    print("=" * 80)
    print("DP SQL REVIEW REQUIRED")
    print("=" * 80)
    print(f"dataset_id: {dataset_id}")
    print(f"description: {description}")
    print(f"feather_path: {feather_path}")
    print(f"metadata_path: {metadata_path}")
    print("-" * 80)
    print(sql.rstrip())
    print("-" * 80)


def require_sql_approval(
    *,
    dataset_id: str,
    description: str,
    feather_path: Path,
    metadata_path: Path,
    sql: str,
    sql_approved: bool,
) -> None:
    print_sql_review(
        dataset_id=dataset_id,
        description=description,
        feather_path=feather_path,
        metadata_path=metadata_path,
        sql=sql,
    )
    if sql_approved:
        print("[SQL] Approval flag received; running DP query.")
        return
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Refusing to query DP without SQL approval. Review the SQL above, then rerun with --sql-approved."
        )
    answer = input("Type APPROVE to run this DP query: ").strip()
    if answer != "APPROVE":
        raise RuntimeError("DP query cancelled before execution.")


def fetch_dp_query_to_feather(
    *,
    project_dir: Path,
    sql: str,
    dataset_id: str,
    description: str,
    feather_path: Path,
    metadata_path: Path,
    sql_approved: bool = False,
    progress: Any | None = None,
) -> Path:
    """Run a reviewed DP query, write feather locally, and record metadata."""
    if progress and not sql_approved:
        progress.emit(
            step="sql_review",
            status="waiting_for_approval",
            message=f"DP SQL 需要审批：{dataset_id}",
            metrics={"dataset_id": dataset_id, "metadata_path": relative_display(metadata_path, project_dir)},
        )
    require_sql_approval(
        dataset_id=dataset_id,
        description=description,
        feather_path=feather_path,
        metadata_path=metadata_path,
        sql=sql,
        sql_approved=sql_approved,
    )

    from tmlpatch.database import TMLSQLClient

    if progress:
        progress.emit(step="dp_query", message=f"开始执行 DP 查询：{dataset_id}", metrics={"dataset_id": dataset_id})
    client = TMLSQLClient()
    try:
        df = client.sql(sql).to_pandas()
    finally:
        client.stop()

    feather_path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_feather(feather_path)
    if progress:
        progress.emit(
            step="dp_query_done",
            message=f"DP 查询完成并写入 feather：{dataset_id}，{len(df)} 行 {len(df.columns)} 列",
            metrics={
                "dataset_id": dataset_id,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "feather_path": relative_display(feather_path, project_dir),
            },
        )
    write_dataset_metadata(
        project_dir=project_dir,
        metadata_path=metadata_path,
        feather_path=feather_path,
        dataset_id=dataset_id,
        description=description,
        sql=sql,
        status="ready",
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        columns=[str(column) for column in df.columns],
    )
    return feather_path


def load_or_fetch_dp_feather(
    *,
    project_dir: Path,
    sql: str,
    dataset_id: str,
    description: str,
    feather_path: Path,
    metadata_path: Path,
    refresh: bool = False,
    sql_approved: bool = False,
    progress: Any | None = None,
) -> pd.DataFrame:
    """Return a local feather-backed DP dataset, fetching only after SQL approval."""
    if refresh or not feather_path.exists() or not metadata_path.exists():
        write_dataset_metadata(
            project_dir=project_dir,
            metadata_path=metadata_path,
            feather_path=feather_path,
            dataset_id=dataset_id,
            description=description,
            sql=sql,
            status="sql_review_required" if refresh or not feather_path.exists() else "ready",
            note="Run with --sql-approved only after the displayed SQL has been reviewed.",
        )
    if refresh or not feather_path.exists():
        fetch_dp_query_to_feather(
            project_dir=project_dir,
            sql=sql,
            dataset_id=dataset_id,
            description=description,
            feather_path=feather_path,
            metadata_path=metadata_path,
            sql_approved=sql_approved,
            progress=progress,
        )
    if progress:
        progress.emit(
            step="read_feather",
            message=f"正在读取本地 feather：{dataset_id}",
            metrics={"dataset_id": dataset_id, "feather_path": relative_display(feather_path, project_dir)},
        )
    df = pd.read_feather(feather_path)
    if progress:
        progress.emit(
            step="read_feather_done",
            message=f"本地 feather 读取完成：{dataset_id}，{len(df)} 行 {len(df.columns)} 列",
            metrics={"dataset_id": dataset_id, "rows": int(len(df)), "columns": int(len(df.columns))},
        )
    write_dataset_metadata(
        project_dir=project_dir,
        metadata_path=metadata_path,
        feather_path=feather_path,
        dataset_id=dataset_id,
        description=description,
        sql=sql,
        status="ready",
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        columns=[str(column) for column in df.columns],
    )
    return df
