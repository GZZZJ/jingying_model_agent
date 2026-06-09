#!/usr/bin/env python3
"""Generate MaxCompute SQL to join D01/D02 remaining features into one wide table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_PROJECT_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from jingying_agent.wide_sql import generate_wide_sql


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build D01/D02 wide feature table SQL.")
    parser.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR), help="Project workspace directory.")
    parser.add_argument(
        "--remain-features",
        default="runs/d01_d02_batch_select/results/d01_d02_final_remain_features.json",
        help="Remaining feature JSON, relative to project-dir unless absolute.",
    )
    parser.add_argument(
        "--sql-output",
        default="queries/06_build_d01_d02_wide_table.sql",
        help="SQL output path, relative to project-dir unless absolute.",
    )
    parser.add_argument(
        "--feature-map-output",
        default="runs/d01_d02_batch_select/results/d01_d02_wide_feature_map.csv",
        help="Output feature mapping CSV, relative to project-dir unless absolute.",
    )
    parser.add_argument(
        "--summary-output",
        default="runs/d01_d02_batch_select/results/d01_d02_wide_sql_summary.json",
        help="Output summary JSON, relative to project-dir unless absolute.",
    )
    parser.add_argument("--base-table", default=None, help="Override base sample table.")
    parser.add_argument("--output-table", default=None, help="Override target wide table.")
    parser.add_argument("--base-where", default=None, help="Optional where clause for base table subquery.")
    parser.add_argument("--feature-where", default=None, help="Optional where clause for every feature table subquery.")
    return parser.parse_args()


def resolve_project_path(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def main() -> int:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    sql_path, feature_map_path, summary_path = generate_wide_sql(
        project_dir=project_dir,
        remain_features_path=resolve_project_path(project_dir, args.remain_features),
        sql_output_path=resolve_project_path(project_dir, args.sql_output),
        feature_map_path=resolve_project_path(project_dir, args.feature_map_output),
        summary_path=resolve_project_path(project_dir, args.summary_output),
        base_table=args.base_table,
        output_table=args.output_table,
        base_where=args.base_where,
        feature_where=args.feature_where,
    )
    print(f"sql: {sql_path}")
    print(f"feature_map: {feature_map_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
