"""Command-line interface for local agent operations."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from jingying_agent.manifest import write_manifest
from jingying_agent.project import REPO_ROOT, create_project
from jingying_agent.wide_sql import generate_wide_sql


def cmd_doctor(_: argparse.Namespace) -> int:
    """Check expected local files and optional dependencies."""
    checks = {
        "planning_doc": REPO_ROOT / "doc" / "AI经营建模Agent规划.md",
        "model_inventory": REPO_ROOT / "doc" / "现有经营模型梳理.md",
        "gcard_workbook": REPO_ROOT / "doc" / "复借G卡模型文档.xlsx",
        "feature_select_v2_code": REPO_ROOT / "vendor" / "feature-select-v2" / "scripts" / "code" / "main.py",
        "project_template": REPO_ROOT / "templates" / "project" / "project.yaml",
    }

    ok = True
    for name, path in checks.items():
        exists = path.exists()
        ok = ok and exists
        status = "OK" if exists else "MISSING"
        print(f"{status:7} {name}: {path}")

    yaml_available = importlib.util.find_spec("yaml") is not None
    ok = ok and yaml_available
    print(f"{'OK' if yaml_available else 'MISSING':7} dependency: PyYAML")
    return 0 if ok else 1


def cmd_init_project(args: argparse.Namespace) -> int:
    project_dir = create_project(
        REPO_ROOT,
        name=args.name,
        display_name=args.display_name,
        scenario=args.scenario,
        template=args.template,
        force=args.force,
    )
    print(f"created: {project_dir}")
    return 0


def cmd_new_run(args: argparse.Namespace) -> int:
    project_dir = (REPO_ROOT / args.project).resolve() if not Path(args.project).is_absolute() else Path(args.project)
    inputs = [
        project_dir / "project.yaml",
        project_dir / "configs" / "sample.yaml",
        project_dir / "configs" / "feature_select.yaml",
        project_dir / "configs" / "train.yaml",
        project_dir / "configs" / "evaluate.yaml",
        project_dir / "configs" / "report.yaml",
    ]
    manifest_path = write_manifest(project_dir, args.step, inputs=inputs, extra={"note": args.note or ""})
    print(f"manifest: {manifest_path}")
    return 0


def resolve_project_path(project_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def cmd_build_wide_sql(args: argparse.Namespace) -> int:
    project_dir = (REPO_ROOT / args.project).resolve() if not Path(args.project).is_absolute() else Path(args.project)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="经营建模 Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check local scaffold and dependencies")
    doctor.set_defaults(func=cmd_doctor)

    init_project = subparsers.add_parser("init-project", help="create a model project workspace")
    init_project.add_argument("--name", required=True, help="project folder name under projects/")
    init_project.add_argument("--display-name", required=True, help="human-readable model name")
    init_project.add_argument("--scenario", required=True, help="business scenario")
    init_project.add_argument("--template", default="generic", choices=["generic", "fujie-gcard"])
    init_project.add_argument("--force", action="store_true", help="overwrite existing files from template")
    init_project.set_defaults(func=cmd_init_project)

    new_run = subparsers.add_parser("new-run", help="create a run manifest skeleton")
    new_run.add_argument("--project", required=True, help="project path, absolute or relative to repo root")
    new_run.add_argument("--step", required=True, help="logical step name")
    new_run.add_argument("--note", default="", help="optional note recorded in manifest")
    new_run.set_defaults(func=cmd_new_run)

    build_wide_sql = subparsers.add_parser("build-wide-sql", help="generate MaxCompute SQL for a wide feature table")
    build_wide_sql.add_argument("--project", required=True, help="project path, absolute or relative to repo root")
    build_wide_sql.add_argument(
        "--remain-features",
        default="runs/d01_d02_batch_select/results/d01_d02_final_remain_features.json",
        help="remaining feature JSON, relative to project unless absolute",
    )
    build_wide_sql.add_argument(
        "--sql-output",
        default="queries/06_build_d01_d02_wide_table.sql",
        help="SQL output path, relative to project unless absolute",
    )
    build_wide_sql.add_argument(
        "--feature-map-output",
        default="runs/d01_d02_batch_select/results/d01_d02_wide_feature_map.csv",
        help="feature mapping CSV output, relative to project unless absolute",
    )
    build_wide_sql.add_argument(
        "--summary-output",
        default="runs/d01_d02_batch_select/results/d01_d02_wide_sql_summary.json",
        help="summary JSON output, relative to project unless absolute",
    )
    build_wide_sql.add_argument("--base-table", default=None, help="override base sample table")
    build_wide_sql.add_argument("--output-table", default=None, help="override target wide table")
    build_wide_sql.add_argument("--base-where", default=None, help="optional base table where clause")
    build_wide_sql.add_argument("--feature-where", default=None, help="optional feature table where clause")
    build_wide_sql.set_defaults(func=cmd_build_wide_sql)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
