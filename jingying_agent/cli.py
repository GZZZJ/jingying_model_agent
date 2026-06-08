"""Command-line interface for local agent operations."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from jingying_agent.manifest import write_manifest
from jingying_agent.feature_screening import write_feature_screening_summary
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
        project_dir / "configs" / "refine_features.yaml",
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


def cmd_export_feature_metadata(args: argparse.Namespace) -> int:
    from jingying_agent.feature_metadata import main as metadata_main

    argv = ["--project-dir", str((REPO_ROOT / args.project).resolve() if not Path(args.project).is_absolute() else Path(args.project))]
    if args.tables_file:
        argv.extend(["--tables-file", args.tables_file])
    return metadata_main(argv)


def cmd_run_d01_d02(args: argparse.Namespace) -> int:
    from jingying_agent.batch_feature_select import main as batch_select_main

    argv = ["--project-dir", str((REPO_ROOT / args.project).resolve() if not Path(args.project).is_absolute() else Path(args.project))]
    if args.config:
        argv.extend(["--config", args.config])
    if args.max_tables is not None:
        argv.extend(["--max-tables", str(args.max_tables)])
    if args.table:
        for table in args.table:
            argv.extend(["--table", table])
    if args.dry_run_sql:
        argv.append("--dry-run-sql")
    if args.refresh_dp_cache:
        argv.append("--refresh-dp-cache")
    if args.sql_approved:
        argv.append("--sql-approved")
    if args.force:
        argv.append("--force")
    return batch_select_main(argv)


def cmd_refine_features(args: argparse.Namespace) -> int:
    from jingying_agent.feature_refine import main as refine_main

    argv = ["--project-dir", str((REPO_ROOT / args.project).resolve() if not Path(args.project).is_absolute() else Path(args.project))]
    if args.config:
        argv.extend(["--config", args.config])
    if args.dry_run_sql:
        argv.append("--dry-run-sql")
    if args.refresh_dp_cache:
        argv.append("--refresh-dp-cache")
    if args.sql_approved:
        argv.append("--sql-approved")
    return refine_main(argv)


def cmd_feature_screening_summary(args: argparse.Namespace) -> int:
    project_dir = (REPO_ROOT / args.project).resolve() if not Path(args.project).is_absolute() else Path(args.project)
    output_path = write_feature_screening_summary(project_dir, args.output)
    print(f"summary: {output_path}")
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

    export_metadata = subparsers.add_parser(
        "export-feature-metadata",
        help="export source and feature table metadata for a model project",
    )
    export_metadata.add_argument("--project", required=True, help="project path, absolute or relative to repo root")
    export_metadata.add_argument("--tables-file", default=None, help="override feature table list path")
    export_metadata.set_defaults(func=cmd_export_feature_metadata)

    run_d01_d02 = subparsers.add_parser(
        "run-d01-d02",
        help="run per-table D01/D02 screening with DP feather caching",
    )
    run_d01_d02.add_argument("--project", required=True, help="project path, absolute or relative to repo root")
    run_d01_d02.add_argument("--config", default=None, help="feature select config path")
    run_d01_d02.add_argument("--table", action="append", help="only run the specified full table name")
    run_d01_d02.add_argument("--max-tables", type=int, default=None, help="optional cap for smoke runs")
    run_d01_d02.add_argument("--dry-run-sql", action="store_true", help="write and print SQL metadata only")
    run_d01_d02.add_argument("--refresh-dp-cache", action="store_true", help="refresh local feather cache after approval")
    run_d01_d02.add_argument("--sql-approved", action="store_true", help="confirm displayed SQL has been reviewed")
    run_d01_d02.add_argument("--force", action="store_true", help="overwrite table checkpoints")
    run_d01_d02.set_defaults(func=cmd_run_d01_d02)

    refine_features = subparsers.add_parser(
        "refine-features",
        help="refine wide-table features with correlation and importance filters",
    )
    refine_features.add_argument("--project", required=True, help="project path, absolute or relative to repo root")
    refine_features.add_argument("--config", default=None, help="refine config path")
    refine_features.add_argument("--dry-run-sql", action="store_true", help="write and print SQL metadata only")
    refine_features.add_argument("--refresh-dp-cache", action="store_true", help="refresh local feather cache after approval")
    refine_features.add_argument("--sql-approved", action="store_true", help="confirm displayed SQL has been reviewed")
    refine_features.set_defaults(func=cmd_refine_features)

    screening_summary = subparsers.add_parser(
        "feature-screening-summary",
        help="write a source-backed feature screening process summary",
    )
    screening_summary.add_argument("--project", required=True, help="project path, absolute or relative to repo root")
    screening_summary.add_argument(
        "--output",
        default="reports/feature_screening_process.json",
        help="summary JSON output path, relative to project unless absolute",
    )
    screening_summary.set_defaults(func=cmd_feature_screening_summary)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
