"""Command-line interface for the local business modeling workbench."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from jingying_model_agent.config import load_yaml
from jingying_model_agent.feature_screening import write_feature_screening_summary
from jingying_model_agent.manifest import make_run_id
from jingying_model_agent.paths import REPO_ROOT, project_config_path, resolve_project_path, workflow_path
from jingying_model_agent.planning import create_execution_plan, save_execution_plan
from jingying_model_agent.project import create_project
from jingying_model_agent.request import parse_model_request, validate_model_request
from jingying_model_agent.state import (
    append_decision,
    create_run_state,
    load_run_state,
    mark_stage_done,
    mark_stage_started,
    register_artifact,
    run_dir,
    save_run_state,
)
from jingying_model_agent.wide_sql import generate_wide_sql


def _run_path(args: argparse.Namespace) -> Path:
    return run_dir(resolve_project_path(args.project), args.run_id)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _copy_if_exists(source: Path, target: Path) -> Path | None:
    if not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _load_project_config(project_dir: Path) -> dict[str, Any]:
    return load_yaml(project_config_path(project_dir))


def cmd_doctor(_: argparse.Namespace) -> int:
    """Check expected local files and optional dependencies."""
    checks = {
        "planning_doc": REPO_ROOT / "docs" / "legacy" / "AI经营建模Agent规划.md",
        "model_inventory": REPO_ROOT / "docs" / "legacy" / "现有经营模型梳理.md",
        "gcard_workbook": REPO_ROOT / "docs" / "legacy" / "复借G卡模型文档.xlsx",
        "feature_select_v2_code": REPO_ROOT / "vendor" / "feature-select-v2" / "scripts" / "code" / "main.py",
        "project_template": REPO_ROOT / "templates" / "project" / "project.yml",
        "workflow_full_modeling": REPO_ROOT / "workflows" / "full_modeling.yml",
    }

    ok = True
    for name, path in checks.items():
        exists = path.exists()
        ok = ok and exists
        print(f"{'OK' if exists else 'MISSING':7} {name}: {path}")

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


def cmd_project_validate(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    config_path = project_config_path(project_dir)
    errors: list[str] = []
    if not config_path.exists():
        errors.append(f"missing project config: {config_path}")
    else:
        config = load_yaml(config_path)
        for key in ["project", "data", "segments"]:
            if key not in config:
                errors.append(f"missing top-level key: {key}")
        data = config.get("data", {})
        for key in ["source_table", "id_columns", "target_column", "time_column", "period_column"]:
            if not data.get(key):
                errors.append(f"missing data.{key}")
    for directory in ["configs", "queries", "runs", "reports"]:
        if not (project_dir / directory).exists():
            errors.append(f"missing directory: {directory}")

    if errors:
        print("project validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"project validation ok: {project_dir}")
    return 0


def cmd_workflow_show(args: argparse.Namespace) -> int:
    path = workflow_path(args.workflow)
    print(path.read_text(encoding="utf-8"))
    return 0


def cmd_workflow_validate(args: argparse.Namespace) -> int:
    path = workflow_path(args.workflow)
    if not path.exists():
        print(f"missing workflow: {path}")
        return 1
    workflow = load_yaml(path)
    stages = workflow.get("stages")
    if not workflow.get("name") or not isinstance(stages, list) or not stages:
        print(f"workflow validation failed: {path}")
        return 1
    print(f"workflow validation ok: {path}")
    return 0


def cmd_workflow_list(_: argparse.Namespace) -> int:
    for path in sorted((REPO_ROOT / "workflows").glob("*.yml")):
        print(path.stem)
    return 0


def cmd_run_init(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    workflow_file = workflow_path(args.workflow)
    workflow = load_yaml(workflow_file)
    run_id = args.run_id or make_run_id()
    path = run_dir(project_dir, run_id)
    if path.exists() and not args.force:
        print(f"run already exists: {path}")
        return 1

    for directory in ["configs_snapshot", "audit", "tasks", "sample_check", "feature_selection", "modeling", "evaluation", "reports"]:
        (path / directory).mkdir(parents=True, exist_ok=True)
    for config_file in [project_config_path(project_dir), *sorted((project_dir / "configs").glob("*.y*ml"))]:
        if config_file.exists():
            shutil.copy2(config_file, path / "configs_snapshot" / config_file.name)

    state = create_run_state(project_dir, run_id=run_id, workflow=workflow.get("name", args.workflow), stages=workflow.get("stages"))
    save_run_state(path, state)
    _write_json(path / "audit" / "artifact_manifest.json", {"version": 1, "artifacts": []})
    _write_text(path / "audit" / "command_log.jsonl", "")
    _write_text(path / "audit" / "decision_log.md", f"# Decision Log\n\n- imported: false\n")
    register_artifact(path, "validate_config", "configs_snapshot", kind="directory", description="Project config snapshot")
    if getattr(args, "request", None):
        request_path = Path(args.request)
        request_path = request_path if request_path.is_absolute() else (REPO_ROOT / request_path)
        if request_path.exists():
            shutil.copy2(request_path, path / "model_request.md")
            register_artifact(path, "validate_config", "model_request.md", description="Model request copied into run workspace")
    if getattr(args, "plan", None):
        plan_path = Path(args.plan)
        plan_path = plan_path if plan_path.is_absolute() else (REPO_ROOT / plan_path)
        if plan_path.exists():
            shutil.copy2(plan_path, path / "execution_plan.yml")
            register_artifact(path, "validate_config", "execution_plan.yml", description="Execution plan copied into run workspace")
    print(f"run_id: {run_id}")
    print(f"run_dir: {path}")
    return 0


def cmd_request_validate(args: argparse.Namespace) -> int:
    request_path = Path(args.request)
    request_path = request_path if request_path.is_absolute() else (REPO_ROOT / request_path)
    project_dir = resolve_project_path(args.project) if args.project else None
    try:
        request_doc = parse_model_request(request_path)
        result = validate_model_request(request_doc, project_dir)
    except Exception as exc:
        print(f"request validation failed: {exc}")
        return 1

    if result["errors"]:
        print("request validation failed:")
        for error in result["errors"]:
            print(f"- {error}")
    else:
        print(f"request validation ok: {request_path}")
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    return 0 if not result["errors"] else 1


def cmd_plan_create(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    request_path = Path(args.request)
    request_path = request_path if request_path.is_absolute() else (REPO_ROOT / request_path)
    request_doc = parse_model_request(request_path)
    validation = validate_model_request(request_doc, project_dir)
    if validation["errors"]:
        print("cannot create plan; request validation failed:")
        for error in validation["errors"]:
            print(f"- {error}")
        return 1

    plan = create_execution_plan(request_doc, args.project)
    output = Path(args.output) if args.output else project_dir / "requests" / f"{request_doc['metadata']['request_id']}.execution_plan.yml"
    output = output if output.is_absolute() else (REPO_ROOT / output)
    output_path = save_execution_plan(plan, output)
    print(f"execution_plan: {output_path}")
    print(f"task_count: {len(plan['tasks'])}")
    for warning in validation["warnings"]:
        print(f"warning: {warning}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    path = _run_path(args)
    state = load_run_state(path)
    print(yaml.safe_dump(state, allow_unicode=True, sort_keys=False))
    return 0


def cmd_sample_check(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    path = _run_path(args)
    mark_stage_started(path, "sample_check")
    config = _load_project_config(project_dir)
    data_cfg = config.get("data", {})
    raw_path = project_dir / data_cfg.get("raw_path", "data/raw/sample.feather")
    status = "scaffold"
    reason = "local data not available"
    if raw_path.exists():
        reason = "local sample file exists; real profiling is not implemented in phase 1"
    summary = {
        "status": status,
        "reason": reason,
        "project": config.get("project", {}),
        "target_column": data_cfg.get("target_column"),
        "id_columns": data_cfg.get("id_columns", []),
        "split_column": data_cfg.get("split_column") or config.get("split", {}).get("source_column"),
        "expected_outputs": [
            "positive_rate_overall.csv",
            "positive_rate_by_split.csv",
            "positive_rate_by_month.csv",
            "positive_rate_by_segment.csv",
        ],
    }
    _write_json(path / "sample_check" / "sample_summary.json", summary)
    _write_text(
        path / "sample_check" / "sample_check_report.md",
        "# Sample Check\n\nstatus: scaffold\n\nreason: local data not available\n",
    )
    register_artifact(path, "sample_check", "sample_check/sample_summary.json")
    register_artifact(path, "sample_check", "sample_check/sample_check_report.md")
    append_decision(path, stage="sample_check", decision="scaffold", reason=reason)
    mark_stage_done(path, "sample_check", scaffold=True)
    print(f"sample_check: {path / 'sample_check' / 'sample_summary.json'}")
    return 0


def cmd_feature_metadata(args: argparse.Namespace) -> int:
    path = _run_path(args)
    mark_stage_started(path, "feature_metadata")
    from jingying_model_agent.feature_metadata import main as metadata_main

    argv = ["--project-dir", str(resolve_project_path(args.project))]
    if args.tables_file:
        argv.extend(["--tables-file", args.tables_file])
    code = metadata_main(argv)
    if code == 0:
        mark_stage_done(path, "feature_metadata")
    return code


def cmd_feature_d01_d02(args: argparse.Namespace) -> int:
    path = _run_path(args)
    mark_stage_started(path, "d01_d02_screening")
    from jingying_model_agent.batch_feature_select import main as batch_select_main

    argv = ["--project-dir", str(resolve_project_path(args.project))]
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
    code = batch_select_main(argv)
    if code == 0:
        mark_stage_done(path, "d01_d02_screening", scaffold=args.dry_run_sql)
    return code


def cmd_build_wide_sql(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    sql_path, feature_map_path, summary_path = generate_wide_sql(
        project_dir=project_dir,
        remain_features_path=resolve_project_path(project_dir / args.remain_features) if not Path(args.remain_features).is_absolute() else Path(args.remain_features),
        sql_output_path=project_dir / args.sql_output if not Path(args.sql_output).is_absolute() else Path(args.sql_output),
        feature_map_path=project_dir / args.feature_map_output if not Path(args.feature_map_output).is_absolute() else Path(args.feature_map_output),
        summary_path=project_dir / args.summary_output if not Path(args.summary_output).is_absolute() else Path(args.summary_output),
        base_table=args.base_table,
        output_table=args.output_table,
        base_where=args.base_where,
        feature_where=args.feature_where,
    )
    print(f"sql: {sql_path}")
    print(f"feature_map: {feature_map_path}")
    print(f"summary: {summary_path}")
    return 0


def cmd_feature_refine(args: argparse.Namespace) -> int:
    path = _run_path(args)
    mark_stage_started(path, "feature_refine")
    from jingying_model_agent.feature_refine import main as refine_main

    argv = ["--project-dir", str(resolve_project_path(args.project))]
    if args.config:
        argv.extend(["--config", args.config])
    if args.dry_run_sql:
        argv.append("--dry-run-sql")
    if args.refresh_dp_cache:
        argv.append("--refresh-dp-cache")
    if args.sql_approved:
        argv.append("--sql-approved")
    code = refine_main(argv)
    if code == 0:
        mark_stage_done(path, "feature_refine", scaffold=args.dry_run_sql)
    return code


def cmd_train(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    path = _run_path(args)
    mark_stage_started(path, "train_baseline")
    config_path = Path(args.config) if getattr(args, "config", None) else project_dir / "configs" / "train.yaml"
    config_path = config_path if config_path.is_absolute() else project_dir / config_path
    train_config = load_yaml(config_path) if config_path.exists() else {}
    configured_input = train_config.get("input", {}).get("feather_path")
    input_feather = Path(args.input_feather or configured_input or "")
    if input_feather and not input_feather.is_absolute():
        input_feather = project_dir / input_feather
    feature_list = Path(args.feature_list or train_config.get("training", {}).get("feature_list_path", "runs/modeling_feature_set/feature_list.txt"))
    if not feature_list.is_absolute():
        feature_list = project_dir / feature_list
    output_dir = path / "modeling" / args.experiment
    score_output = Path(args.score_output or path / "modeling" / args.experiment / "scores_all_splits.feather")
    if not score_output.is_absolute():
        score_output = project_dir / score_output
    input_snapshot_dir = Path(args.input_dir or path / "modeling_input")
    if not input_snapshot_dir.is_absolute():
        input_snapshot_dir = project_dir / input_snapshot_dir

    if input_feather and Path(input_feather).exists() and feature_list.exists() and train_config:
        try:
            from jingying_model_agent.modeling.train_lgb import train_lightgbm_from_feather

            metrics = train_lightgbm_from_feather(
                input_feather=input_feather,
                feature_list_path=feature_list,
                output_dir=output_dir,
                score_output=score_output,
                input_snapshot_dir=input_snapshot_dir,
                config=train_config,
            )
            _write_json(output_dir / "train_metrics.json", {"status": "done", "metrics": metrics, "experiment": args.experiment})
            for artifact in [
                "train_metrics.json",
                "metrics_train_valid.json",
                "feature_importance.csv",
                "feature_drop_detail.csv",
                "actual_feature_list.txt",
                "preprocessing.json",
                "run_config.json",
                "model.pkl",
            ]:
                register_artifact(path, "train_baseline", f"modeling/{args.experiment}/{artifact}")
            append_decision(path, stage="train_baseline", decision="done", reason="LightGBM training completed from local feather data")
            mark_stage_done(path, "train_baseline")
            print(f"train complete: {output_dir}")
            return 0
        except Exception as exc:
            payload = {"status": "scaffold", "reason": f"training failed or dependency missing: {exc}", "experiment": args.experiment}
    else:
        payload = {
            "status": "scaffold",
            "reason": "training data not available",
            "experiment": args.experiment,
            "input_feather": str(input_feather) if input_feather else "",
            "input_feather_exists": bool(input_feather and Path(input_feather).exists()),
            "feature_list": str(feature_list),
            "feature_list_exists": feature_list.exists(),
            "train_config_keys": sorted(train_config.keys()),
        }
    _write_json(output_dir / "train_metrics.json", payload)
    if feature_list.exists():
        _copy_if_exists(feature_list, output_dir / "feature_list.txt")
    register_artifact(path, "train_baseline", f"modeling/{args.experiment}/train_metrics.json")
    append_decision(path, stage="train_baseline", decision="scaffold", reason=payload["reason"])
    mark_stage_done(path, "train_baseline", scaffold=True)
    print(f"train scaffold: {output_dir / 'train_metrics.json'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    path = _run_path(args)
    mark_stage_started(path, "evaluate")
    evaluate_config = load_yaml(project_dir / "configs" / "evaluate.yaml") if (project_dir / "configs" / "evaluate.yaml").exists() else {}
    metrics = evaluate_config.get("metrics") or evaluate_config.get("evaluation", {}).get("metrics") or []
    scores_feather = Path(args.scores_feather or path / "modeling" / "scores_all_splits.feather")
    if not scores_feather.is_absolute():
        scores_feather = project_dir / scores_feather
    output_dir = Path(args.output_dir or path / "evaluation")
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    if scores_feather.exists() and evaluate_config:
        try:
            from jingying_model_agent.evaluation.run import evaluate_scores_from_feather

            summary = evaluate_scores_from_feather(scores_feather=scores_feather, output_dir=output_dir, config=evaluate_config)
            for artifact in [
                "evaluation_summary.json",
                "overall_metrics.csv",
                "monthly_metrics.csv",
                "segment_metrics.csv",
                "benchmark_uplift.csv",
                "score_psi_by_month.csv",
            ]:
                if (output_dir / artifact).exists():
                    register_artifact(path, "evaluate", output_dir / artifact)
            append_decision(path, stage="evaluate", decision="done", reason="Evaluation completed from local score feather")
            mark_stage_done(path, "evaluate")
            print(f"evaluation complete: {output_dir / 'evaluation_summary.json'}")
            return 0
        except Exception as exc:
            payload = {"status": "scaffold", "reason": f"evaluation failed or dependency missing: {exc}", "configured_metrics": metrics}
    else:
        payload = {
            "status": "scaffold",
            "reason": "prediction data not available",
            "configured_metrics": metrics,
            "scores_feather": str(scores_feather),
            "scores_feather_exists": scores_feather.exists(),
        }
    _write_json(path / "evaluation" / "evaluation_summary.json", payload)
    register_artifact(path, "evaluate", "evaluation/evaluation_summary.json")
    append_decision(path, stage="evaluate", decision="scaffold", reason=payload["reason"])
    mark_stage_done(path, "evaluate", scaffold=True)
    print(f"evaluation scaffold: {path / 'evaluation' / 'evaluation_summary.json'}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    path = _run_path(args)
    mark_stage_started(path, "compare")
    payload = {"status": "scaffold", "reason": "candidate and champion predictions not available", "champion": args.champion}
    _write_json(path / "evaluation" / "champion_challenger.json", payload)
    register_artifact(path, "compare", "evaluation/champion_challenger.json")
    append_decision(path, stage="compare", decision="scaffold", reason=payload["reason"])
    mark_stage_done(path, "compare", scaffold=True)
    print(f"compare scaffold: {path / 'evaluation' / 'champion_challenger.json'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    path = _run_path(args)
    mark_stage_started(path, "report")
    report_config = load_yaml(project_dir / "configs" / "report.yaml") if (project_dir / "configs" / "report.yaml").exists() else {}
    sections = report_config.get("sections") or report_config.get("report", {}).get("sections") or []
    state = load_run_state(path)
    manifest_path = path / "audit" / "artifact_manifest.json"
    text = (
        "# Model Report\n\n"
        "status: scaffold\n\n"
        "This report is generated only from registered run artifacts. Missing metrics are not fabricated.\n\n"
        f"- run_id: {args.run_id}\n"
        f"- workflow: {state.get('workflow')}\n"
        f"- configured_sections: {', '.join(sections) if sections else 'not configured'}\n"
        f"- artifact_manifest: {manifest_path.relative_to(path)}\n"
    )
    _write_text(path / "reports" / "model_report.md", text)
    _write_text(path / "reports" / "model_card.md", "# Model Card\n\nstatus: scaffold\n")
    _write_text(path / "reports" / "executive_summary.md", "# Executive Summary\n\nstatus: scaffold\n")
    for artifact in ["reports/model_report.md", "reports/model_card.md", "reports/executive_summary.md"]:
        register_artifact(path, "report", artifact)
    train_dirs = [item for item in (path / "modeling").glob("*") if item.is_dir() and (item / "metrics_train_valid.json").exists()]
    excel_path = None
    if (path / "evaluation" / "evaluation_summary.json").exists() and train_dirs:
        try:
            from jingying_model_agent.reporting.excel_report import generate_excel_report

            excel_path = generate_excel_report(
                eval_dir=path / "evaluation",
                train_dir=train_dirs[0],
                input_dir=path / "modeling_input",
                feature_dir=path / "feature_selection",
                output_path=path / "reports" / "model_report.xlsx",
            )
            register_artifact(path, "report", excel_path)
            for sidecar in [
                excel_path.with_name("model_report.md"),
                excel_path.with_name("model_report.html"),
                excel_path.with_name("model_report_missing_results.md"),
            ]:
                if sidecar.exists():
                    register_artifact(path, "report", sidecar)
        except Exception as exc:
            append_decision(path, stage="report", decision="excel_scaffold", reason=f"Excel report not generated: {exc}")
    if excel_path:
        append_decision(path, stage="report", decision="done", reason="Excel report generated from standard train and evaluation artifacts")
        mark_stage_done(path, "report")
        print(f"report complete: {excel_path}")
    else:
        append_decision(path, stage="report", decision="scaffold", reason="report generated with missing real evaluation artifacts")
        mark_stage_done(path, "report", scaffold=True)
        print(f"report scaffold: {path / 'reports' / 'model_report.md'}")
    return 0


def cmd_feature_screening_summary(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    output_path = write_feature_screening_summary(project_dir, args.output)
    print(f"summary: {output_path}")
    return 0


def cmd_import_gcard(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    run_id = args.run_id
    path = run_dir(project_dir, run_id)
    for directory in ["audit", "feature_selection"]:
        (path / directory).mkdir(parents=True, exist_ok=True)
    state = create_run_state(project_dir, run_id=run_id, workflow="imported_feature_screening", status="imported")
    save_run_state(path, state)
    _write_json(path / "audit" / "artifact_manifest.json", {"version": 1, "artifacts": []})
    _write_text(path / "audit" / "command_log.jsonl", "")
    _write_text(path / "audit" / "decision_log.md", "# Decision Log\n")

    copies = [
        (project_dir / "reports" / "feature_screening_process.json", path / "feature_selection" / "feature_screening_process.json"),
        (project_dir / "runs" / "feature_refine_feather" / "final_500_features.txt", path / "feature_selection" / "final_features.txt"),
    ]
    for source, target in copies:
        copied = _copy_if_exists(source, target)
        if copied:
            register_artifact(path, "feature_refine", copied.relative_to(path), source="imported")

    for source in [
        project_dir / "reports" / "feature_screening_process.xlsx",
        project_dir / "runs" / "feature_refine_feather",
        project_dir / "runs" / "modeling_feature_set",
    ]:
        register_artifact(path, "feature_refine", source, kind="directory" if source.is_dir() else "file", source="imported")
    append_decision(path, stage="feature_refine", decision="imported", reason="本 run 是从历史产物导入，不是重新执行得到。")
    mark_stage_done(path, "feature_refine")
    print(f"imported run: {path}")
    return 0


def cmd_import_gcard_model(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    run_id = args.run_id
    path = run_dir(project_dir, run_id)
    for directory in ["audit", "configs_snapshot", "sample_check", "feature_selection", "modeling/main_lgbm", "modeling_input", "evaluation", "reports"]:
        (path / directory).mkdir(parents=True, exist_ok=True)
    state = create_run_state(project_dir, run_id=run_id, workflow="imported_gcard_main_lgbm", status="imported")
    save_run_state(path, state)
    _write_json(path / "audit" / "artifact_manifest.json", {"version": 1, "artifacts": []})
    _write_text(path / "audit" / "command_log.jsonl", "")
    _write_text(path / "audit" / "decision_log.md", "# Decision Log\n")

    copy_groups = [
        (project_dir / "configs" / "train.yaml", path / "configs_snapshot" / "train.yaml", "validate_config"),
        (project_dir / "runs" / "modeling_input" / "sample_split_summary.csv", path / "sample_check" / "sample_split_summary.csv", "sample_check"),
        (project_dir / "runs" / "modeling_input" / "label_distribution.csv", path / "sample_check" / "label_distribution.csv", "sample_check"),
        (project_dir / "runs" / "modeling_input" / "segment_distribution.csv", path / "sample_check" / "segment_distribution.csv", "sample_check"),
        (project_dir / "runs" / "modeling_feature_set" / "feature_list.txt", path / "feature_selection" / "final_features.txt", "feature_refine"),
        (project_dir / "runs" / "modeling_feature_set" / "feature_availability.csv", path / "feature_selection" / "feature_availability.csv", "feature_refine"),
        (project_dir / "runs" / "modeling_feature_set" / "feature_stage_summary.json", path / "feature_selection" / "feature_stage_summary.json", "feature_refine"),
        (project_dir / "runs" / "modeling_input" / "input_config.json", path / "modeling_input" / "input_config.json", "train_baseline"),
        (project_dir / "runs" / "modeling_input" / "input_schema.csv", path / "modeling_input" / "input_schema.csv", "train_baseline"),
        (project_dir / "runs" / "model_train" / "main_lgbm" / "model.pkl", path / "modeling" / "main_lgbm" / "model.pkl", "train_baseline"),
        (project_dir / "runs" / "model_train" / "main_lgbm" / "metrics_train_valid.json", path / "modeling" / "main_lgbm" / "metrics_train_valid.json", "train_baseline"),
        (project_dir / "runs" / "model_train" / "main_lgbm" / "feature_importance.csv", path / "modeling" / "main_lgbm" / "feature_importance.csv", "train_baseline"),
        (project_dir / "runs" / "model_train" / "main_lgbm" / "feature_drop_detail.csv", path / "modeling" / "main_lgbm" / "feature_drop_detail.csv", "train_baseline"),
        (project_dir / "runs" / "model_train" / "main_lgbm" / "actual_feature_list.txt", path / "modeling" / "main_lgbm" / "actual_feature_list.txt", "train_baseline"),
        (project_dir / "runs" / "model_train" / "main_lgbm" / "preprocessing.json", path / "modeling" / "main_lgbm" / "preprocessing.json", "train_baseline"),
        (project_dir / "runs" / "model_train" / "main_lgbm" / "run_config.json", path / "modeling" / "main_lgbm" / "run_config.json", "train_baseline"),
        (project_dir / "runs" / "model_eval" / "evaluation_summary.json", path / "evaluation" / "evaluation_summary.json", "evaluate"),
        (project_dir / "runs" / "model_eval" / "overall_metrics.csv", path / "evaluation" / "overall_metrics.csv", "evaluate"),
        (project_dir / "runs" / "model_eval" / "monthly_metrics.csv", path / "evaluation" / "monthly_metrics.csv", "evaluate"),
        (project_dir / "runs" / "model_eval" / "segment_metrics.csv", path / "evaluation" / "segment_metrics.csv", "evaluate"),
        (project_dir / "runs" / "model_eval" / "benchmark_uplift.csv", path / "evaluation" / "benchmark_uplift.csv", "compare"),
        (project_dir / "reports" / "model_report.xlsx", path / "reports" / "model_report.xlsx", "report"),
    ]
    for source, target, stage in copy_groups:
        copied = _copy_if_exists(source, target)
        if copied:
            register_artifact(path, stage, copied.relative_to(path), source="imported")

    eval_dir = project_dir / "runs" / "model_eval"
    for source in sorted(eval_dir.glob("decile_lift*.csv")) + sorted(eval_dir.glob("intent_zc*.csv")) + sorted(eval_dir.glob("score_psi_by_month.csv")):
        target = path / "evaluation" / source.name
        copied = _copy_if_exists(source, target)
        if copied:
            register_artifact(path, "evaluate", copied.relative_to(path), source="imported")

    for stage in ["validate_config", "sample_check", "feature_refine", "train_baseline", "evaluate", "compare", "report"]:
        mark_stage_done(path, stage)
    append_decision(path, stage="train_baseline", decision="imported", reason="本 run 从远端真实复借 G 卡 main_lgbm 训练、评估和报告产物导入，不是当前环境重新执行。")
    print(f"imported model run: {path}")
    return 0


def _add_project_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    project = subparsers.add_parser("project", help="project workspace commands")
    project_sub = project.add_subparsers(dest="project_command", required=True)
    validate = project_sub.add_parser("validate", help="validate project config")
    validate.add_argument("--project", required=True)
    validate.set_defaults(func=cmd_project_validate)


def _add_workflow_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    workflow = subparsers.add_parser("workflow", help="workflow commands")
    workflow_sub = workflow.add_subparsers(dest="workflow_command", required=True)
    workflow_sub.add_parser("list", help="list workflows").set_defaults(func=cmd_workflow_list)
    show = workflow_sub.add_parser("show", help="show workflow YAML")
    show.add_argument("--workflow", required=True)
    show.set_defaults(func=cmd_workflow_show)
    validate = workflow_sub.add_parser("validate", help="validate workflow YAML")
    validate.add_argument("--workflow", required=True)
    validate.set_defaults(func=cmd_workflow_validate)


def _add_run_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run = subparsers.add_parser("run", help="run lifecycle commands")
    run_sub = run.add_subparsers(dest="run_command", required=True)
    init = run_sub.add_parser("init", help="initialize a workflow run")
    init.add_argument("--project", required=True)
    init.add_argument("--workflow", required=True)
    init.add_argument("--run-id", default=None)
    init.add_argument("--request", default=None, help="optional model request Markdown copied into the run")
    init.add_argument("--plan", default=None, help="optional execution plan YAML copied into the run")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_run_init)
    imported = run_sub.add_parser("import-gcard-artifacts", help="import existing Fujie GCard artifacts")
    imported.add_argument("--project", default="projects/2026-05-fujie-gcard-v1")
    imported.add_argument("--run-id", default="2026-05-imported-feature-screening")
    imported.set_defaults(func=cmd_import_gcard)
    imported_model = run_sub.add_parser("import-gcard-model-artifacts", help="import existing Fujie GCard training/evaluation/report artifacts")
    imported_model.add_argument("--project", default="projects/2026-05-fujie-gcard-v1")
    imported_model.add_argument("--run-id", default="2026-06-imported-gcard-main-lgbm")
    imported_model.set_defaults(func=cmd_import_gcard_model)
    status = run_sub.add_parser("status", help="show run state")
    status.add_argument("--project", required=True)
    status.add_argument("--run-id", required=True)
    status.set_defaults(func=cmd_status)


def _add_request_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    request = subparsers.add_parser("request", help="model request commands")
    request_sub = request.add_subparsers(dest="request_command", required=True)
    validate = request_sub.add_parser("validate", help="validate a model request Markdown file")
    validate.add_argument("--request", required=True)
    validate.add_argument("--project", default=None)
    validate.set_defaults(func=cmd_request_validate)


def _add_plan_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    plan = subparsers.add_parser("plan", help="execution plan commands")
    plan_sub = plan.add_subparsers(dest="plan_command", required=True)
    create = plan_sub.add_parser("create", help="create an execution plan from a model request")
    create.add_argument("--project", required=True)
    create.add_argument("--request", required=True)
    create.add_argument("--output", default=None)
    create.set_defaults(func=cmd_plan_create)


def _add_feature_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    feature = subparsers.add_parser("feature", help="feature selection commands")
    feature_sub = feature.add_subparsers(dest="feature_command", required=True)
    metadata = feature_sub.add_parser("metadata")
    metadata.add_argument("--project", required=True)
    metadata.add_argument("--run-id", required=True)
    metadata.add_argument("--tables-file", default=None)
    metadata.set_defaults(func=cmd_feature_metadata)

    d01_d02 = feature_sub.add_parser("d01-d02")
    d01_d02.add_argument("--project", required=True)
    d01_d02.add_argument("--run-id", required=True)
    d01_d02.add_argument("--config", default=None)
    d01_d02.add_argument("--table", action="append")
    d01_d02.add_argument("--max-tables", type=int, default=None)
    d01_d02.add_argument("--dry-run-sql", action="store_true")
    d01_d02.add_argument("--refresh-dp-cache", action="store_true")
    d01_d02.add_argument("--sql-approved", action="store_true")
    d01_d02.add_argument("--force", action="store_true")
    d01_d02.set_defaults(func=cmd_feature_d01_d02)

    refine = feature_sub.add_parser("refine")
    refine.add_argument("--project", required=True)
    refine.add_argument("--run-id", required=True)
    refine.add_argument("--config", default=None)
    refine.add_argument("--dry-run-sql", action="store_true")
    refine.add_argument("--refresh-dp-cache", action="store_true")
    refine.add_argument("--sql-approved", action="store_true")
    refine.set_defaults(func=cmd_feature_refine)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="经营建模工作台 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="check local scaffold and dependencies").set_defaults(func=cmd_doctor)
    _add_project_parser(subparsers)
    _add_workflow_parser(subparsers)
    _add_run_parser(subparsers)
    _add_request_parser(subparsers)
    _add_plan_parser(subparsers)
    _add_feature_parser(subparsers)

    status = subparsers.add_parser("status", help="show run state")
    status.add_argument("--project", required=True)
    status.add_argument("--run-id", required=True)
    status.set_defaults(func=cmd_status)

    sample = subparsers.add_parser("sample", help="sample commands")
    sample_sub = sample.add_subparsers(dest="sample_command", required=True)
    check = sample_sub.add_parser("check")
    check.add_argument("--project", required=True)
    check.add_argument("--run-id", required=True)
    check.set_defaults(func=cmd_sample_check)

    train = subparsers.add_parser("train")
    train.add_argument("--project", required=True)
    train.add_argument("--run-id", required=True)
    train.add_argument("--experiment", required=True)
    train.add_argument("--input-feather", default=None)
    train.add_argument("--feature-list", default=None)
    train.add_argument("--score-output", default=None)
    train.add_argument("--input-dir", default=None)
    train.add_argument("--config", default=None)
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--project", required=True)
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--scores-feather", default=None)
    evaluate.add_argument("--output-dir", default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--project", required=True)
    compare.add_argument("--run-id", required=True)
    compare.add_argument("--champion", required=True)
    compare.set_defaults(func=cmd_compare)

    report = subparsers.add_parser("report")
    report.add_argument("--project", required=True)
    report.add_argument("--run-id", required=True)
    report.set_defaults(func=cmd_report)

    init_project = subparsers.add_parser("init-project", help="create a model project workspace")
    init_project.add_argument("--name", required=True)
    init_project.add_argument("--display-name", required=True)
    init_project.add_argument("--scenario", required=True)
    init_project.add_argument("--template", default="generic", choices=["generic", "fujie-gcard"])
    init_project.add_argument("--force", action="store_true")
    init_project.set_defaults(func=cmd_init_project)

    new_run = subparsers.add_parser("new-run", help="legacy alias for run init")
    new_run.add_argument("--project", required=True)
    new_run.add_argument("--step", default="legacy")
    new_run.add_argument("--note", default="")
    new_run.set_defaults(func=lambda args: cmd_run_init(argparse.Namespace(project=args.project, workflow="full_modeling", run_id=None, request=None, plan=None, force=False)))

    screening_summary = subparsers.add_parser("feature-screening-summary")
    screening_summary.add_argument("--project", required=True)
    screening_summary.add_argument("--output", default="reports/feature_screening_process.json")
    screening_summary.set_defaults(func=cmd_feature_screening_summary)

    build_wide_sql = subparsers.add_parser("build-wide-sql")
    build_wide_sql.add_argument("--project", required=True)
    build_wide_sql.add_argument("--remain-features", default="runs/d01_d02_batch_select/results/d01_d02_final_remain_features.json")
    build_wide_sql.add_argument("--sql-output", default="queries/06_build_d01_d02_wide_table.sql")
    build_wide_sql.add_argument("--feature-map-output", default="runs/d01_d02_batch_select/results/d01_d02_wide_feature_map.csv")
    build_wide_sql.add_argument("--summary-output", default="runs/d01_d02_batch_select/results/d01_d02_wide_sql_summary.json")
    build_wide_sql.add_argument("--base-table", default=None)
    build_wide_sql.add_argument("--output-table", default=None)
    build_wide_sql.add_argument("--base-where", default=None)
    build_wide_sql.add_argument("--feature-where", default=None)
    build_wide_sql.set_defaults(func=cmd_build_wide_sql)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
