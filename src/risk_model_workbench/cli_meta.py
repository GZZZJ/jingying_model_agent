"""CLI wiring for workbench metadata, registry, auditor, fact, and snapshot commands."""

from __future__ import annotations

import argparse
import json

from risk_model_workbench.auditors import (
    format_auditor_list,
    format_auditor_result,
    list_auditors,
    run_auditor,
)
from risk_model_workbench.context_snapshot import write_context_snapshot
from risk_model_workbench.facts import FACT_CATEGORIES, add_fact, format_facts, list_facts
from risk_model_workbench.harness.actions import (
    action_to_dict,
    format_action_detail,
    format_action_list,
    get_action_spec,
    list_action_specs,
)
from risk_model_workbench.harness.tools import (
    format_tool_detail,
    format_tool_list,
    get_tool_spec,
    list_tool_specs,
    tool_to_dict,
)
from risk_model_workbench.paths import resolve_project_path


def cmd_action_list(args: argparse.Namespace) -> int:
    specs = list_action_specs(kind=args.kind)
    if args.json:
        print(json.dumps([action_to_dict(spec) for spec in specs], ensure_ascii=False, indent=2))
    else:
        print(format_action_list(specs), end="")
    return 0


def cmd_action_show(args: argparse.Namespace) -> int:
    try:
        spec = get_action_spec(args.action_id)
    except KeyError as exc:
        print(str(exc))
        return 1
    if args.json:
        print(json.dumps(action_to_dict(spec), ensure_ascii=False, indent=2))
    else:
        print(format_action_detail(spec), end="")
    return 0


def cmd_tool_list(args: argparse.Namespace) -> int:
    specs = list_tool_specs(permission=args.permission)
    if args.json:
        print(json.dumps([tool_to_dict(spec) for spec in specs], ensure_ascii=False, indent=2))
    else:
        print(format_tool_list(specs), end="")
    return 0


def cmd_tool_show(args: argparse.Namespace) -> int:
    try:
        spec = get_tool_spec(args.tool_name)
    except KeyError as exc:
        print(str(exc))
        return 1
    if args.json:
        print(json.dumps(tool_to_dict(spec), ensure_ascii=False, indent=2))
    else:
        print(format_tool_detail(spec), end="")
    return 0


def cmd_auditor_list(args: argparse.Namespace) -> int:
    specs = list_auditors()
    if args.json:
        print(json.dumps([spec.to_dict() for spec in specs], ensure_ascii=False, indent=2))
    else:
        print(format_auditor_list(specs), end="")
    return 0


def cmd_auditor_run(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    try:
        result = run_auditor(args.name, project_dir, args.run_id)
    except KeyError as exc:
        print(str(exc))
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_auditor_result(result), end="")
    return 0


def cmd_fact_list(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    facts = list_facts(project_dir, category=args.category)
    if args.json:
        print(json.dumps(facts, ensure_ascii=False, indent=2))
    else:
        print(format_facts(facts), end="")
    return 0


def cmd_fact_add(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    try:
        path, fact = add_fact(
            project_dir,
            category=args.category,
            statement=args.statement,
            source_path=args.source_path,
            source_type=args.source_type,
            source_ref=args.source_ref or "",
            confidence=args.confidence,
        )
    except ValueError as exc:
        print(f"fact add failed: {exc}")
        return 1
    print(f"facts: {path}")
    print(f"fact_id: {fact.get('id')}")
    return 0


def cmd_context_snapshot(args: argparse.Namespace) -> int:
    project_dir = resolve_project_path(args.project)
    output_path, markdown_path = write_context_snapshot(project_dir, args.run_id, output=args.output, markdown=args.markdown)
    print(f"context_snapshot: {output_path}")
    if markdown_path:
        print(f"context_snapshot_markdown: {markdown_path}")
    return 0


def add_metadata_parsers(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    _add_action_parser(subparsers)
    _add_tool_parser(subparsers)
    _add_auditor_parser(subparsers)
    _add_fact_parser(subparsers)
    _add_context_parser(subparsers)


def _add_action_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    action = subparsers.add_parser("action", help="harness action registry commands")
    action_sub = action.add_subparsers(dest="action_command", required=True)
    list_cmd = action_sub.add_parser("list", help="list harness action specs")
    list_cmd.add_argument("--kind", choices=["stage", "audit", "utility"], default=None)
    list_cmd.add_argument("--json", action="store_true")
    list_cmd.set_defaults(func=cmd_action_list)
    show = action_sub.add_parser("show", help="show one harness action spec")
    show.add_argument("action_id")
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=cmd_action_show)


def _add_tool_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    tool = subparsers.add_parser("tool", help="permission-scoped tool registry commands")
    tool_sub = tool.add_subparsers(dest="tool_command", required=True)
    list_cmd = tool_sub.add_parser("list", help="list registered tools")
    list_cmd.add_argument("--permission", choices=["read_only", "writes_run", "dp_sql_pull", "external_data"], default=None)
    list_cmd.add_argument("--json", action="store_true")
    list_cmd.set_defaults(func=cmd_tool_list)
    show = tool_sub.add_parser("show", help="show one registered tool")
    show.add_argument("tool_name")
    show.add_argument("--json", action="store_true")
    show.set_defaults(func=cmd_tool_show)


def _add_auditor_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    auditor = subparsers.add_parser("auditor", help="read-only run auditor commands")
    auditor_sub = auditor.add_subparsers(dest="auditor_command", required=True)
    list_cmd = auditor_sub.add_parser("list", help="list read-only auditors")
    list_cmd.add_argument("--json", action="store_true")
    list_cmd.set_defaults(func=cmd_auditor_list)
    run_cmd = auditor_sub.add_parser("run", help="run one read-only auditor")
    run_cmd.add_argument("name")
    run_cmd.add_argument("--project", required=True)
    run_cmd.add_argument("--run-id", required=True)
    run_cmd.add_argument("--json", action="store_true")
    run_cmd.set_defaults(func=cmd_auditor_run)


def _add_fact_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    fact = subparsers.add_parser("fact", help="source-backed project fact commands")
    fact_sub = fact.add_subparsers(dest="fact_command", required=True)
    list_cmd = fact_sub.add_parser("list", help="list project facts")
    list_cmd.add_argument("--project", required=True)
    list_cmd.add_argument("--category", choices=FACT_CATEGORIES, default=None)
    list_cmd.add_argument("--json", action="store_true")
    list_cmd.set_defaults(func=cmd_fact_list)
    add_cmd = fact_sub.add_parser("add", help="add a source-backed project fact")
    add_cmd.add_argument("--project", required=True)
    add_cmd.add_argument("--category", choices=FACT_CATEGORIES, required=True)
    add_cmd.add_argument("--statement", required=True)
    add_cmd.add_argument("--source-path", required=True)
    add_cmd.add_argument("--source-type", default="manual")
    add_cmd.add_argument("--source-ref", default="")
    add_cmd.add_argument("--confidence", choices=["confirmed", "inferred"], default="confirmed")
    add_cmd.set_defaults(func=cmd_fact_add)


def _add_context_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    context = subparsers.add_parser("context", help="context snapshot commands")
    context_sub = context.add_subparsers(dest="context_command", required=True)
    snapshot = context_sub.add_parser("snapshot", help="write an explicit run context snapshot")
    snapshot.add_argument("--project", required=True)
    snapshot.add_argument("--run-id", required=True)
    snapshot.add_argument("--output", default=None)
    snapshot.add_argument("--markdown", action="store_true")
    snapshot.set_defaults(func=cmd_context_snapshot)
