"""Project continuity helpers for long-running modeling workspaces."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from jingying_model_agent.paths import REPO_ROOT, project_config_path
from jingying_model_agent.registry import load_artifact_manifest
from jingying_model_agent.state import load_run_state, run_dir


PROJECT_STATE_VERSION = 1


def project_state_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / "project_state.yml"


def load_project_state(project_dir: str | Path) -> dict[str, Any]:
    path = project_state_path(project_dir)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_project_state(project_dir: str | Path, state: dict[str, Any]) -> Path:
    path = project_state_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    state["version"] = PROJECT_STATE_VERSION
    state["updated_at"] = _now()
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(state, handle, allow_unicode=True, sort_keys=False)
    return path


def update_project_state(
    project_dir: str | Path,
    *,
    active_run_id: str | None = None,
    current_objective: str | None = None,
    status: str | None = None,
    next_actions: list[str] | None = None,
    blockers: list[str] | None = None,
    risks: list[str] | None = None,
    last_verified_commands: list[str] | None = None,
) -> dict[str, Any]:
    project_path = Path(project_dir)
    state = load_project_state(project_path)
    state.setdefault("project", str(project_path.resolve()))
    if active_run_id is not None:
        state["active_run_id"] = active_run_id
    if current_objective is not None:
        state["current_objective"] = current_objective
    if status is not None:
        state["status"] = status
    if next_actions:
        state["next_actions"] = _append_unique(state.get("next_actions", []), next_actions)
    if blockers:
        state["blockers"] = _append_unique(state.get("blockers", []), blockers)
    if risks:
        state["risks"] = _append_unique(state.get("risks", []), risks)
    if last_verified_commands is not None:
        state["last_verified_at"] = _now()
        state["last_verified_commands"] = last_verified_commands
    save_project_state(project_path, state)
    return state


def summarize_project(project_dir: str | Path, run_id: str | None = None) -> dict[str, Any]:
    project_path = Path(project_dir)
    persisted = load_project_state(project_path)
    selected_run_id = run_id or persisted.get("active_run_id") or _latest_run_id(project_path)
    project_info = _load_project_info(project_path)

    summary: dict[str, Any] = {
        "project": str(project_path.resolve()),
        "project_name": project_info.get("name") or project_path.name,
        "display_name": project_info.get("display_name") or project_path.name,
        "active_run_id": selected_run_id,
        "current_objective": persisted.get("current_objective", ""),
        "status": persisted.get("status", "not_started"),
        "last_verified_at": persisted.get("last_verified_at", ""),
        "last_verified_commands": persisted.get("last_verified_commands", []),
        "next_actions": list(persisted.get("next_actions", [])),
        "blockers": list(persisted.get("blockers", [])),
        "risks": list(persisted.get("risks", [])),
        "run": None,
    }

    if not selected_run_id:
        summary["next_actions"] = summary["next_actions"] or ["Initialize a run or set active_run_id in project_state.yml."]
        return summary

    selected_run_dir = run_dir(project_path, selected_run_id)
    try:
        run_state = load_run_state(selected_run_dir)
    except FileNotFoundError:
        summary["status"] = "blocked"
        summary["blockers"] = _append_unique(summary["blockers"], [f"active_run_id does not exist: {selected_run_id}"])
        return summary

    stage_rows = _stage_rows(run_state)
    inferred_next, inferred_blockers, inferred_risks = _infer_run_followups(run_state)
    summary["status"] = persisted.get("status") or run_state.get("status", "unknown")
    summary["next_actions"] = summary["next_actions"] or inferred_next
    summary["blockers"] = _append_unique(summary["blockers"], inferred_blockers)
    summary["risks"] = _append_unique(summary["risks"], inferred_risks)
    summary["run"] = {
        "run_id": run_state.get("run_id", selected_run_id),
        "workflow": run_state.get("workflow", ""),
        "status": run_state.get("status", ""),
        "current_stage": run_state.get("current_stage", ""),
        "updated_at": run_state.get("updated_at", ""),
        "stage_counts": _stage_counts(stage_rows),
        "stages": stage_rows,
        "recent_decisions": list(run_state.get("decisions", []))[-5:],
    }
    return summary


def write_project_state_from_summary(project_dir: str | Path, summary: dict[str, Any], commands: list[str]) -> Path:
    project_path = Path(project_dir)
    state = load_project_state(project_path)
    state.setdefault("project", str(project_path.resolve()))
    if summary.get("active_run_id"):
        state["active_run_id"] = summary["active_run_id"]
    state.setdefault("current_objective", summary.get("current_objective", ""))
    state["status"] = summary.get("status", state.get("status", "unknown"))
    state.setdefault("next_actions", summary.get("next_actions", []))
    state["blockers"] = _append_unique(state.get("blockers", []), summary.get("blockers", []))
    state["risks"] = _append_unique(state.get("risks", []), summary.get("risks", []))
    state["last_verified_at"] = _now()
    state["last_verified_commands"] = commands
    return save_project_state(project_path, state)


def format_project_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"project: {summary.get('display_name') or summary.get('project_name')}",
        f"path: {summary.get('project')}",
        f"status: {summary.get('status')}",
        f"active_run_id: {summary.get('active_run_id') or ''}",
    ]
    if summary.get("current_objective"):
        lines.append(f"current_objective: {summary['current_objective']}")
    if summary.get("last_verified_at"):
        lines.append(f"last_verified_at: {summary['last_verified_at']}")

    run = summary.get("run")
    if run:
        counts = ", ".join(f"{key}={value}" for key, value in sorted(run.get("stage_counts", {}).items()))
        lines.extend(
            [
                "",
                "run:",
                f"  workflow: {run.get('workflow')}",
                f"  status: {run.get('status')}",
                f"  current_stage: {run.get('current_stage')}",
                f"  stage_counts: {counts}",
            ]
        )

    _extend_list(lines, "next_actions", summary.get("next_actions", []))
    _extend_list(lines, "blockers", summary.get("blockers", []))
    _extend_list(lines, "risks", summary.get("risks", []))
    return "\n".join(lines) + "\n"


def write_handoff(
    project_dir: str | Path,
    *,
    run_id: str | None = None,
    note: str = "",
    output: str | Path | None = None,
) -> Path:
    project_path = Path(project_dir)
    summary = summarize_project(project_path, run_id=run_id)
    selected_run_id = summary.get("active_run_id") or "no-run"
    if output is None:
        output_path = project_path / "handoffs" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-{selected_run_id}.md"
    else:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = project_path / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_handoff(summary, note=note), encoding="utf-8")

    state = update_project_state(
        project_path,
        active_run_id=summary.get("active_run_id"),
        status=summary.get("status"),
        next_actions=summary.get("next_actions", []),
        blockers=summary.get("blockers", []),
        risks=summary.get("risks", []),
    )
    state["last_handoff"] = str(output_path)
    save_project_state(project_path, state)
    return output_path


def format_handoff(summary: dict[str, Any], *, note: str = "") -> str:
    run = summary.get("run") or {}
    lines = [
        f"# Handoff - {summary.get('display_name') or summary.get('project_name')}",
        "",
        f"- generated_at: {_now()}",
        f"- project: {summary.get('project')}",
        f"- active_run_id: {summary.get('active_run_id') or ''}",
        f"- status: {summary.get('status')}",
    ]
    if summary.get("current_objective"):
        lines.append(f"- current_objective: {summary['current_objective']}")
    if note:
        lines.extend(["", "## Note", "", note])

    lines.extend(["", "## Source Of Truth", ""])
    lines.append("- project_state.yml")
    if summary.get("active_run_id"):
        lines.append(f"- runs/{summary['active_run_id']}/run_state.yml")
        lines.append(f"- runs/{summary['active_run_id']}/audit/artifact_manifest.json")

    _extend_markdown_list(lines, "Next Actions", summary.get("next_actions", []))
    _extend_markdown_list(lines, "Blockers", summary.get("blockers", []))
    _extend_markdown_list(lines, "Risks", summary.get("risks", []))

    if run:
        lines.extend(["", "## Run Summary", ""])
        lines.append(f"- workflow: {run.get('workflow')}")
        lines.append(f"- run_status: {run.get('status')}")
        lines.append(f"- current_stage: {run.get('current_stage')}")
        lines.extend(["", "| Stage | Status | Artifacts |", "| --- | --- | ---: |"])
        for stage in run.get("stages", []):
            lines.append(f"| {stage['name']} | {stage['status']} | {stage['artifact_count']} |")
        decisions = run.get("recent_decisions", [])
        if decisions:
            lines.extend(["", "## Recent Decisions", ""])
            for decision in decisions:
                lines.append(
                    f"- {decision.get('created_at', '')} [{decision.get('stage', '')}] "
                    f"{decision.get('decision', '')}: {decision.get('reason', '')}"
                )
    lines.append("")
    return "\n".join(lines)


def append_lesson(
    project_dir: str | Path,
    *,
    title: str,
    body: str,
    kind: str,
    scope: str = "project",
    source: str = "",
    tags: list[str] | None = None,
) -> Path:
    if not body.strip():
        raise ValueError("lesson body cannot be empty")
    path = _lesson_path(project_dir, scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("# Lessons\n\n", encoding="utf-8")

    tags_text = ", ".join(tags or [])
    entry = (
        f"\n## {title}\n\n"
        f"- captured_at: {_now()}\n"
        f"- kind: {kind}\n"
        f"- scope: {scope}\n"
        f"- source: {source or 'manual'}\n"
        f"- tags: {tags_text}\n\n"
        f"{body.strip()}\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    return path


def audit_run(project_dir: str | Path, run_id: str, *, stage: str | None = None) -> dict[str, Any]:
    project_path = Path(project_dir)
    selected_run_dir = run_dir(project_path, run_id)
    run_state = load_run_state(selected_run_dir)
    manifest = load_artifact_manifest(selected_run_dir)
    manifest_artifacts = manifest.get("artifacts", [])
    manifest_by_stage: dict[str, list[dict[str, Any]]] = {}
    for artifact in manifest_artifacts:
        manifest_by_stage.setdefault(str(artifact.get("stage", "")), []).append(artifact)

    stage_states = run_state.get("stages") or {}
    selected_names = [stage] if stage else list(stage_states.keys())
    stage_results = []
    for name in selected_names:
        state = stage_states.get(name)
        if state is None:
            stage_results.append(
                {
                    "stage": name,
                    "status": "missing",
                    "verdict": "missing",
                    "artifact_count": 0,
                    "registered_count": 0,
                    "issues": [f"stage is not present in run_state.yml: {name}"],
                }
            )
            continue
        stage_results.append(_audit_stage(name, state, manifest_by_stage.get(name, []), run_state))

    verdict = _rollup_audit_verdict(stage_results)
    return {
        "project": str(project_path.resolve()),
        "run_id": run_id,
        "workflow": run_state.get("workflow", ""),
        "run_status": run_state.get("status", ""),
        "stage": stage or "",
        "verdict": verdict,
        "source_of_truth": [
            f"runs/{run_id}/run_state.yml",
            f"runs/{run_id}/audit/artifact_manifest.json",
        ],
        "stages": stage_results,
    }


def format_run_audit(audit: dict[str, Any]) -> str:
    lines = [
        f"run_id: {audit.get('run_id')}",
        f"workflow: {audit.get('workflow')}",
        f"run_status: {audit.get('run_status')}",
        f"verdict: {audit.get('verdict')}",
        "",
        "stages:",
    ]
    for stage in audit.get("stages", []):
        lines.append(
            f"  - {stage['stage']}: {stage['verdict']} "
            f"(status={stage['status']}, artifacts={stage['artifact_count']}, registered={stage['registered_count']})"
        )
        for issue in stage.get("issues", []):
            lines.append(f"    issue: {issue}")
    return "\n".join(lines) + "\n"


def write_retrospective(
    project_dir: str | Path,
    *,
    run_id: str | None = None,
    scope: str = "session",
    stage: str | None = None,
    outcome: str = "",
    note: str = "",
    lessons: list[str] | None = None,
    output: str | Path | None = None,
) -> Path:
    if scope == "stage" and not stage:
        raise ValueError("stage retrospective requires --stage")

    project_path = Path(project_dir)
    summary = summarize_project(project_path, run_id=run_id)
    selected_run_id = run_id or summary.get("active_run_id")
    audit = audit_run(project_path, selected_run_id, stage=stage) if selected_run_id else None
    if output is None:
        suffix = stage if scope == "stage" else (selected_run_id or "no-run")
        output_path = project_path / "retrospectives" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}-{scope}-{suffix}.md"
    else:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = project_path / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_retrospective(summary, audit=audit, scope=scope, stage=stage, outcome=outcome, note=note, lessons=lessons or []),
        encoding="utf-8",
    )

    state = update_project_state(
        project_path,
        active_run_id=selected_run_id,
        status=summary.get("status"),
        next_actions=summary.get("next_actions", []),
        blockers=summary.get("blockers", []),
        risks=summary.get("risks", []),
    )
    state["last_retrospective"] = str(output_path)
    save_project_state(project_path, state)
    return output_path


def format_retrospective(
    summary: dict[str, Any],
    *,
    audit: dict[str, Any] | None,
    scope: str,
    stage: str | None,
    outcome: str,
    note: str,
    lessons: list[str],
) -> str:
    lines = [
        f"# Retrospective - {summary.get('display_name') or summary.get('project_name')}",
        "",
        f"- generated_at: {_now()}",
        "- trigger: explicit",
        f"- scope: {scope}",
        f"- project: {summary.get('project')}",
        f"- active_run_id: {summary.get('active_run_id') or ''}",
    ]
    if stage:
        lines.append(f"- stage: {stage}")
    if outcome:
        lines.append(f"- outcome: {outcome}")
    if note:
        lines.extend(["", "## Note", "", note])

    lines.extend(["", "## Source Of Truth", "", "- project_state.yml"])
    if summary.get("active_run_id"):
        lines.append(f"- runs/{summary['active_run_id']}/run_state.yml")
        lines.append(f"- runs/{summary['active_run_id']}/audit/artifact_manifest.json")

    if audit:
        lines.extend(["", "## Audit", "", f"- verdict: {audit.get('verdict')}"])
        lines.extend(["", "| Stage | Status | Verdict | Artifacts | Registered |", "| --- | --- | --- | ---: | ---: |"])
        for item in audit.get("stages", []):
            lines.append(
                f"| {item['stage']} | {item['status']} | {item['verdict']} | "
                f"{item['artifact_count']} | {item['registered_count']} |"
            )
        issues = _append_unique([], [issue for item in audit.get("stages", []) for issue in item.get("issues", [])])
        _extend_markdown_list(lines, "Audit Issues", issues)

    _extend_markdown_list(lines, "Next Actions", summary.get("next_actions", []))
    _extend_markdown_list(lines, "Risks", summary.get("risks", []))
    _extend_markdown_list(lines, "Lessons", lessons)
    lines.append("")
    return "\n".join(lines)


def _lesson_path(project_dir: str | Path, scope: str) -> Path:
    if scope == "project":
        return Path(project_dir) / "docs" / "lessons.md"
    if scope == "workbench":
        return REPO_ROOT / "docs" / "workbench_lessons.md"
    raise ValueError(f"unknown lesson scope: {scope}")


def _audit_stage(name: str, state: dict[str, Any], manifest_items: list[dict[str, Any]], run_state: dict[str, Any]) -> dict[str, Any]:
    status = state.get("status", "unknown")
    artifacts = state.get("artifacts") or []
    registered_paths = {str(item.get("path")) for item in manifest_items}
    missing_registrations = [artifact for artifact in artifacts if str(artifact) not in registered_paths]
    missing_files = [
        str(item.get("path"))
        for item in manifest_items
        if item.get("kind") != "directory" and item.get("exists") is False
    ]
    scaffold_sources = [item for item in manifest_items if item.get("source") == "scaffold"]
    imported_sources = [item for item in manifest_items if item.get("source") == "imported"]
    issues: list[str] = []
    issues.extend(f"artifact listed in run_state.yml but not registered: {item}" for item in missing_registrations)
    issues.extend(f"registered artifact does not exist: {item}" for item in missing_files)

    if status in {"pending", "running", "failed", "missing"}:
        verdict = "open"
    elif status == "scaffold" or scaffold_sources:
        verdict = "scaffold"
        issues.append("scaffold evidence is not real modeling evidence")
    elif status == "done" and not artifacts and not manifest_items:
        verdict = "incomplete"
        issues.append("done stage has no registered artifacts")
    elif issues:
        verdict = "incomplete"
    elif status == "done" and (imported_sources or str(run_state.get("workflow", "")).startswith("imported")):
        verdict = "imported"
        issues.append("imported evidence should be reviewed before treating the stage as locally reproduced")
    elif status == "done":
        verdict = "complete"
    else:
        verdict = "unknown"

    return {
        "stage": name,
        "status": status,
        "verdict": verdict,
        "artifact_count": len(artifacts),
        "registered_count": len(manifest_items),
        "issues": issues,
    }


def _rollup_audit_verdict(stage_results: list[dict[str, Any]]) -> str:
    verdicts = {item.get("verdict") for item in stage_results}
    if not stage_results:
        return "empty"
    if verdicts <= {"complete"}:
        return "complete"
    if "open" in verdicts or "missing" in verdicts:
        return "open"
    if "incomplete" in verdicts:
        return "incomplete"
    if "scaffold" in verdicts:
        return "scaffold"
    if "imported" in verdicts:
        return "imported"
    return "unknown"


def _latest_run_id(project_dir: Path) -> str | None:
    candidates: list[tuple[datetime, str]] = []
    for state_file in (project_dir / "runs").glob("*/run_state.yml"):
        try:
            state = yaml.safe_load(state_file.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        run_id = str(state.get("run_id") or state_file.parent.name)
        timestamp = _parse_datetime(state.get("updated_at") or state.get("created_at")) or datetime.fromtimestamp(state_file.stat().st_mtime)
        candidates.append((timestamp, run_id))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1]))[-1][1]


def _load_project_info(project_dir: Path) -> dict[str, Any]:
    config_path = project_config_path(project_dir)
    if not config_path.exists():
        return {}
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    project = config.get("project")
    return project if isinstance(project, dict) else {}


def _stage_rows(run_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, stage in (run_state.get("stages") or {}).items():
        artifacts = stage.get("artifacts") if isinstance(stage, dict) else []
        rows.append(
            {
                "name": name,
                "status": stage.get("status", "unknown") if isinstance(stage, dict) else "unknown",
                "artifact_count": len(artifacts or []),
            }
        )
    return rows


def _stage_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _infer_run_followups(run_state: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    stages = run_state.get("stages") or {}
    failed = [name for name, stage in stages.items() if stage.get("status") == "failed"]
    running = [name for name, stage in stages.items() if stage.get("status") == "running"]
    pending = [name for name, stage in stages.items() if stage.get("status") == "pending"]
    scaffold = [name for name, stage in stages.items() if stage.get("status") == "scaffold"]

    next_actions: list[str] = []
    blockers = [f"stage failed: {name}" for name in failed]
    risks: list[str] = []
    if failed:
        next_actions.append(f"Investigate failed stage: {failed[0]}")
    elif running:
        next_actions.append(f"Resume or reconcile running stage: {running[0]}")
    elif pending:
        next_actions.append(f"Continue or explicitly reconcile pending stage: {pending[0]}")
    elif scaffold:
        next_actions.append(f"Replace scaffold artifacts with real evidence where required: {scaffold[0]}")
    else:
        next_actions.append("Review final artifacts and capture lessons before closing the project.")

    if str(run_state.get("workflow", "")).startswith("imported"):
        risks.append("imported run is not proof that the full workflow was rerun locally")
    if scaffold:
        risks.append("scaffold artifacts exist and must not be treated as real modeling evidence")
    return next_actions, blockers, risks


def _append_unique(existing: list[Any], new_items: list[Any]) -> list[Any]:
    result = list(existing or [])
    for item in new_items or []:
        if item and item not in result:
            result.append(item)
    return result


def _extend_list(lines: list[str], title: str, values: list[str]) -> None:
    if not values:
        return
    lines.extend(["", f"{title}:"])
    for value in values:
        lines.append(f"  - {value}")


def _extend_markdown_list(lines: list[str], title: str, values: list[str]) -> None:
    if not values:
        return
    lines.extend(["", f"## {title}", ""])
    for value in values:
        lines.append(f"- {value}")


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
