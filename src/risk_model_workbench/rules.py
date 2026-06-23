"""Workbench rule registry and lesson promotion helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from risk_model_workbench.paths import REPO_ROOT


RULES_PATH = REPO_ROOT / "docs" / "workbench_rules.yml"
ALLOWED_TARGETS = {"guardrail", "test", "skill", "adr", "glossary"}


def load_workbench_rules(path: str | Path | None = None) -> dict[str, Any]:
    rules_path = Path(path) if path is not None else RULES_PATH
    if not rules_path.exists():
        return {"version": 1, "rules": []}
    with rules_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    payload.setdefault("version", 1)
    payload.setdefault("rules", [])
    return payload


def save_workbench_rules(payload: dict[str, Any], path: str | Path | None = None) -> Path:
    rules_path = Path(path) if path is not None else RULES_PATH
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    payload.setdefault("version", 1)
    payload.setdefault("rules", [])
    with rules_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return rules_path


def promote_lesson_to_rule(
    project_dir: str | Path,
    *,
    title: str,
    target: str,
    rule_id: str,
    note: str = "",
) -> tuple[Path, dict[str, Any]]:
    if target not in ALLOWED_TARGETS:
        raise ValueError(f"unknown rule target: {target}")
    if not rule_id.strip():
        raise ValueError("rule_id cannot be empty")

    project_path = Path(project_dir)
    lesson_path = project_path / "docs" / "lessons.md"
    lesson = _find_lesson(lesson_path, title)
    if lesson is None:
        raise ValueError(f"lesson not found: {title}")

    payload = load_workbench_rules()
    rules = list(payload.get("rules") or [])
    now = datetime.now().isoformat(timespec="seconds")
    entry = {
        "id": rule_id,
        "title": title,
        "target": target,
        "status": "proposed",
        "source": {
            "type": "project_lesson",
            "project": str(project_path.resolve()),
            "lesson_path": _display_path(lesson_path),
            "lesson_title": title,
        },
        "note": note,
        "body": lesson,
        "updated_at": now,
    }

    existing_idx = next((idx for idx, item in enumerate(rules) if item.get("id") == rule_id), None)
    if existing_idx is None:
        entry["created_at"] = now
        rules.append(entry)
    else:
        existing = dict(rules[existing_idx])
        previous_status = existing.get("status") or "proposed"
        existing.update(entry)
        existing["status"] = previous_status
        existing.setdefault("created_at", now)
        rules[existing_idx] = existing
        entry = existing

    payload["rules"] = rules
    return save_workbench_rules(payload), entry


def summarize_rules(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or load_workbench_rules()
    rules = list(payload.get("rules") or [])
    proposed = [item for item in rules if item.get("status") == "proposed"]
    unenforced_guardrails = [
        item for item in rules if item.get("target") == "guardrail" and item.get("status") != "enforced"
    ]
    return {
        "rules_count": len(rules),
        "proposed_count": len(proposed),
        "unenforced_guardrail_count": len(unenforced_guardrails),
        "proposed_rules": _compact_rules(proposed),
        "unenforced_guardrails": _compact_rules(unenforced_guardrails),
    }


def format_rules(payload: dict[str, Any] | None = None) -> str:
    payload = payload or load_workbench_rules()
    rules = list(payload.get("rules") or [])
    lines = ["workbench_rules:"]
    if not rules:
        lines.append("  none")
        return "\n".join(lines) + "\n"
    for item in rules:
        lines.append(
            f"  - {item.get('id')}: {item.get('status', 'unknown')} "
            f"(target={item.get('target', '')}) {item.get('title', '')}"
        )
        if item.get("note"):
            lines.append(f"    note: {item['note']}")
    return "\n".join(lines) + "\n"


def _compact_rules(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.get("id", ""),
            "title": item.get("title", ""),
            "target": item.get("target", ""),
            "status": item.get("status", ""),
        }
        for item in rules
    ]


def _find_lesson(path: Path, title: str) -> str | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    heading = f"## {title}".strip()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == heading:
            start = idx + 1
            break
    if start is None:
        return None
    end = len(lines)
    for idx in range(start, len(lines)):
        if lines[idx].startswith("## "):
            end = idx
            break
    return "\n".join(lines[start:end]).strip()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())
