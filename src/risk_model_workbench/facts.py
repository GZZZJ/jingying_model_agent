"""Source-backed project fact store."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from risk_model_workbench.paths import REPO_ROOT


FACT_STORE_VERSION = 1
FACT_CATEGORIES = (
    "business_definition",
    "label_definition",
    "approval",
    "lesson",
    "risk",
    "decision",
)


def fact_store_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / "project_facts.yml"


def load_fact_store(project_dir: str | Path) -> dict[str, Any]:
    path = fact_store_path(project_dir)
    if not path.exists():
        return {"version": FACT_STORE_VERSION, "facts": []}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    payload.setdefault("version", FACT_STORE_VERSION)
    payload.setdefault("facts", [])
    return payload


def save_fact_store(project_dir: str | Path, payload: dict[str, Any]) -> Path:
    path = fact_store_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["version"] = FACT_STORE_VERSION
    payload.setdefault("facts", [])
    payload["updated_at"] = _now()
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    return path


def list_facts(project_dir: str | Path, *, category: str | None = None) -> list[dict[str, Any]]:
    payload = load_fact_store(project_dir)
    facts = list(payload.get("facts") or [])
    if category:
        facts = [fact for fact in facts if fact.get("category") == category]
    return facts


def add_fact(
    project_dir: str | Path,
    *,
    category: str,
    statement: str,
    source_path: str | Path,
    source_type: str = "manual",
    source_ref: str = "",
    confidence: str = "confirmed",
) -> tuple[Path, dict[str, Any]]:
    if category not in FACT_CATEGORIES:
        raise ValueError(f"unknown fact category: {category}")
    if not statement.strip():
        raise ValueError("fact statement cannot be empty")
    source = _resolve_source_path(project_dir, source_path)
    if not source.exists():
        raise ValueError(f"fact source path does not exist: {source}")

    payload = load_fact_store(project_dir)
    facts = list(payload.get("facts") or [])
    now = _now()
    fact = {
        "id": _next_fact_id(facts),
        "category": category,
        "statement": statement.strip(),
        "source_type": source_type,
        "source_path": _display_source_path(project_dir, source),
        "source_ref": source_ref,
        "confidence": confidence,
        "created_at": now,
        "updated_at": now,
    }
    facts.append(fact)
    payload["facts"] = facts
    path = save_fact_store(project_dir, payload)
    return path, fact


def format_facts(facts: list[dict[str, Any]]) -> str:
    if not facts:
        return "facts: []\n"
    lines = ["facts:"]
    for fact in facts:
        lines.append(f"- {fact.get('id')} [{fact.get('category')}] {fact.get('statement')}")
        lines.append(f"  source: {fact.get('source_path')}")
    return "\n".join(lines) + "\n"


def _resolve_source_path(project_dir: str | Path, source_path: str | Path) -> Path:
    raw = Path(source_path)
    if raw.is_absolute():
        return raw
    project_candidate = Path(project_dir) / raw
    if project_candidate.exists():
        return project_candidate
    repo_candidate = REPO_ROOT / raw
    if repo_candidate.exists():
        return repo_candidate
    return project_candidate


def _display_source_path(project_dir: str | Path, source: Path) -> str:
    project_path = Path(project_dir).resolve()
    try:
        return str(source.resolve().relative_to(project_path))
    except ValueError:
        try:
            return str(source.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(source.resolve())


def _next_fact_id(facts: list[dict[str, Any]]) -> str:
    return f"fact_{len(facts) + 1:04d}"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
