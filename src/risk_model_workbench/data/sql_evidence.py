"""SQL evidence registry helpers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


SQL_EVIDENCE_MANIFEST = Path("queries") / "sql_evidence_manifest.json"
VALID_SQL_KINDS = {"user_sql", "generated"}


def _sanitize_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    return value or "sql"


def _relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def load_sql_evidence_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / SQL_EVIDENCE_MANIFEST
    if not path.exists():
        return {"version": 1, "entries": []}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.setdefault("version", 1)
    payload.setdefault("entries", [])
    return payload


def save_sql_evidence_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    path = Path(run_dir) / SQL_EVIDENCE_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.setdefault("version", 1)
    manifest.setdefault("entries", [])
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def write_sql_evidence(
    run_dir: str | Path,
    sql: str,
    *,
    source: str,
    purpose: str,
    stage: str,
    sql_kind: str,
    name: str | None = None,
) -> dict[str, Any]:
    """Write SQL text and update the run-scoped SQL evidence manifest."""
    if sql_kind not in VALID_SQL_KINDS:
        raise ValueError(f"sql_kind must be one of {sorted(VALID_SQL_KINDS)}")

    run_path = Path(run_dir)
    sql_hash = hashlib.sha256(sql.encode("utf-8")).hexdigest()
    filename = _sanitize_name(name or f"{stage}_{purpose}_{sql_hash[:12]}")
    if not filename.endswith(".sql"):
        filename = f"{filename}.sql"
    sql_path = run_path / "queries" / sql_kind / filename
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    sql_path.write_text(sql, encoding="utf-8")

    entry = {
        "path": _relative(sql_path, run_path),
        "source": source,
        "purpose": purpose,
        "stage": stage,
        "sql_kind": sql_kind,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sql_sha256": sql_hash,
    }

    manifest = load_sql_evidence_manifest(run_path)
    entries = [item for item in manifest.get("entries", []) if item.get("path") != entry["path"]]
    entries.append(entry)
    manifest["entries"] = entries
    save_sql_evidence_manifest(run_path, manifest)
    return entry


def is_tracked_sql_evidence_path(path: str | Path) -> bool:
    value = Path(path)
    normalized = value.as_posix()
    if normalized == SQL_EVIDENCE_MANIFEST.as_posix():
        return True
    if value.suffix != ".sql":
        return False
    return normalized.startswith("queries/user_sql/") or normalized.startswith("queries/generated/")
