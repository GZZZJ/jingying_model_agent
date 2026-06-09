"""Artifact registry helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jingying_model_agent.manifest import describe_file


def manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "audit" / "artifact_manifest.json"


def load_artifact_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = manifest_path(run_dir)
    if not path.exists():
        return {"version": 1, "artifacts": []}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_artifact_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    path = manifest_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.setdefault("version", 1)
    manifest.setdefault("artifacts", [])
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def register_artifact(
    run_dir: str | Path,
    artifact: str | Path,
    *,
    stage: str,
    kind: str = "file",
    source: str = "generated",
    description: str = "",
) -> dict[str, Any]:
    """Register an artifact relative to the run directory when possible."""
    run_path = Path(run_dir).resolve()
    artifact_path = Path(artifact)
    if not artifact_path.is_absolute():
        artifact_path = run_path / artifact_path
    artifact_path = artifact_path.resolve()

    if artifact_path.exists() and artifact_path.is_file():
        entry = describe_file(artifact_path, run_path)
    else:
        try:
            display_path = artifact_path.relative_to(run_path)
        except ValueError:
            display_path = artifact_path
        entry = {"path": str(display_path), "exists": artifact_path.exists()}

    entry.update(
        {
            "stage": stage,
            "kind": kind,
            "source": source,
            "description": description,
            "registered_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    manifest = load_artifact_manifest(run_path)
    artifacts = [
        item for item in manifest.get("artifacts", []) if not (item.get("path") == entry["path"] and item.get("stage") == stage)
    ]
    artifacts.append(entry)
    manifest["artifacts"] = artifacts
    save_artifact_manifest(run_path, manifest)
    return entry
