"""Run manifest helpers for reproducible model workflows."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any


def make_run_id(now: datetime | None = None) -> str:
    """Return a sortable local run id."""
    current = now or datetime.now()
    return current.strftime("%Y%m%d_%H%M%S_%f")


def sha256_file(path: str | Path) -> str:
    """Compute a sha256 hash for a file."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: str | Path, base_dir: str | Path | None = None) -> dict[str, Any]:
    """Return path, size, mtime, and hash metadata for an existing file."""
    file_path = Path(path)
    display_path = file_path
    if base_dir:
        try:
            display_path = file_path.relative_to(Path(base_dir))
        except ValueError:
            display_path = file_path

    return {
        "path": str(display_path),
        "size_bytes": file_path.stat().st_size,
        "mtime": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(timespec="seconds"),
        "sha256": sha256_file(file_path),
    }


def write_manifest(
    project_dir: str | Path,
    step: str,
    *,
    run_id: str | None = None,
    inputs: list[str | Path] | None = None,
    outputs: list[str | Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Create a run manifest and return its path."""
    project_path = Path(project_dir).resolve()
    current_run_id = run_id or make_run_id()
    run_dir = project_path / "runs" / current_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    existing_inputs = [Path(item).resolve() for item in (inputs or []) if Path(item).exists()]
    existing_outputs = [Path(item).resolve() for item in (outputs or []) if Path(item).exists()]

    manifest = {
        "run_id": current_run_id,
        "step": step,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_dir": str(project_path),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "inputs": [describe_file(item, project_path) for item in existing_inputs],
        "outputs": [describe_file(item, project_path) for item in existing_outputs],
        "extra": extra or {},
    }

    manifest_path = run_dir / "artifacts_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest_path
