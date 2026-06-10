"""Path helpers for the local modeling workbench."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a project path relative to the repository root."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def project_config_path(project_dir: str | Path) -> Path:
    """Return the preferred project config path, accepting legacy project.yaml."""
    project_path = Path(project_dir)
    preferred = project_path / "project.yml"
    if preferred.exists():
        return preferred
    legacy = project_path / "project.yaml"
    if legacy.exists():
        return legacy
    return preferred


def workflow_path(workflow: str) -> Path:
    """Resolve a workflow name or path."""
    path = Path(workflow)
    if path.suffix:
        return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    return REPO_ROOT / "workflows" / f"{workflow}.yml"
