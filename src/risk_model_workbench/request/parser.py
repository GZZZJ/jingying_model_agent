"""Parse Markdown model request files with YAML front matter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML front matter from a Markdown document."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("YAML front matter starts with --- but has no closing ---")
    raw_meta = text[4:end]
    body = text[end + len("\n---\n") :]
    metadata = yaml.safe_load(raw_meta) or {}
    if not isinstance(metadata, dict):
        raise ValueError("YAML front matter must be a mapping")
    return metadata, body


def parse_model_request(path: str | Path) -> dict[str, Any]:
    """Parse a model request Markdown file."""
    request_path = Path(path).resolve()
    text = request_path.read_text(encoding="utf-8")
    metadata, body = split_front_matter(text)
    return {
        "path": str(request_path),
        "metadata": metadata,
        "body": body,
    }
