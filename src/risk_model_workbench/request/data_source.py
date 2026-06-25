"""Helpers for request-level data source mode resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any


REMOTE_TABLE = "remote_table"
LOCAL_FEATHER = "local_feather"
VALID_DATA_SOURCE_MODES = {REMOTE_TABLE, LOCAL_FEATHER}


def sample_location(metadata: dict[str, Any]) -> str:
    return str(metadata.get("sample_location") or "").strip()


def is_feather_location(value: str) -> bool:
    return Path(str(value).strip().lower()).suffix == ".feather"


def resolve_data_source_mode(metadata: dict[str, Any]) -> str:
    raw_mode = str(metadata.get("data_source_mode") or "").strip()
    if raw_mode in VALID_DATA_SOURCE_MODES:
        return raw_mode
    return LOCAL_FEATHER if is_feather_location(sample_location(metadata)) else REMOTE_TABLE


def has_explicit_data_source_mode(metadata: dict[str, Any]) -> bool:
    return str(metadata.get("data_source_mode") or "").strip() in VALID_DATA_SOURCE_MODES


def has_remote_feature_source(metadata: dict[str, Any]) -> bool:
    return bool(str(metadata.get("feature_location") or "").strip())
