"""Data-pull engine selection and fakeable remote query adapters."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


ENGINE_AUTO = "auto"
ENGINE_DP_CLI = "dp_cli"
ENGINE_TMLSQLCLIENT = "tmlsqlclient"
SUPPORTED_ENGINES = {ENGINE_AUTO, ENGINE_DP_CLI, ENGINE_TMLSQLCLIENT}


class DataPullEngineError(RuntimeError):
    """Base exception for data-pull engine failures."""


class DataPullEngineUnavailable(DataPullEngineError):
    """Raised when the selected engine is unavailable."""


class SqlApprovalRequired(DataPullEngineError):
    """Raised when a DP pull is attempted without SQL approval."""


@dataclass(frozen=True)
class DataPullEngineSelection:
    platform: str
    data_source_mode: str
    engine: str | None
    override: str
    override_source: str
    available: bool | None
    reason: str


def normalize_platform(system_name: str | None = None) -> str:
    value = (system_name or platform.system() or "").lower()
    if value.startswith("win"):
        return "windows"
    if value.startswith("darwin") or value.startswith("mac"):
        return "macos"
    if value.startswith("linux"):
        return "linux"
    return "other"


def default_engine_for_platform(normalized_platform: str) -> str:
    if normalized_platform in {"windows", "macos"}:
        return ""
    return ENGINE_TMLSQLCLIENT


def is_engine_available(engine: str) -> bool:
    if engine == ENGINE_DP_CLI:
        return shutil.which("dp_cli") is not None
    if engine == ENGINE_TMLSQLCLIENT:
        try:
            return importlib.util.find_spec("tmlpatch.database") is not None
        except ModuleNotFoundError:
            return False
    raise ValueError(f"unsupported data pull engine: {engine}")


def select_data_pull_engine(
    *,
    platform_system: str | None = None,
    override: str | None = ENGINE_AUTO,
    data_source_mode: str = "remote_table",
    availability_checker: Callable[[str], bool] | None = None,
    require_available: bool = False,
) -> DataPullEngineSelection:
    """Select the bulk DP data-pull engine for remote-source feature selection."""
    normalized = normalize_platform(platform_system)
    normalized_override = (override or ENGINE_AUTO).strip().lower()
    if normalized_override not in SUPPORTED_ENGINES:
        raise ValueError(f"unsupported data pull engine override: {override}")

    if data_source_mode == "local_feather":
        return DataPullEngineSelection(
            platform=normalized,
            data_source_mode=data_source_mode,
            engine=None,
            override=normalized_override,
            override_source="bypassed",
            available=None,
            reason="remote data pull engine bypassed for local_feather mode",
        )

    if data_source_mode != "remote_table":
        raise ValueError(f"unsupported data_source_mode: {data_source_mode}")

    if normalized_override == ENGINE_AUTO:
        engine = default_engine_for_platform(normalized)
        override_source = "auto"
        if not engine:
            reason = f"remote DP pull engine is not auto-selected on {normalized}; use local_feather mode for an existing local file or set an explicit engine override"
        else:
            reason = f"default {engine} selected for {normalized}"
    else:
        engine = normalized_override
        override_source = "explicit"
        reason = f"explicit override selected {engine}"

    if not engine:
        available = None
    else:
        checker = availability_checker or is_engine_available
        available = checker(engine)
    if require_available and available is False:
        raise DataPullEngineUnavailable(f"selected data pull engine is unavailable: {engine}")

    return DataPullEngineSelection(
        platform=normalized,
        data_source_mode=data_source_mode,
        engine=engine or None,
        override=normalized_override,
        override_source=override_source,
        available=available,
        reason=reason,
    )


def build_execution_environment_payload(selection: DataPullEngineSelection) -> dict[str, Any]:
    payload = asdict(selection)
    payload["data_pull_engine"] = payload.pop("engine")
    payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    return payload


def write_execution_environment(run_dir: str | Path, selection: DataPullEngineSelection) -> Path:
    path = Path(run_dir) / "feature_selection" / "execution_environment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(build_execution_environment_payload(selection), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def _require_sql_approval(sql_approved: bool) -> None:
    if not sql_approved:
        raise SqlApprovalRequired("SQL approval is required before remote DP data pull")


def _default_tmlsqlclient() -> Any:
    try:
        from tmlpatch.database import TMLSQLClient
    except ImportError as exc:
        raise DataPullEngineUnavailable("TMLSQLClient is unavailable") from exc
    return TMLSQLClient()


def pull_query_to_dataframe(
    sql: str,
    *,
    engine: str,
    client: Any | None = None,
    sql_approved: bool = False,
) -> Any:
    """Pull a reviewed select query into a DataFrame-like object.

    Real clients are deliberately injected or lazily imported so tests can use
    fakes and this module does not perform network work by construction.
    """
    _require_sql_approval(sql_approved)

    if engine == ENGINE_TMLSQLCLIENT:
        active_client = client or _default_tmlsqlclient()
        if not hasattr(active_client, "sql"):
            raise DataPullEngineUnavailable("tmlsqlclient adapter requires a client with sql(sql)")
        result = active_client.sql(sql)
        return result.to_pandas() if hasattr(result, "to_pandas") else result

    if engine == ENGINE_DP_CLI:
        if client is None:
            raise DataPullEngineUnavailable("dp_cli adapter requires an injected client in this pure module")
        if hasattr(client, "query_to_dataframe"):
            return client.query_to_dataframe(sql)
        if callable(client):
            return client(sql)
        raise DataPullEngineUnavailable("dp_cli adapter requires query_to_dataframe(sql) or a callable client")

    raise ValueError(f"unsupported data pull engine: {engine}")


def _shape_metadata(dataframe: Any) -> tuple[int | None, int | None]:
    shape = getattr(dataframe, "shape", None)
    if shape and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    if isinstance(dataframe, dict):
        rows = dataframe.get("rows")
        return (len(rows), None) if isinstance(rows, list) else (None, None)
    return None, None


def pull_query_to_feather(
    sql: str,
    feather_path: str | Path,
    metadata_path: str | Path,
    *,
    engine: str,
    client: Any | None = None,
    sql_approved: bool = False,
) -> dict[str, Any]:
    """Pull a reviewed query and persist a local feather plus metadata."""
    dataframe = pull_query_to_dataframe(sql, engine=engine, client=client, sql_approved=sql_approved)
    output_path = Path(feather_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(dataframe, "to_feather"):
        raise DataPullEngineUnavailable("pulled result does not support to_feather(path)")
    dataframe.to_feather(output_path)

    row_count, column_count = _shape_metadata(dataframe)
    metadata = {
        "engine": engine,
        "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
        "row_count": row_count,
        "column_count": column_count,
        "local_data_path": str(output_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_output = Path(metadata_path)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with metadata_output.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return metadata
