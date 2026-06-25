"""Resource-aware capacity planning helpers for feature-selection intake."""

from __future__ import annotations

import math
import os
import platform as platform_module
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MEMORY_BUDGET_FRACTION = 0.8
DEFAULT_BYTES_PER_NUMERIC_VALUE = 8
DEFAULT_PEAK_MULTIPLIERS = {
    "feature_prescreen": 3.0,
    "prescreen": 3.0,
    "feature_refine": 4.0,
    "refine": 4.0,
    "model_importance": 4.0,
}
CAPACITY_FORMULA = (
    "floor((available_memory_bytes * memory_budget_fraction / peak_multiplier) "
    "/ ((feature_column_count + required_non_feature_column_count) * bytes_per_numeric_value))"
)


class MemoryProbeError(RuntimeError):
    """Raised when memory cannot be probed and no override is supplied."""


@dataclass(frozen=True)
class MemorySnapshot:
    total_bytes: int
    available_bytes: int
    platform: str
    source: str


@dataclass(frozen=True)
class CapacityEstimate:
    total_memory_bytes: int
    available_memory_bytes: int
    memory_budget_fraction: float
    memory_budget_bytes: int
    peak_multiplier: float
    matrix_budget_bytes: int
    feature_column_count: int
    required_non_feature_column_count: int
    bytes_per_numeric_value: int
    row_width_bytes: int
    max_rows: int
    formula: str = CAPACITY_FORMULA


@dataclass(frozen=True)
class UniformSamplingDecision:
    total_rows: int
    max_rows: int
    sampling_required: bool
    ratio: float
    estimated_rows: int
    limit: int | None
    reason: str


def normalize_platform_name(system_name: str | None = None) -> str:
    """Normalize platform names for resource evidence."""
    value = (system_name or platform_module.system() or "").lower()
    if value.startswith("darwin") or value.startswith("mac"):
        return "macos"
    if value.startswith("win"):
        return "windows"
    if value.startswith("linux"):
        return "linux"
    return "other"


def _positive_int(value: int | None, field_name: str) -> int:
    if value is None or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _probe_memory_from_sysconf() -> tuple[int | None, int | None]:
    page_size = None
    total_pages = None
    available_pages = None
    for key in ("SC_PAGE_SIZE", "SC_PAGESIZE"):
        try:
            page_size = os.sysconf(key)
            break
        except (ValueError, OSError, AttributeError):
            continue
    try:
        total_pages = os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        total_pages = None
    try:
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        available_pages = None
    if not page_size:
        return None, None
    total = int(total_pages * page_size) if total_pages else None
    available = int(available_pages * page_size) if available_pages else None
    return total, available


def _probe_linux_meminfo() -> tuple[int | None, int | None]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None, None
    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            values[parts[0].rstrip(":")] = int(parts[1]) * 1024
    return values.get("MemTotal"), values.get("MemAvailable") or values.get("MemFree")


def _probe_macos_total_memory() -> int | None:
    try:
        output = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return int(output.strip())
    except ValueError:
        return None


def _probe_macos_available_memory() -> int | None:
    try:
        output = subprocess.check_output(["vm_stat"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None

    page_size = 4096
    pages: dict[str, int] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if "page size of" in stripped:
            digits = "".join(char if char.isdigit() else " " for char in stripped).split()
            if digits:
                page_size = int(digits[-1])
            continue
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        digits = "".join(char for char in raw_value if char.isdigit())
        if digits:
            pages[key.strip().lower()] = int(digits)

    available_pages = (
        pages.get("pages free", 0)
        + pages.get("pages inactive", 0)
        + pages.get("pages speculative", 0)
    )
    return int(available_pages * page_size) if available_pages else None


def probe_memory(
    *,
    total_bytes: int | None = None,
    available_bytes: int | None = None,
    platform_name: str | None = None,
) -> MemorySnapshot:
    """Probe total and available memory, with explicit overrides for tests/runs."""
    normalized_platform = normalize_platform_name(platform_name)
    if total_bytes is not None and available_bytes is not None:
        return MemorySnapshot(
            total_bytes=_positive_int(total_bytes, "total_bytes"),
            available_bytes=_positive_int(available_bytes, "available_bytes"),
            platform=normalized_platform,
            source="override",
        )

    total = total_bytes
    available = available_bytes
    source = "sysconf"

    if normalized_platform == "linux":
        linux_total, linux_available = _probe_linux_meminfo()
        total = total or linux_total
        available = available or linux_available
        source = "proc_meminfo"

    if normalized_platform == "macos":
        total = total or _probe_macos_total_memory()
        available = available or _probe_macos_available_memory()
        source = "sysctl/sysconf"

    sysconf_total, sysconf_available = _probe_memory_from_sysconf()
    total = total or sysconf_total
    available = available or sysconf_available

    if not total or not available:
        raise MemoryProbeError("memory cannot be probed; provide total_bytes and available_bytes overrides")

    return MemorySnapshot(
        total_bytes=_positive_int(total, "total_bytes"),
        available_bytes=_positive_int(available, "available_bytes"),
        platform=normalized_platform,
        source=source,
    )


def default_peak_multiplier_for_stage(stage: str | None) -> float:
    return DEFAULT_PEAK_MULTIPLIERS.get((stage or "").strip().lower(), 3.0)


def estimate_max_rows(
    memory_snapshot: MemorySnapshot,
    *,
    feature_column_count: int,
    required_non_feature_column_count: int,
    peak_multiplier: float | None = None,
    memory_budget_fraction: float = DEFAULT_MEMORY_BUDGET_FRACTION,
    bytes_per_numeric_value: int = DEFAULT_BYTES_PER_NUMERIC_VALUE,
) -> CapacityEstimate:
    """Estimate safe in-memory row capacity for a feature matrix."""
    if feature_column_count < 0:
        raise ValueError("feature_column_count must be non-negative")
    if required_non_feature_column_count < 0:
        raise ValueError("required_non_feature_column_count must be non-negative")
    if not 0 < memory_budget_fraction <= 1:
        raise ValueError("memory_budget_fraction must be in (0, 1]")
    if peak_multiplier is None:
        peak_multiplier = 3.0
    if peak_multiplier <= 0:
        raise ValueError("peak_multiplier must be positive")
    if bytes_per_numeric_value <= 0:
        raise ValueError("bytes_per_numeric_value must be positive")

    total_columns = feature_column_count + required_non_feature_column_count
    if total_columns <= 0:
        raise ValueError("at least one feature or required non-feature column is required")

    row_width_bytes = total_columns * bytes_per_numeric_value
    memory_budget_bytes = int(memory_snapshot.available_bytes * memory_budget_fraction)
    matrix_budget_bytes = int(memory_budget_bytes / peak_multiplier)
    max_rows = max(0, matrix_budget_bytes // row_width_bytes)

    return CapacityEstimate(
        total_memory_bytes=memory_snapshot.total_bytes,
        available_memory_bytes=memory_snapshot.available_bytes,
        memory_budget_fraction=memory_budget_fraction,
        memory_budget_bytes=memory_budget_bytes,
        peak_multiplier=peak_multiplier,
        matrix_budget_bytes=matrix_budget_bytes,
        feature_column_count=feature_column_count,
        required_non_feature_column_count=required_non_feature_column_count,
        bytes_per_numeric_value=bytes_per_numeric_value,
        row_width_bytes=row_width_bytes,
        max_rows=max_rows,
    )


def choose_uniform_sampling_ratio(
    *,
    total_rows: int,
    max_rows: int,
    min_ratio: float | None = None,
    max_ratio: float = 1.0,
) -> UniformSamplingDecision:
    """Choose a full-table uniform sampling ratio and optional row cap."""
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative")
    if max_rows < 0:
        raise ValueError("max_rows must be non-negative")
    if min_ratio is not None and not 0 <= min_ratio <= 1:
        raise ValueError("min_ratio must be in [0, 1]")
    if not 0 < max_ratio <= 1:
        raise ValueError("max_ratio must be in (0, 1]")

    if total_rows == 0:
        return UniformSamplingDecision(
            total_rows=0,
            max_rows=max_rows,
            sampling_required=False,
            ratio=1.0,
            estimated_rows=0,
            limit=None,
            reason="empty source",
        )

    if max_rows >= total_rows:
        return UniformSamplingDecision(
            total_rows=total_rows,
            max_rows=max_rows,
            sampling_required=False,
            ratio=1.0,
            estimated_rows=total_rows,
            limit=None,
            reason="source rows fit memory capacity",
        )

    raw_ratio = max_rows / total_rows if total_rows else 1.0
    ratio = min(raw_ratio, max_ratio)
    reason = "row count exceeds memory capacity"
    if min_ratio is not None and ratio < min_ratio:
        ratio = min_ratio
        reason = "row count exceeds memory capacity; min_ratio clamp requires limit fallback"

    estimated_rows = int(math.ceil(total_rows * ratio))
    limit = max_rows if estimated_rows > max_rows else None
    return UniformSamplingDecision(
        total_rows=total_rows,
        max_rows=max_rows,
        sampling_required=True,
        ratio=ratio,
        estimated_rows=estimated_rows,
        limit=limit,
        reason=reason,
    )


def build_resource_plan_payload(
    *,
    data_source_mode: str,
    stage: str,
    memory_snapshot: MemorySnapshot,
    total_rows: int,
    feature_column_count: int,
    required_non_feature_column_count: int,
    peak_multiplier: float | None = None,
    memory_budget_fraction: float = DEFAULT_MEMORY_BUDGET_FRACTION,
    bytes_per_numeric_value: int = DEFAULT_BYTES_PER_NUMERIC_VALUE,
    min_sampling_ratio: float | None = None,
    local_file_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable resource plan payload."""
    effective_multiplier = peak_multiplier
    if effective_multiplier is None:
        effective_multiplier = default_peak_multiplier_for_stage(stage)
    estimate = estimate_max_rows(
        memory_snapshot,
        feature_column_count=feature_column_count,
        required_non_feature_column_count=required_non_feature_column_count,
        peak_multiplier=effective_multiplier,
        memory_budget_fraction=memory_budget_fraction,
        bytes_per_numeric_value=bytes_per_numeric_value,
    )
    sampling = choose_uniform_sampling_ratio(
        total_rows=total_rows,
        max_rows=estimate.max_rows,
        min_ratio=min_sampling_ratio,
    )

    payload: dict[str, Any] = {
        "data_source_mode": data_source_mode,
        "stage": stage,
        "memory": asdict(memory_snapshot),
        "capacity": asdict(estimate),
        "sampling": asdict(sampling),
    }
    if local_file_size_bytes is not None:
        payload["local_source"] = {
            "file_size_bytes": local_file_size_bytes,
            "estimated_in_memory_expansion_basis": "row_width_bytes * row_count * peak_multiplier",
        }
    return payload
