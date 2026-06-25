"""Runtime memory usage evidence for resource-aware modeling stages."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


def _ru_maxrss_to_bytes(value: int) -> int:
    if platform.system().lower().startswith("darwin"):
        return int(value)
    return int(value) * 1024


def peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    try:
        return _ru_maxrss_to_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return None


def _current_rss_from_proc() -> int | None:
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    try:
        resident_pages = int(statm.read_text(encoding="utf-8").split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _current_rss_from_ps() -> int | None:
    try:
        output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return int(output.strip()) * 1024
    except ValueError:
        return None


def current_rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        psutil = None
    if psutil is not None:
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except (OSError, ValueError):
            pass
    return _current_rss_from_proc() or _current_rss_from_ps()


def dataframe_memory_bytes(dataframe: Any) -> int | None:
    if dataframe is None or not hasattr(dataframe, "memory_usage"):
        return None
    try:
        usage = dataframe.memory_usage(index=True, deep=True)
        return int(usage.sum()) if hasattr(usage, "sum") else int(usage)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class MemoryCheckpoint:
    label: str
    created_at: str
    current_rss_bytes: int | None
    peak_rss_bytes: int | None
    current_rss_delta_bytes: int | None
    peak_rss_delta_bytes: int | None
    metrics: dict[str, Any]


class ProcessMemoryTracker:
    """Collect process RSS checkpoints for one stage run."""

    def __init__(
        self,
        *,
        stage: str,
        current_rss_fn: Callable[[], int | None] = current_rss_bytes,
        peak_rss_fn: Callable[[], int | None] = peak_rss_bytes,
    ) -> None:
        self.stage = stage
        self._current_rss_fn = current_rss_fn
        self._peak_rss_fn = peak_rss_fn
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.baseline_current_rss_bytes = current_rss_fn()
        self.baseline_peak_rss_bytes = peak_rss_fn()
        self.checkpoints: list[MemoryCheckpoint] = []

    @staticmethod
    def _delta(value: int | None, baseline: int | None) -> int | None:
        if value is None or baseline is None:
            return None
        return max(0, int(value) - int(baseline))

    def record(self, label: str, **metrics: Any) -> MemoryCheckpoint:
        current = self._current_rss_fn()
        peak = self._peak_rss_fn()
        checkpoint = MemoryCheckpoint(
            label=label,
            created_at=datetime.now().isoformat(timespec="seconds"),
            current_rss_bytes=current,
            peak_rss_bytes=peak,
            current_rss_delta_bytes=self._delta(current, self.baseline_current_rss_bytes),
            peak_rss_delta_bytes=self._delta(peak, self.baseline_peak_rss_bytes),
            metrics=metrics,
        )
        self.checkpoints.append(checkpoint)
        return checkpoint

    def summary(
        self,
        *,
        matrix_bytes: int | None = None,
        row_count: int | None = None,
        column_count: int | None = None,
        feature_count: int | None = None,
        configured_peak_multiplier: float | None = None,
    ) -> dict[str, Any]:
        current_deltas = [item.current_rss_delta_bytes for item in self.checkpoints if item.current_rss_delta_bytes is not None]
        peak_deltas = [item.peak_rss_delta_bytes for item in self.checkpoints if item.peak_rss_delta_bytes is not None]
        max_current_delta = max(current_deltas) if current_deltas else None
        max_peak_delta = max(peak_deltas) if peak_deltas else None
        observed_multiplier = None
        if matrix_bytes and matrix_bytes > 0 and max_peak_delta is not None:
            observed_multiplier = max_peak_delta / matrix_bytes

        return {
            "stage": self.stage,
            "started_at": self.started_at,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "baseline_current_rss_bytes": self.baseline_current_rss_bytes,
            "baseline_peak_rss_bytes": self.baseline_peak_rss_bytes,
            "max_current_rss_delta_bytes": max_current_delta,
            "max_peak_rss_delta_bytes": max_peak_delta,
            "matrix_memory_bytes": matrix_bytes,
            "row_count": row_count,
            "column_count": column_count,
            "feature_count": feature_count,
            "configured_peak_multiplier": configured_peak_multiplier,
            "observed_peak_multiplier": observed_multiplier,
            "multiplier_basis": "max_peak_rss_delta_bytes / pandas_dataframe_memory_usage_deep_bytes",
            "checkpoints": [
                {
                    "label": item.label,
                    "created_at": item.created_at,
                    "current_rss_bytes": item.current_rss_bytes,
                    "peak_rss_bytes": item.peak_rss_bytes,
                    "current_rss_delta_bytes": item.current_rss_delta_bytes,
                    "peak_rss_delta_bytes": item.peak_rss_delta_bytes,
                    "metrics": item.metrics,
                }
                for item in self.checkpoints
            ],
        }
