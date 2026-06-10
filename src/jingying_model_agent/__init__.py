"""Compatibility package for legacy ``jingying_model_agent`` imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_new_package = import_module("risk_model_workbench")

__path__ = _new_package.__path__
__version__ = getattr(_new_package, "__version__", "0.0.0")
__all__ = list(getattr(_new_package, "__all__", []))


def __getattr__(name: str) -> Any:
    return getattr(_new_package, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_new_package)))
