"""Compatibility package for legacy imports.

New code should import from ``risk_model_workbench``.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import risk_model_workbench as _new_package

__path__ = _new_package.__path__
__version__ = getattr(_new_package, "__version__", "0.0.0")
__all__ = list(getattr(_new_package, "__all__", []))


def __getattr__(name: str) -> Any:
    return getattr(_new_package, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_new_package)))
