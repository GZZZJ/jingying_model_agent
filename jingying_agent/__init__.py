"""Utilities for local business-modeling agent projects."""

__version__ = "0.1.0"
"""Compatibility package for legacy imports.

New code should import from ``jingying_model_agent``.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jingying_model_agent import *  # noqa: F401,F403
