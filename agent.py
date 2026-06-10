#!/usr/bin/env python3
"""CLI entrypoint for the risk-modeling workbench."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from risk_model_workbench.cli import main


if __name__ == "__main__":
    main()
