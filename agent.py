#!/usr/bin/env python3
"""CLI entrypoint for the business-modeling agent."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jingying_model_agent.cli import main


if __name__ == "__main__":
    main()
