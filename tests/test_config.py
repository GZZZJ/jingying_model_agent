from pathlib import Path

from risk_model_workbench.config import load_yaml


def test_load_yaml_project_config():
    data = load_yaml(Path("projects/2026-05-fujie-gcard-v1/project.yml"))
    assert data["data"]["target_column"] == "ftr_30d_ord_flag"
