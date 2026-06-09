from pathlib import Path

from jingying_model_agent.config import load_yaml


def test_load_yaml_project_config():
    data = load_yaml(Path("projects/2026-05-fujie-gcard-v1/project.yml"))
    assert data["data"]["target_column"] == "ftr_30d_ord_flag"
