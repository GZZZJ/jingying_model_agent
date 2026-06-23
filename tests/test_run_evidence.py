from pathlib import Path

import yaml

from risk_model_workbench.cli import main
from risk_model_workbench.run_evidence import load_run_evidence


def test_load_run_evidence_groups_manifest_and_contracts(tmp_path):
    project = _make_project(tmp_path)
    assert main(["run", "init", "--project", str(project), "--workflow", "full_modeling", "--run-id", "run1"]) == 0

    evidence = load_run_evidence(project, "run1")

    assert evidence.run_id == "run1"
    assert evidence.workflow == "full_modeling"
    assert evidence.run_state["stages"]["validate_config"]["status"] == "done"
    assert "validate_config" in evidence.manifest_by_stage
    assert "sample_check" in evidence.stage_contracts
    assert evidence.contract_source.endswith("workflows/full_modeling.yml")


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "demo_project"
    for directory in ["configs", "queries", "runs", "reports", "docs"]:
        (project / directory).mkdir(parents=True, exist_ok=True)
    (project / "project.yml").write_text(
        yaml.safe_dump(
            {
                "project": {"name": "demo_project", "display_name": "Demo Project"},
                "data": {
                    "source_table": "demo.sample",
                    "id_columns": ["uid"],
                    "target_column": "label",
                    "time_column": "event_time",
                    "period_column": "ds",
                },
                "segments": [{"name": "all", "display_name": "All", "filter": None}],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return project
