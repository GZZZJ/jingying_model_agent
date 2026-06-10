import tomllib
from pathlib import Path


def test_new_and_legacy_imports_resolve():
    import risk_model_workbench
    from risk_model_workbench.cli import main as rmw_main

    import jingying_model_agent
    from jingying_model_agent.cli import main as old_main

    import jingying_agent

    assert risk_model_workbench.__version__
    assert rmw_main is not None
    assert old_main is not None
    assert jingying_model_agent is not None
    assert jingying_agent is not None


def test_console_script_declarations_keep_new_and_legacy_entries():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert scripts["rmw"] == "risk_model_workbench.cli:main"
    assert scripts["jm"] == "risk_model_workbench.cli:main"
    assert scripts["jingying-agent"] == "risk_model_workbench.cli:main"
