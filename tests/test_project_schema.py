from jingying_model_agent.paths import project_config_path, resolve_project_path


def test_project_config_path_prefers_yml():
    project = resolve_project_path("projects/2026-05-fujie-gcard-v1")
    assert project_config_path(project).name == "project.yml"
