from jingying_model_agent.state import create_run_state, mark_stage_done, save_run_state


def test_create_run_state_contains_stages():
    state = create_run_state("projects/example", run_id="run1", workflow="full_modeling")
    assert state["run_id"] == "run1"
    assert state["stages"]["sample_check"]["status"] == "pending"


def test_imported_run_keeps_imported_status_after_stage_done(tmp_path):
    run_path = tmp_path / "run"
    state = create_run_state("projects/example", run_id="run1", workflow="imported_gcard_main_lgbm", status="imported")
    save_run_state(run_path, state)

    updated = mark_stage_done(run_path, "report")

    assert updated["status"] == "imported"
    assert updated["stages"]["report"]["status"] == "done"
