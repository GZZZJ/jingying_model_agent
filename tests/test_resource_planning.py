from risk_model_workbench.resource_planning import (
    MemorySnapshot,
    build_resource_plan_payload,
    choose_uniform_sampling_ratio,
    estimate_max_rows,
    probe_memory,
)


def test_estimate_max_rows_uses_available_memory_and_peak_multiplier():
    snapshot = MemorySnapshot(
        total_bytes=32 * 1024**3,
        available_bytes=16 * 1024**3,
        platform="macos",
        source="test",
    )

    estimate = estimate_max_rows(
        snapshot,
        feature_column_count=96,
        required_non_feature_column_count=20,
        peak_multiplier=4.0,
    )

    assert estimate.row_width_bytes == 116 * 8
    assert estimate.memory_budget_bytes == int(16 * 1024**3 * 0.8)
    assert estimate.matrix_budget_bytes == int((16 * 1024**3 * 0.8) / 4.0)
    assert estimate.max_rows == estimate.matrix_budget_bytes // estimate.row_width_bytes


def test_large_feature_count_forces_small_capacity():
    snapshot = MemorySnapshot(
        total_bytes=16 * 1024**3,
        available_bytes=4 * 1024**3,
        platform="linux",
        source="test",
    )

    estimate = estimate_max_rows(
        snapshot,
        feature_column_count=15028,
        required_non_feature_column_count=12,
        peak_multiplier=3.0,
    )

    assert estimate.row_width_bytes == 15040 * 8
    assert estimate.max_rows < 10_000


def test_choose_uniform_sampling_ratio_and_limit_fallback():
    decision = choose_uniform_sampling_ratio(total_rows=10_000_000, max_rows=1_000_000)

    assert decision.sampling_required is True
    assert decision.ratio == 0.1
    assert decision.estimated_rows == 1_000_000
    assert decision.limit is None

    clamped = choose_uniform_sampling_ratio(
        total_rows=10_000_000,
        max_rows=100_000,
        min_ratio=0.05,
    )

    assert clamped.ratio == 0.05
    assert clamped.estimated_rows == 500_000
    assert clamped.limit == 100_000
    assert "min_ratio" in clamped.reason


def test_build_resource_plan_payload_preserves_formula_details_and_local_file_context():
    snapshot = MemorySnapshot(
        total_bytes=32 * 1024**3,
        available_bytes=16 * 1024**3,
        platform="macos",
        source="test",
    )

    payload = build_resource_plan_payload(
        data_source_mode="local_feather",
        stage="feature_refine",
        memory_snapshot=snapshot,
        total_rows=10_000_000,
        feature_column_count=96,
        required_non_feature_column_count=20,
        peak_multiplier=4.0,
        local_file_size_bytes=123456,
    )

    assert payload["data_source_mode"] == "local_feather"
    assert payload["stage"] == "feature_refine"
    assert payload["memory"]["available_bytes"] == 16 * 1024**3
    assert payload["capacity"]["feature_column_count"] == 96
    assert payload["capacity"]["required_non_feature_column_count"] == 20
    assert payload["capacity"]["formula"] == (
        "floor((available_memory_bytes * memory_budget_fraction / peak_multiplier) "
        "/ ((feature_column_count + required_non_feature_column_count) * bytes_per_numeric_value))"
    )
    assert payload["sampling"]["ratio"] < 1.0
    assert payload["local_source"]["file_size_bytes"] == 123456


def test_probe_memory_uses_macos_vm_stat_available_memory(monkeypatch):
    def fake_check_output(cmd, text=True, stderr=None):
        if cmd == ["sysctl", "-n", "hw.memsize"]:
            return str(32 * 1024**3)
        if cmd == ["vm_stat"]:
            return "\n".join(
                [
                    "Mach Virtual Memory Statistics: (page size of 16384 bytes)",
                    "Pages free:                               100.",
                    "Pages inactive:                           200.",
                    "Pages speculative:                         50.",
                ]
            )
        raise AssertionError(cmd)

    monkeypatch.setattr("risk_model_workbench.resource_planning.subprocess.check_output", fake_check_output)
    monkeypatch.setattr("risk_model_workbench.resource_planning._probe_memory_from_sysconf", lambda: (None, None))

    snapshot = probe_memory(platform_name="Darwin")

    assert snapshot.total_bytes == 32 * 1024**3
    assert snapshot.available_bytes == 350 * 16384
    assert snapshot.platform == "macos"
