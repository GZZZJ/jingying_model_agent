from risk_model_workbench.resource_usage import ProcessMemoryTracker


def test_process_memory_tracker_reports_observed_multiplier():
    current_values = iter([1000, 1600, 2100])
    peak_values = iter([1200, 1800, 2600])

    tracker = ProcessMemoryTracker(
        stage="feature_refine",
        current_rss_fn=lambda: next(current_values),
        peak_rss_fn=lambda: next(peak_values),
    )

    tracker.record("load_sample")
    tracker.record("d03_done")
    summary = tracker.summary(matrix_bytes=700, row_count=10, column_count=7, feature_count=5, configured_peak_multiplier=4.0)

    assert summary["max_current_rss_delta_bytes"] == 1100
    assert summary["max_peak_rss_delta_bytes"] == 1400
    assert summary["observed_peak_multiplier"] == 2.0
    assert summary["configured_peak_multiplier"] == 4.0
    assert [item["label"] for item in summary["checkpoints"]] == ["load_sample", "d03_done"]
