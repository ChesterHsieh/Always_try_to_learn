from pipeline.telemetry import lifecycle_metric_payload


def test_run_lifecycle_payload_has_required_fields() -> None:
    payload = lifecycle_metric_payload("run-1", "running")
    assert payload["run_id"] == "run-1"
    assert payload["status"] == "running"
    assert payload["pipeline_name"] == "ai_monitor_simple_pipeline"
    assert "timestamp" in payload


def test_failed_lifecycle_payload_has_failure_context() -> None:
    payload = lifecycle_metric_payload(
        "run-2",
        "failed",
        failure_category="runtime_error",
        failure_message="boom",
    )
    assert payload["status"] == "failed"
    assert payload["failure_category"] == "runtime_error"
    assert payload["failure_message"] == "boom"
