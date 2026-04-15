from pipeline.telemetry import lifecycle_metric_payload


def test_run_lifecycle_payload_has_required_fields() -> None:
    payload = lifecycle_metric_payload("run-1", "running")
    assert payload["run_id"] == "run-1"
    assert payload["status"] == "running"
    assert "timestamp" in payload
