from pipeline.lineage import build_openlineage_event
from pipeline.telemetry import build_correlation_attributes


def test_lineage_contains_run_id() -> None:
    event = build_openlineage_event("r1", "job", "ns", "in", "out")
    assert event["run"]["runId"] == "r1"


def test_lineage_and_telemetry_share_same_run_id() -> None:
    run_id = "shared-run"
    event = build_openlineage_event(run_id, "job", "ns", "in", "out", trace_id="trace-1")
    attrs = build_correlation_attributes(run_id=run_id, trace_id="trace-1", lineage_run_id=run_id)
    assert event["run"]["runId"] == attrs["run_id"]
    assert attrs["lineage_run_id"] == run_id
