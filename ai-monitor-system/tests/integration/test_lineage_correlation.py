from pipeline.lineage import build_openlineage_event


def test_lineage_contains_run_id() -> None:
    event = build_openlineage_event("r1", "job", "ns", "in", "out")
    assert event["run"]["runId"] == "r1"
