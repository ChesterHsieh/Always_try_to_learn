from pipeline.lineage import build_openlineage_event


def test_lineage_event_has_required_fields() -> None:
    event = build_openlineage_event("run", "job", "ns", "src", "dst")
    assert event["run"]["runId"] == "run"
    assert event["job"]["namespace"] == "ns"
    assert event["inputs"][0]["name"] == "src"
    assert event["outputs"][0]["name"] == "dst"
