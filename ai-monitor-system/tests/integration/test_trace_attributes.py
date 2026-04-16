from pipeline.tracing import build_trace_attributes


def test_trace_attributes_have_run_id() -> None:
    attrs = build_trace_attributes("r1", "pipe", "ns", "ok")
    assert attrs["run_id"] == "r1"
    assert attrs["trace_id"]
    assert attrs["k8s_namespace"] == "ns"
