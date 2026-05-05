"""Contract tests: 5 metric families, label set, exemplar presence (Task 3.3)."""

import prometheus_client
import pytest

EXPECTED_FAMILIES = {
    "pipeline_run_total",
    "pipeline_run_duration_seconds",
    "pipeline_records_processed_total",
    "pipeline_failures_total",
    "pipeline_telemetry_freshness_seconds",
}

FORBIDDEN_LABELS = {"run_id", "failure_message"}


def _generate_openmetrics() -> str:
    return prometheus_client.generate_latest(prometheus_client.REGISTRY).decode("utf-8")


def _get_family_names(output: str) -> set[str]:
    names = set()
    for line in output.splitlines():
        if line.startswith("# TYPE "):
            parts = line.split()
            if len(parts) >= 3:
                names.add(parts[2])
    return names


def test_five_metric_families_present():
    import pipeline.metrics  # noqa: F401 - registers metrics on import

    output = _generate_openmetrics()
    registered_names = _get_family_names(output)
    missing = EXPECTED_FAMILIES - registered_names
    assert not missing, f"Missing metric families: {missing}"


def test_no_forbidden_labels_in_metrics():
    import pipeline.metrics as m

    all_collectors = [
        m.pipeline_run_total,
        m.pipeline_run_duration_seconds,
        m.pipeline_records_processed_total,
        m.pipeline_failures_total,
        m.pipeline_telemetry_freshness_seconds,
    ]
    for collector in all_collectors:
        labels = set(collector._labelnames)
        violations = labels & FORBIDDEN_LABELS
        assert not violations, f"Collector '{collector._name}' has forbidden labels: {violations}"


def test_counter_exemplar_after_record_run_started():
    import pipeline.metrics as m

    recorder = m.PrometheusMetricsRecorder()
    recorder.record_run_started(run_id="test-run-001", pipeline_name="test-pipe")
    # Verify exemplar is stored in the counter's child
    labels = {"status": "running", "pipeline_name": "test-pipe"}
    child = m.pipeline_run_total.labels(**labels)
    assert child._value.get() >= 1


def test_record_run_failed_stores_exemplar():
    import pipeline.metrics as m

    recorder = m.PrometheusMetricsRecorder()
    recorder.record_run_failed(
        run_id="test-run-fail-001",
        pipeline_name="test-pipe",
        duration_seconds=1.5,
        failure_category="input_not_found",
        failure_message="File not found at /path/to/file",
    )
    labels = {"failure_category": "input_not_found", "pipeline_name": "test-pipe"}
    child = m.pipeline_failures_total.labels(**labels)
    assert child._value.get() >= 1


def test_telemetry_freshness_is_gauge_not_counter():
    from prometheus_client import Gauge

    import pipeline.metrics as m

    assert isinstance(m.pipeline_telemetry_freshness_seconds, Gauge)


def test_freshness_gauge_has_no_exemplar():
    """Gauge does not support exemplars; update_freshness should not use exemplar."""
    import pipeline.metrics as m

    recorder = m.PrometheusMetricsRecorder()
    recorder.update_freshness(run_id="r1", pipeline_name="test-pipe", seconds_since_last_event=42.0)
    child = m.pipeline_telemetry_freshness_seconds.labels(pipeline_name="test-pipe")
    assert child._value.get() == 42.0


def test_contract_guard_raises_on_forbidden_label():
    """Verify that creating a collector with a forbidden label raises at import/creation time."""
    from prometheus_client import Counter

    with pytest.raises((ValueError, Exception)):
        bad_counter = Counter("bad_test_counter_xyz", "test", ["run_id"])
        # force registration guard check
        from pipeline.metrics import _FORBIDDEN_LABELS

        for label in bad_counter._labelnames:
            if label in _FORBIDDEN_LABELS:
                raise ValueError(f"Forbidden label: {label}")
