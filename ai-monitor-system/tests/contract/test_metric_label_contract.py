"""Contract tests: metric label contract for failure detection (Task 1.2).

Anchors the design decision to keep `pipeline_run_duration_seconds` and
`pipeline_run_total` label sets stable, and to route failure-category
information through `pipeline_failures_total` only.
"""

from __future__ import annotations

import pytest

from pipeline.failure_classifier import KNOWN_CATEGORIES
from pipeline.metrics import (
    PrometheusMetricsRecorder,
    pipeline_failures_total,
    pipeline_run_duration_seconds,
    pipeline_run_total,
)


def _failures_value(*, failure_category: str, pipeline_name: str) -> float:
    return pipeline_failures_total.labels(
        failure_category=failure_category, pipeline_name=pipeline_name
    )._value.get()  # type: ignore[attr-defined]


def test_run_metrics_label_set_stable() -> None:
    assert tuple(pipeline_run_total._labelnames) == ("status", "pipeline_name")
    assert tuple(pipeline_run_duration_seconds._labelnames) == ("status", "pipeline_name")


def test_pipeline_failures_total_label_set() -> None:
    assert tuple(pipeline_failures_total._labelnames) == (
        "failure_category",
        "pipeline_name",
    )


def test_record_run_failed_increments_failures_total() -> None:
    recorder = PrometheusMetricsRecorder()
    pipeline_name = "test-1.2-known-category"
    before = _failures_value(failure_category="input_not_found", pipeline_name=pipeline_name)
    recorder.record_run_failed(
        run_id="run-1",
        pipeline_name=pipeline_name,
        duration_seconds=0.1,
        failure_category="input_not_found",
        failure_message="missing input",
    )
    after = _failures_value(failure_category="input_not_found", pipeline_name=pipeline_name)
    assert after - before >= 1.0


def test_record_run_failed_rejects_unknown_category() -> None:
    recorder = PrometheusMetricsRecorder()
    with pytest.raises(ValueError, match="failure_category"):
        recorder.record_run_failed(
            run_id="run-2",
            pipeline_name="test-1.2-bad",
            duration_seconds=0.1,
            failure_category="not_a_real_category",
            failure_message="boom",
        )


@pytest.mark.parametrize("category", sorted(KNOWN_CATEGORIES))
def test_record_run_failed_accepts_all_known_categories(category: str) -> None:
    recorder = PrometheusMetricsRecorder()
    pipeline_name = f"test-1.2-{category}"
    before = _failures_value(failure_category=category, pipeline_name=pipeline_name)
    recorder.record_run_failed(
        run_id=f"run-{category}",
        pipeline_name=pipeline_name,
        duration_seconds=0.1,
        failure_category=category,
        failure_message="x",
    )
    after = _failures_value(failure_category=category, pipeline_name=pipeline_name)
    assert after - before >= 1.0
