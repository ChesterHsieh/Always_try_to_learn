"""Integration tests: failure alert + metric label verification.

For each KNOWN_CATEGORY:
  - Sets INJECT_FAILURE=<category>
  - Runs run_pipeline() with mocked Spark
  - Asserts lifecycle payload failure_category equals expected category
  - Asserts pipeline_failures_total counter incremented with correct label
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import pipeline.job as job_module
from pipeline.failure_classifier import KNOWN_CATEGORIES
from telemetry.metrics import pipeline_failures_total
from utils.io_adapter import write_local_file


@pytest.fixture(autouse=True)
def skip_metrics_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METRICS_SKIP_SERVER", "1")


def _failures_value(*, failure_category: str, pipeline_name: str) -> float:
    return pipeline_failures_total.labels(
        failure_category=failure_category, pipeline_name=pipeline_name
    )._value.get()  # type: ignore[attr-defined]


def _make_spark_mock(tmp_output_path: str) -> MagicMock:
    def fake_write_text(path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "part-00000-fake.txt").write_text("HELLO\n")

    df_mock = MagicMock()
    df_mock.write.mode.return_value.text = fake_write_text
    df_mock.select.return_value = df_mock

    spark_mock = MagicMock()
    spark_mock.read.text.return_value = df_mock
    return spark_mock


# ---------------------------------------------------------------------------
# Task 6.1: All 9 KNOWN_CATEGORIES — lifecycle payload + metric label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", sorted(KNOWN_CATEGORIES))
def test_failure_lifecycle_payload_and_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    category: str,
) -> None:
    """For each KNOWN_CATEGORY: payload.failure_category correct + metric counter increments."""
    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"
    write_local_file(str(src), "hello\n")
    monkeypatch.setenv("INJECT_FAILURE", category)

    pipeline_name = job_module.PIPELINE_NAME
    before = _failures_value(failure_category=category, pipeline_name=pipeline_name)

    spark_mock = _make_spark_mock(str(dst))

    with (
        patch.object(job_module, "create_spark_session", return_value=spark_mock),
        patch("pipeline.job.F") as mock_f,
    ):
        mock_f.upper.return_value = MagicMock()
        mock_f.col.return_value = MagicMock()
        payload = job_module.run_pipeline(str(src), str(dst))

    # Lifecycle payload assertions
    assert payload["status"] == "failed", f"Expected failed for {category!r}, got {payload}"
    assert payload["failure_category"] == category, (
        f"Expected failure_category={category!r}, got {payload.get('failure_category')!r}"
    )

    # Metric label assertion: pipeline_failures_total{failure_category=<category>} incremented
    after = _failures_value(failure_category=category, pipeline_name=pipeline_name)
    assert after - before >= 1.0, (
        f"pipeline_failures_total{{failure_category={category!r}}} did not increment; "
        f"before={before} after={after}"
    )


@pytest.mark.parametrize("category", sorted(KNOWN_CATEGORIES))
def test_failure_run_total_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    category: str,
) -> None:
    """pipeline_run_total{status='failed'} must increment for every failure category."""
    from telemetry.metrics import pipeline_run_total

    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"
    write_local_file(str(src), "hello\n")
    monkeypatch.setenv("INJECT_FAILURE", category)

    pipeline_name = job_module.PIPELINE_NAME
    before = pipeline_run_total.labels(status="failed", pipeline_name=pipeline_name)._value.get()  # type: ignore[attr-defined]

    spark_mock = _make_spark_mock(str(dst))

    with (
        patch.object(job_module, "create_spark_session", return_value=spark_mock),
        patch("pipeline.job.F") as mock_f,
    ):
        mock_f.upper.return_value = MagicMock()
        mock_f.col.return_value = MagicMock()
        job_module.run_pipeline(str(src), str(dst))

    after = pipeline_run_total.labels(status="failed", pipeline_name=pipeline_name)._value.get()  # type: ignore[attr-defined]
    assert after - before >= 1.0, (
        f"pipeline_run_total{{status='failed'}} did not increment for {category!r}"
    )


