"""Integration tests: failure alert + metric label verification (Tasks 6.1, 6.2).

For each KNOWN_CATEGORY:
  - Sets INJECT_FAILURE=<category>
  - Runs run_pipeline() with mocked Spark
  - Asserts lifecycle payload failure_category equals expected category
  - Asserts pipeline_failures_total counter incremented with correct label

For schema-mismatch (Task 6.2):
  - Injects schema_mismatch
  - Asserts classification = spark_driver_error (three detection paths in lifecycle)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import pipeline.job as job_module
from pipeline.failure_classifier import KNOWN_CATEGORIES
from pipeline.io_adapter import write_local_file
from pipeline.metrics import pipeline_failures_total


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
    from pipeline.metrics import pipeline_run_total

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


# ---------------------------------------------------------------------------
# Task 6.2: schema-mismatch E2E — three detection paths
# ---------------------------------------------------------------------------


def test_schema_mismatch_classified_as_spark_driver_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """schema_mismatch injection must produce failure_category=spark_driver_error in payload."""
    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"
    write_local_file(str(src), "hello\n")
    monkeypatch.setenv("INJECT_FAILURE", "schema_mismatch")

    spark_mock = _make_spark_mock(str(dst))

    with (
        patch.object(job_module, "create_spark_session", return_value=spark_mock),
        patch("pipeline.job.F") as mock_f,
    ):
        mock_f.upper.return_value = MagicMock()
        mock_f.col.return_value = MagicMock()
        payload = job_module.run_pipeline(str(src), str(dst))

    assert payload["status"] == "failed"
    assert payload["failure_category"] == "spark_driver_error", (
        f"schema_mismatch must map to spark_driver_error, got {payload.get('failure_category')!r}"
    )


def test_schema_mismatch_increments_spark_driver_error_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pipeline_failures_total spark_driver_error increments on schema_mismatch injection."""
    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"
    write_local_file(str(src), "hello\n")
    monkeypatch.setenv("INJECT_FAILURE", "schema_mismatch")

    pipeline_name = job_module.PIPELINE_NAME
    before = _failures_value(failure_category="spark_driver_error", pipeline_name=pipeline_name)

    spark_mock = _make_spark_mock(str(dst))

    with (
        patch.object(job_module, "create_spark_session", return_value=spark_mock),
        patch("pipeline.job.F") as mock_f,
    ):
        mock_f.upper.return_value = MagicMock()
        mock_f.col.return_value = MagicMock()
        job_module.run_pipeline(str(src), str(dst))

    after = _failures_value(failure_category="spark_driver_error", pipeline_name=pipeline_name)
    assert after - before >= 1.0, (
        "pipeline_failures_total{failure_category='spark_driver_error'} did not increment "
        "for schema_mismatch injection"
    )


def test_schema_mismatch_failure_message_contains_analysis_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure_message in the payload must reference AnalysisException for schema_mismatch."""
    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"
    write_local_file(str(src), "hello\n")
    monkeypatch.setenv("INJECT_FAILURE", "schema_mismatch")

    spark_mock = _make_spark_mock(str(dst))

    with (
        patch.object(job_module, "create_spark_session", return_value=spark_mock),
        patch("pipeline.job.F") as mock_f,
    ):
        mock_f.upper.return_value = MagicMock()
        mock_f.col.return_value = MagicMock()
        payload = job_module.run_pipeline(str(src), str(dst))

    failure_message = payload.get("failure_message", "")
    has_analysis_exc = "AnalysisException" in failure_message
    has_schema_mismatch = "schema mismatch" in failure_message.lower()
    assert has_analysis_exc or has_schema_mismatch, (
        f"schema_mismatch failure_message should reference AnalysisException; "
        f"got: {failure_message!r}"
    )
