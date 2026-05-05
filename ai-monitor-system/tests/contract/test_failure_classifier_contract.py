"""Contract tests: KNOWN_CATEGORIES stability and classify_failure determinism (Tasks 2.2, 2.4)."""

import socket

import pytest

from pipeline.failure_classifier import KNOWN_CATEGORIES, classify_failure
from pipeline.failure_injection import STAGE_FOR_CATEGORY, maybe_inject


@pytest.mark.parametrize(
    "exc,expected_category",
    [
        (FileNotFoundError("no file"), "input_not_found"),
        (IsADirectoryError("dir"), "invalid_path"),
        (PermissionError("denied"), "permission_denied"),
        (TimeoutError("timeout"), "timeout"),
        (socket.timeout("sock timeout"), "timeout"),
        (RuntimeError("spark task failed TaskFailed"), "spark_task_failed"),
        (RuntimeError("SparkException driver error"), "spark_driver_error"),
        (RuntimeError("openlineage error"), "lineage_emission_failed"),
        (RuntimeError("otel collector issue"), "telemetry_unavailable"),
        (ValueError("unexpected"), "runtime_error"),
    ],
)
def test_classify_failure_returns_known_category(exc, expected_category):
    result = classify_failure(exc)
    assert result == expected_category
    assert result in KNOWN_CATEGORIES


def test_classify_failure_is_deterministic():
    err = FileNotFoundError("same error")
    first = classify_failure(err)
    second = classify_failure(err)
    assert first == second


def test_known_categories_is_frozenset():
    assert isinstance(KNOWN_CATEGORIES, frozenset)


def test_known_categories_contains_all_9():
    expected = {
        "input_not_found",
        "invalid_path",
        "permission_denied",
        "spark_task_failed",
        "spark_driver_error",
        "lineage_emission_failed",
        "telemetry_unavailable",
        "timeout",
        "runtime_error",
    }
    assert expected == KNOWN_CATEGORIES


def test_classify_failure_never_returns_empty():
    for exc_type in [
        FileNotFoundError,
        IsADirectoryError,
        PermissionError,
        TimeoutError,
        RuntimeError,
        ValueError,
        Exception,
    ]:
        result = classify_failure(exc_type("test"))
        assert result, f"classify_failure returned empty string for {exc_type}"
        assert result in KNOWN_CATEGORIES


def test_classify_failure_with_requests_connection_error():
    try:
        import requests.exceptions as req_exc

        err = req_exc.ConnectionError("Failed to connect to openlineage")
        result = classify_failure(err)
        assert result in KNOWN_CATEGORIES
    except ImportError:
        pytest.skip("requests not installed")


def test_classify_failure_with_requests_timeout():
    try:
        import requests.exceptions as req_exc

        err = req_exc.Timeout("Connection timed out")
        result = classify_failure(err)
        assert result == "timeout"
    except ImportError:
        pytest.skip("requests not installed")


# ---------------------------------------------------------------------------
# Task 2.4: Degradation detection — injected exceptions must round-trip
# through classify_failure back to their source category (never fall back
# to runtime_error for classifiable exceptions).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category",
    sorted(KNOWN_CATEGORIES),
)
def test_injected_exception_classified_correctly(category: str) -> None:
    """Each KNOWN_CATEGORY injection must round-trip: inject → raise → classify → same category."""
    assigned_stage = STAGE_FOR_CATEGORY[category]
    with pytest.raises(BaseException) as exc_info:
        maybe_inject(category, stage=assigned_stage)
    result = classify_failure(exc_info.value)
    assert result == category, (
        f"injection {category!r} produced exception classified as {result!r}; "
        f"expected {category!r} — classifier may have regressed to 'runtime_error'"
    )


def test_spark_task_failed_not_degraded_to_runtime_error() -> None:
    """Py4JJavaError with TaskFailed in message must be spark_task_failed, not runtime_error.

    This is the canonical degradation regression: if the Py4J type-name check
    is accidentally removed, TaskFailed messages fall through to runtime_error.
    """
    from pipeline.failure_injection import Py4JJavaError

    err = Py4JJavaError("org.apache.spark TaskFailed in stage 0.0")
    result = classify_failure(err)
    assert result != "runtime_error", (
        "spark_task_failed Py4JJavaError was downgraded to runtime_error — "
        "classifier Py4J dispatch may be broken"
    )
    assert result == "spark_task_failed"


def test_spark_driver_error_not_degraded_to_runtime_error() -> None:
    """Py4JJavaError with SparkException must be spark_driver_error, not runtime_error."""
    from pipeline.failure_injection import Py4JJavaError

    err = Py4JJavaError("org.apache.spark.SparkException at driver")
    result = classify_failure(err)
    assert result != "runtime_error", (
        "spark_driver_error Py4JJavaError was downgraded to runtime_error — "
        "classifier Py4J dispatch may be broken"
    )
    assert result == "spark_driver_error"


def test_schema_mismatch_injection_maps_to_spark_driver_error() -> None:
    """schema_mismatch injection must classify as spark_driver_error (AnalysisException path)."""
    with pytest.raises(BaseException) as exc_info:
        maybe_inject("schema_mismatch", stage="during_spark")
    result = classify_failure(exc_info.value)
    assert result == "spark_driver_error", (
        f"schema_mismatch injection produced {result!r}; expected 'spark_driver_error'"
    )


def test_pyspark_native_analysis_exception_maps_to_spark_driver_error() -> None:
    """PySpark-native AnalysisException (e.g. UNRESOLVED_COLUMN from schema drift)
    must classify as spark_driver_error.

    This guards the schema-drift scenario where a real Spark plan analyzer
    error surfaces as `pyspark.errors.exceptions.captured.AnalysisException`
    rather than Py4JJavaError. Without this, the failure regresses to
    runtime_error and the failure-category metric label is wrong.
    """

    class AnalysisException(Exception):
        """Stand-in for `pyspark.errors.exceptions.captured.AnalysisException`.

        Identified by type name only — the classifier never imports PySpark.
        """

    err = AnalysisException(
        "[UNRESOLVED_COLUMN.WITH_SUGGESTION] A column or function parameter "
        "with name `nonexistent_column` cannot be resolved."
    )
    result = classify_failure(err)
    assert result == "spark_driver_error", (
        f"AnalysisException was classified as {result!r}; "
        "schema-drift scenario requires spark_driver_error"
    )
