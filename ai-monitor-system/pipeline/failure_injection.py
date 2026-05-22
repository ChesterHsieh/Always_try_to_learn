"""Failure injection harness for monitor coverage scenarios.

This module is *only* exercised when the pipeline is run with
`INJECT_FAILURE=<category>` set — production helm values never expose this
env var. The single purpose is to drive the monitoring system through
every documented `failure_category`, so the test harness can verify the
pipeline failure is detected, classified, and alerted on.

Each category is bound to one stage in `STAGE_FOR_CATEGORY`. Calling
`maybe_inject` from any other stage is a no-op, which lets the pipeline
plant injection points at all three boundaries (pre_spark, during_spark,
post_spark) without worrying about a category firing twice.
"""

from __future__ import annotations

import socket
from typing import Final, Literal, Mapping

from pipeline.failure_classifier import KNOWN_CATEGORIES

InjectionStage = Literal["pre_spark", "during_spark", "post_spark"]

INJECTION_STAGES: Final[tuple[InjectionStage, ...]] = (
    "pre_spark",
    "during_spark",
    "post_spark",
)

SUPPORTED_INJECTIONS: Final[frozenset[str]] = frozenset(
    {"none", *KNOWN_CATEGORIES}
)

STAGE_FOR_CATEGORY: Final[Mapping[str, InjectionStage]] = {
    "input_not_found": "pre_spark",
    "invalid_path": "pre_spark",
    "permission_denied": "pre_spark",
    "timeout": "pre_spark",
    "runtime_error": "pre_spark",
    "spark_task_failed": "during_spark",
    "spark_driver_error": "during_spark",
    "lineage_emission_failed": "post_spark",
    "telemetry_unavailable": "post_spark",
}


# A stand-in for py4j.protocol.Py4JJavaError so we can produce an exception
# whose type *name* is `Py4JJavaError` — `failure_classifier.classify_failure`
# dispatches on `type(error).__name__`, so this is enough to drive the
# spark_task_failed / spark_driver_error code paths without requiring py4j.
class Py4JJavaError(Exception):  # noqa: N818 — name must match real class
    pass


def _raise_for_category(category: str) -> None:
    if category == "input_not_found":
        raise FileNotFoundError("injected: input file missing")
    if category == "invalid_path":
        raise IsADirectoryError("injected: input path is a directory")
    if category == "permission_denied":
        raise PermissionError("injected: input path not readable")
    if category == "timeout":
        raise socket.timeout("injected: external call timed out")
    if category == "runtime_error":
        raise RuntimeError("injected: unclassified runtime error")
    if category == "spark_task_failed":
        raise Py4JJavaError("injected: org.apache.spark TaskFailed in stage 0.0")
    if category == "spark_driver_error":
        raise Py4JJavaError("injected: org.apache.spark.SparkException at driver")
    if category == "lineage_emission_failed":
        raise RuntimeError("injected: openlineage emit rejected by backend")
    if category == "telemetry_unavailable":
        raise RuntimeError("injected: otel collector unreachable")
    # Defensive: should be unreachable thanks to the SUPPORTED_INJECTIONS guard
    # in maybe_inject(), but kept so future additions to STAGE_FOR_CATEGORY
    # without a corresponding branch above raise loudly instead of silently.
    raise AssertionError(f"no raise rule registered for category {category!r}")


def maybe_inject(category: str | None, *, stage: InjectionStage) -> None:
    """Raise a category-specific exception, but only when called from the
    stage that category is bound to.

    `None` and `"none"` are no-ops in every stage. Unknown categories or
    stages raise ValueError so callers fail fast in development.
    """
    if stage not in INJECTION_STAGES:
        raise ValueError(f"unknown stage {stage!r}; allowed: {list(INJECTION_STAGES)}")

    if category is None or category == "none":
        return

    if category not in SUPPORTED_INJECTIONS:
        raise ValueError(
            f"unknown injection category {category!r}; allowed: {sorted(SUPPORTED_INJECTIONS)}"
        )

    if STAGE_FOR_CATEGORY[category] != stage:
        return

    _raise_for_category(category)
