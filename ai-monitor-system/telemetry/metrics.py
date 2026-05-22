import logging
import os
from typing import Protocol

from prometheus_client import (  # noqa: I001 — keep grouped imports below
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    push_to_gateway,
    start_http_server,
)

from pipeline.failure_classifier import KNOWN_CATEGORIES

logger = logging.getLogger(__name__)

_FORBIDDEN_LABELS = frozenset({"run_id", "failure_message"})

pipeline_run_total = Counter(
    "pipeline_run_total",
    "Total pipeline runs by status",
    ["status", "pipeline_name"],
)
pipeline_run_duration_seconds = Histogram(
    "pipeline_run_duration_seconds",
    "Pipeline run duration in seconds",
    ["status", "pipeline_name"],
)
pipeline_records_processed_total = Counter(
    "pipeline_records_processed_total",
    "Total records processed by pipeline",
    ["pipeline_name"],
)
pipeline_failures_total = Counter(
    "pipeline_failures_total",
    "Total pipeline failures by category",
    ["failure_category", "pipeline_name"],
)
pipeline_telemetry_freshness_seconds = Gauge(
    "pipeline_telemetry_freshness_seconds",
    "Seconds since last telemetry event",
    ["pipeline_name"],
)

# Contract guard: verify no forbidden labels in any collector at import time
for _collector in [
    pipeline_run_total,
    pipeline_run_duration_seconds,
    pipeline_records_processed_total,
    pipeline_failures_total,
    pipeline_telemetry_freshness_seconds,
]:
    for _label in _collector._labelnames:
        if _label in _FORBIDDEN_LABELS:
            raise ValueError(
                f"Collector '{_collector._name}' contains forbidden "
                f"high-cardinality label '{_label}'. Use exemplars instead."
            )


class MetricsRecorder(Protocol):
    def start_endpoint(self, *, port: int) -> None: ...
    def record_run_started(self, *, run_id: str, pipeline_name: str) -> None: ...
    def record_run_succeeded(
        self,
        *,
        run_id: str,
        pipeline_name: str,
        duration_seconds: float,
        records_processed: int,
    ) -> None: ...
    def record_run_failed(
        self,
        *,
        run_id: str,
        pipeline_name: str,
        duration_seconds: float,
        failure_category: str,
        failure_message: str,
    ) -> None: ...
    def update_freshness(
        self, *, run_id: str, pipeline_name: str, seconds_since_last_event: float
    ) -> None: ...


class PrometheusMetricsRecorder:
    def start_endpoint(self, *, port: int) -> None:
        try:
            start_http_server(port)
            logger.info("Metrics endpoint started on port %d", port)
        except Exception as exc:
            logger.error("Failed to start metrics endpoint: %s", exc)
            raise SystemExit(78) from exc

    def record_run_started(self, *, run_id: str, pipeline_name: str) -> None:
        try:
            pipeline_run_total.labels(status="running", pipeline_name=pipeline_name).inc(
                exemplar={"run_id": run_id}
            )
        except Exception:
            logger.exception("metrics record_run_started failed", extra={"run_id": run_id})

    def record_run_succeeded(
        self,
        *,
        run_id: str,
        pipeline_name: str,
        duration_seconds: float,
        records_processed: int,
        trace_id: str | None = None,
    ) -> None:
        try:
            exemplar = {"run_id": run_id}
            if trace_id:
                exemplar["trace_id"] = trace_id
            pipeline_run_total.labels(status="succeeded", pipeline_name=pipeline_name).inc(
                exemplar=exemplar
            )
            pipeline_run_duration_seconds.labels(
                status="succeeded", pipeline_name=pipeline_name
            ).observe(duration_seconds, exemplar=exemplar)
            pipeline_records_processed_total.labels(pipeline_name=pipeline_name).inc(
                records_processed, exemplar=exemplar
            )
        except Exception:
            logger.exception("metrics record_run_succeeded failed", extra={"run_id": run_id})

    def record_run_failed(
        self,
        *,
        run_id: str,
        pipeline_name: str,
        duration_seconds: float,
        failure_category: str,
        failure_message: str,
        trace_id: str | None = None,
    ) -> None:
        # Fail fast on unknown categories: a bad label here would silently
        # break alert PromQL (which matches on a closed enum) and corrupt
        # the cardinality bound documented in design.md.
        if failure_category not in KNOWN_CATEGORIES:
            raise ValueError(
                f"failure_category {failure_category!r} is not in KNOWN_CATEGORIES; "
                f"allowed: {sorted(KNOWN_CATEGORIES)}"
            )
        try:
            exemplar = {
                "run_id": run_id,
                "failure_message": failure_message[:80],
            }
            if trace_id:
                exemplar["trace_id"] = trace_id
            pipeline_run_total.labels(status="failed", pipeline_name=pipeline_name).inc(
                exemplar={"run_id": run_id}
            )
            pipeline_run_duration_seconds.labels(
                status="failed", pipeline_name=pipeline_name
            ).observe(duration_seconds, exemplar={"run_id": run_id})
            pipeline_failures_total.labels(
                failure_category=failure_category, pipeline_name=pipeline_name
            ).inc(exemplar=exemplar)
        except Exception:
            logger.exception("metrics record_run_failed failed", extra={"run_id": run_id})

    def update_freshness(
        self, *, run_id: str, pipeline_name: str, seconds_since_last_event: float
    ) -> None:
        try:
            pipeline_telemetry_freshness_seconds.labels(pipeline_name=pipeline_name).set(
                seconds_since_last_event
            )
        except Exception:
            logger.exception("metrics update_freshness failed", extra={"run_id": run_id})


def get_recorder() -> PrometheusMetricsRecorder:
    return PrometheusMetricsRecorder()


def get_metrics_port() -> int:
    return int(os.environ.get("METRICS_PORT", "9095"))


def push_metrics_if_configured(
    *, job_name: str, pipeline_name: str, run_id: str
) -> None:
    """Push collected metrics to the Pushgateway if PUSHGATEWAY_URL is set.

    Pipeline pods are batch jobs — by the time Prometheus scrapes the pod, the
    pod has often already exited. The Pushgateway is the standard fix: the
    job pushes its final metrics before terminating, and Prometheus scrapes
    the gateway instead of ephemeral pod IPs.

    Grouping key includes run_id so each run gets its own slot in the gateway
    (no cross-run overwrite during scenario test runs).
    """
    url = os.environ.get("PUSHGATEWAY_URL")
    if not url:
        return
    try:
        push_to_gateway(
            url,
            job=job_name,
            registry=REGISTRY,
            grouping_key={"pipeline_name": pipeline_name, "run_id": run_id},
        )
        logger.info("Pushed metrics to %s (job=%s, run_id=%s)", url, job_name, run_id)
    except Exception:
        logger.exception("push_to_gateway failed (url=%s)", url)
