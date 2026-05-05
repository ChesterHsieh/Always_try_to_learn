from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TelemetryTags:
    run_id: str
    pipeline_name: str
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class TelemetryEnvelope:
    signal_type: str
    run_id: str
    timestamp: str
    attributes: dict[str, str] = field(default_factory=dict)


def build_otel_attributes(tags: TelemetryTags) -> dict[str, str]:
    attrs = {
        "run_id": tags.run_id,
        "pipeline_name": tags.pipeline_name,
    }
    attrs.update(tags.labels)
    return attrs


def build_signal_envelope(
    signal_type: str,
    run_id: str,
    attributes: dict[str, str] | None = None,
) -> TelemetryEnvelope:
    return TelemetryEnvelope(
        signal_type=signal_type,
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        attributes=attributes or {},
    )


def build_correlation_attributes(
    *,
    run_id: str,
    trace_id: str | None = None,
    lineage_run_id: str | None = None,
) -> dict[str, str]:
    attrs = {"run_id": run_id}
    if trace_id:
        attrs["trace_id"] = trace_id
    if lineage_run_id:
        attrs["lineage_run_id"] = lineage_run_id
    return attrs


def lifecycle_metric_payload(
    run_id: str,
    status: str,
    *,
    pipeline_name: str = "ai_monitor_simple_pipeline",
    failure_category: str | None = None,
    failure_message: str | None = None,
) -> dict[str, str]:
    payload: dict[str, str] = {
        "run_id": run_id,
        "status": status,
        "pipeline_name": pipeline_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if failure_category:
        payload["failure_category"] = failure_category
    if failure_message:
        payload["failure_message"] = failure_message
    return payload
