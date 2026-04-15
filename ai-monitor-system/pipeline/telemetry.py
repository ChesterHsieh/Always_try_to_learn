from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TelemetryTags:
    run_id: str
    pipeline_name: str
    labels: dict[str, str] = field(default_factory=dict)


def build_otel_attributes(tags: TelemetryTags) -> dict[str, str]:
    attrs = {
        "run_id": tags.run_id,
        "pipeline_name": tags.pipeline_name,
    }
    attrs.update(tags.labels)
    return attrs


def lifecycle_metric_payload(run_id: str, status: str) -> dict[str, str]:
    return {"run_id": run_id, "status": status, "timestamp": datetime.now(timezone.utc).isoformat()}
