from datetime import datetime, timezone


def build_openlineage_event(
    run_id: str,
    job_name: str,
    namespace: str,
    source_dataset: str,
    target_dataset: str,
    *,
    trace_id: str | None = None,
) -> dict:
    event = {
        "eventType": "COMPLETE",
        "run": {"runId": run_id},
        "job": {"namespace": namespace, "name": job_name},
        "inputs": [{"namespace": namespace, "name": source_dataset}],
        "outputs": [{"namespace": namespace, "name": target_dataset}],
        "eventTime": datetime.now(timezone.utc).isoformat(),
    }
    if trace_id:
        event["run"]["facets"] = {"trace": {"traceId": trace_id}}
    return event
