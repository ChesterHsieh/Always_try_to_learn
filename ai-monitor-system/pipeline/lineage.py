from datetime import datetime, timezone


def build_openlineage_event(
    run_id: str,
    job_name: str,
    namespace: str,
    source_dataset: str,
    target_dataset: str,
    *,
    trace_id: str | None = None,
    event_type: str = "COMPLETE",
    error_message: str | None = None,
    failure_category: str | None = None,
) -> dict:
    """Build a minimal OpenLineage event payload.

    `event_type` should be one of START / RUNNING / COMPLETE / ABORT / FAIL.
    For FAIL/ABORT events, `error_message` and `failure_category` populate
    an errorMessage facet so downstream lineage backends can surface why
    the run terminated — this is how the schema-drift scenario gets
    visibility on plan-analyzer failures that the OpenLineage Spark
    listener never observes (because no Spark job is launched).
    """

    event = {
        "eventType": event_type,
        "eventTime": datetime.now(timezone.utc).isoformat(),
        "run": {"runId": run_id},
        "job": {"namespace": namespace, "name": job_name},
        "inputs": [{"namespace": namespace, "name": source_dataset}],
        "outputs": [{"namespace": namespace, "name": target_dataset}],
        # OpenLineage spec requires top-level `producer` and `schemaURL`.
        "producer": "https://github.com/OpenLineage/OpenLineage/tree/main/integration/python",
        "schemaURL": (
            "https://openlineage.io/spec/2-0-2/OpenLineage.json"
            "#/definitions/RunEvent"
        ),
    }

    facets: dict = {}
    if trace_id:
        facets["trace"] = {"traceId": trace_id}
    if event_type in {"FAIL", "ABORT"} and (error_message or failure_category):
        facets["errorMessage"] = {
            "_producer": "ai-monitor-system",
            "_schemaURL": (
                "https://openlineage.io/spec/facets/1-0-0/ErrorMessageRunFacet.json"
            ),
            "message": error_message or "(no message)",
            "programmingLanguage": "python",
            "stackTrace": failure_category or "",
        }

    if facets:
        event["run"]["facets"] = facets

    return event
