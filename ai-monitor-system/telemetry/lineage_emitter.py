import logging
import os
import time

import requests

from pipeline.failure_classifier import classify_failure
from telemetry.lineage import build_openlineage_event

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY_SECONDS = 1.0


def maybe_shadow_emit(
    *,
    run_id: str,
    job_name: str,
    namespace: str,
    source_dataset: str,
    target_dataset: str,
    marquez_url: str | None = None,
    trace_id: str | None = None,
    event_type: str = "COMPLETE",
    error_message: str | None = None,
    failure_category: str | None = None,
) -> bool:
    """Emit an OpenLineage event to Marquez if shadow mode is enabled.

    For successful runs, called with default event_type="COMPLETE".
    For failed runs (especially plan-analyzer errors that the OpenLineage
    Spark listener cannot observe), call with event_type="FAIL" plus
    error_message/failure_category so the lineage backend records the
    correct terminal state — this is the only way the schema-drift
    scenario can verify R3.5/R3.6's "lineage detection path".
    """

    if os.environ.get("LINEAGE_SHADOW_EMIT", "").lower() != "true":
        return False

    url = marquez_url or os.environ.get(
        "MARQUEZ_URL", "http://ai-monitor-system-upstream-marquez:9555"
    )
    endpoint = f"{url}/api/v1/lineage"

    # OpenLineage protocol requires a START event to precede any terminal
    # event (COMPLETE / FAIL / ABORT) for the same runId. The OpenLineage
    # Spark listener does not emit START for plan-analyzer failures (no
    # Spark job is launched), so the shadow emitter is what creates the
    # run record in the lineage backend. Best-effort — failures here are
    # logged but never escalated.
    if event_type in {"FAIL", "ABORT", "COMPLETE"}:
        start_payload = build_openlineage_event(
            run_id=run_id,
            job_name=job_name,
            namespace=namespace,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            trace_id=trace_id,
            event_type="START",
        )
        try:
            requests.post(endpoint, json=start_payload, timeout=5)
        except Exception as exc:
            logger.warning(
                "Shadow lineage START emit error for run_id=%s: %s", run_id, exc
            )

    payload = build_openlineage_event(
        run_id=run_id,
        job_name=job_name,
        namespace=namespace,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        trace_id=trace_id,
        event_type=event_type,
        error_message=error_message,
        failure_category=failure_category,
    )

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(endpoint, json=payload, timeout=5)
            if resp.status_code < 400:
                logger.info("Shadow lineage emit succeeded for run_id=%s", run_id)
                return True
            logger.warning(
                "Shadow lineage emit attempt %d failed with status %d for run_id=%s",
                attempt + 1,
                resp.status_code,
                run_id,
            )
        except Exception as exc:
            category = classify_failure(exc)
            logger.warning(
                "Shadow lineage emit attempt %d error (category=%s) for run_id=%s: %s",
                attempt + 1,
                category,
                run_id,
                exc,
            )

        if attempt < _MAX_RETRIES - 1:
            delay = _BASE_DELAY_SECONDS * (2**attempt)
            time.sleep(delay)

    category = "lineage_emission_failed"
    logger.error(
        "Shadow lineage emit failed after %d retries (category=%s) run_id=%s",
        _MAX_RETRIES,
        category,
        run_id,
        extra={"run_id": run_id, "failure_category": category},
    )
    return False
