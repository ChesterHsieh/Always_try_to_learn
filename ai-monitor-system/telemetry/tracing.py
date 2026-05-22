import uuid
from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace


def build_trace_attributes(
    run_id: str, pipeline_name: str, namespace: str, status: str
) -> dict[str, str]:
    return {
        "trace_id": str(uuid.uuid4()),
        "run_id": run_id,
        "pipeline_name": pipeline_name,
        "k8s_namespace": namespace,
        "status": status,
    }


@contextmanager
def start_run_span(
    *,
    run_id: str,
    pipeline_name: str,
    k8s_namespace: str,
) -> Generator:
    """Context manager that creates a run span and sets terminal attributes."""
    tracer = trace.get_tracer("pipeline.tracing")
    with tracer.start_as_current_span("pipeline_run") as span:
        span.set_attribute("pipeline.run_id", run_id)
        span.set_attribute("pipeline.name", pipeline_name)
        span.set_attribute("k8s.namespace", k8s_namespace)
        try:
            yield span
            span.set_attribute("status", "succeeded")
        except Exception as exc:
            span.set_attribute("status", "failed")
            span.record_exception(exc)
            from opentelemetry.trace import StatusCode

            span.set_status(StatusCode.ERROR, str(exc))
            raise


def get_current_trace_id() -> str | None:
    """Extract hex trace_id from the current active span, if any."""
    current_span = trace.get_current_span()
    ctx = current_span.get_span_context()
    if ctx.is_valid:
        return format(ctx.trace_id, "032x")
    return None
