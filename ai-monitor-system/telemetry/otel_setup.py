import logging
import os
from contextlib import contextmanager
from typing import Generator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

_provider: TracerProvider | None = None


def configure_tracer(*, otlp_endpoint: str | None = None) -> TracerProvider:
    global _provider
    if _provider is not None:
        return _provider

    endpoint = otlp_endpoint or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
    )

    resource = Resource.create(
        {
            "service.name": "pyspark-pipeline",
            "k8s.namespace": os.environ.get("K8S_NAMESPACE", "ai-monitor-system"),
        }
    )

    try:
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _provider = provider
        logger.info("OTel TracerProvider configured with endpoint: %s", endpoint)
        return provider
    except Exception as exc:
        logger.error("Failed to configure OTel tracer: %s", exc)
        raise SystemExit(78) from exc


def get_provider() -> TracerProvider | None:
    return _provider


def reset_provider() -> None:
    """Reset for testing purposes only."""
    global _provider
    _provider = None


@contextmanager
def start_run_span(
    *,
    run_id: str,
    pipeline_name: str,
    k8s_namespace: str,
) -> Generator:
    tracer = trace.get_tracer("pipeline.job")
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


def force_flush(timeout_millis: int = 2000) -> None:
    provider = get_provider()
    if provider is not None:
        try:
            provider.force_flush(timeout_millis=timeout_millis)
        except Exception:
            logger.warning("OTel force_flush failed or timed out")
