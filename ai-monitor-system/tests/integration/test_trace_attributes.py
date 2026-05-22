"""Integration tests: OTel span attributes via InMemorySpanExporter (Task 4.3)."""

from contextlib import contextmanager

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from telemetry.tracing import build_trace_attributes


@contextmanager
def _run_span_with_provider(provider, run_id, pipeline_name, k8s_namespace):
    """Create a run span using a specific provider (bypasses global set_tracer_provider lock)."""
    tracer = provider.get_tracer("pipeline.tracing")
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


@pytest.fixture()
def provider_with_exporter():
    """Fresh TracerProvider + InMemorySpanExporter pair (not global)."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_trace_attributes_have_run_id() -> None:
    attrs = build_trace_attributes("r1", "pipe", "ns", "ok")
    assert attrs["run_id"] == "r1"
    assert attrs["trace_id"]
    assert attrs["k8s_namespace"] == "ns"


def test_successful_run_span_attributes(provider_with_exporter):
    provider, exporter = provider_with_exporter
    with _run_span_with_provider(provider, "run-success-1", "test-pipe", "ai-monitor-system"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[-1]
    assert span.attributes.get("pipeline.run_id") == "run-success-1"
    assert span.attributes.get("pipeline.name") == "test-pipe"
    assert span.attributes.get("k8s.namespace") == "ai-monitor-system"
    assert span.attributes.get("status") == "succeeded"


def test_failed_run_span_attributes(provider_with_exporter):
    provider, exporter = provider_with_exporter
    with pytest.raises(RuntimeError):
        with _run_span_with_provider(provider, "run-fail-1", "test-pipe", "ai-monitor-system"):
            raise RuntimeError("test failure")

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    span = spans[-1]
    assert span.attributes.get("pipeline.run_id") == "run-fail-1"
    assert span.attributes.get("status") == "failed"
    from opentelemetry.trace import StatusCode

    assert span.status.status_code == StatusCode.ERROR


def test_span_has_all_required_attributes(provider_with_exporter):
    provider, exporter = provider_with_exporter
    with _run_span_with_provider(provider, "run-attrs-test", "my-pipeline", "ns-test"):
        pass

    spans = exporter.get_finished_spans()
    span = spans[-1]
    required = {"pipeline.run_id", "pipeline.name", "k8s.namespace", "status"}
    actual = set(span.attributes.keys())
    missing = required - actual
    assert not missing, f"Span missing required attributes: {missing}"
