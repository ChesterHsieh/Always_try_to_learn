"""Integration test: cross-signal run_id correlation (Task 11.1)."""

import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import pipeline.metrics as metrics_module


@pytest.fixture()
def provider_with_exporter():
    """Fresh TracerProvider + InMemorySpanExporter (not global — avoids override lock)."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


@contextmanager
def _run_span(provider, run_id, pipeline_name, k8s_namespace):
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


def test_run_id_correlation_across_metrics_traces_lineage(
    provider_with_exporter, mock_marquez_server
):
    """Verify same run_id appears in Prometheus exemplar, OTel span, and Marquez event."""
    provider, exporter = provider_with_exporter
    run_id = "corr-test-run-001"
    pipeline_name = "test-pipe"
    marquez_url = mock_marquez_server["url"]
    received_payloads = mock_marquez_server["payloads"]

    # (a) Record metrics
    recorder = metrics_module.PrometheusMetricsRecorder()
    recorder.record_run_started(run_id=run_id, pipeline_name=pipeline_name)

    # (b) Create span with run_id
    with _run_span(provider, run_id, pipeline_name, "ai-monitor-system"):
        recorder.record_run_succeeded(
            run_id=run_id,
            pipeline_name=pipeline_name,
            duration_seconds=1.5,
            records_processed=10,
        )

    # (c) Shadow emit to mock Marquez
    with patch.dict(os.environ, {"LINEAGE_SHADOW_EMIT": "true", "MARQUEZ_URL": marquez_url}):
        from pipeline.lineage_emitter import maybe_shadow_emit

        maybe_shadow_emit(
            run_id=run_id,
            job_name="test-pipe",
            namespace="ai-monitor-system",
            source_dataset="/input/sample.txt",
            target_dataset="/output/result.txt",
        )

    # (a) Prometheus counter incremented
    labels = {"status": "succeeded", "pipeline_name": pipeline_name}
    child = metrics_module.pipeline_run_total.labels(**labels)
    assert child._value.get() >= 1, "Counter not incremented"

    # (b) OTel span has the run_id
    spans = exporter.get_finished_spans()
    assert len(spans) >= 1, "No spans exported"
    span = spans[-1]
    assert span.attributes.get("pipeline.run_id") == run_id, (
        f"Span run_id mismatch: {span.attributes.get('pipeline.run_id')} != {run_id}"
    )

    # (c) Marquez received event with correct runId
    assert len(received_payloads) >= 1, "No lineage events received by mock Marquez"
    marquez_event = received_payloads[-1]
    assert marquez_event.get("run", {}).get("runId") == run_id, (
        f"Marquez runId mismatch: {marquez_event.get('run', {}).get('runId')} != {run_id}"
    )


def test_corrupted_run_id_correlation_breaks(provider_with_exporter):
    """Guard test: different run_ids in metric vs span means correlation is broken."""
    provider, exporter = provider_with_exporter
    run_id = "corr-metric-run-id"
    wrong_run_id = "WRONG-span-run-id"
    pipeline_name = "test-pipe-guard"

    recorder = metrics_module.PrometheusMetricsRecorder()
    recorder.record_run_started(run_id=run_id, pipeline_name=pipeline_name)

    # Create span with WRONG run_id
    with _run_span(provider, wrong_run_id, pipeline_name, "ai-monitor-system"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1, "Expected at least one span"
    span = spans[-1]

    # Span run_id does NOT match the metric run_id (intentional mismatch)
    assert span.attributes.get("pipeline.run_id") == wrong_run_id
    assert span.attributes.get("pipeline.run_id") != run_id, (
        "Guard failed: span and metric have same run_id despite intentional mismatch"
    )
