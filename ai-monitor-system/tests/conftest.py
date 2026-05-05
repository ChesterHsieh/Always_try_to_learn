import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)


@pytest.fixture()
def in_memory_span_exporter():
    """OTel InMemorySpanExporter for verifying span attributes in tests."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    import opentelemetry.trace as trace_api

    original_provider = trace_api.get_tracer_provider()
    trace_api.set_tracer_provider(provider)
    yield exporter
    exporter.clear()
    trace_api.set_tracer_provider(original_provider)


@pytest.fixture()
def mock_marquez_server():
    """Mock HTTP server simulating Marquez /api/v1/lineage endpoint."""
    received_payloads = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            import json

            received_payloads.append(json.loads(body))
            self.send_response(200)
            self.end_headers()

        def do_GET(self):
            import json

            if self.path.startswith("/api/v1/namespaces"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"namespaces": [{"name": "ai_monitor_system"}]}).encode()
                )
            elif "/api/v1/events/lineage" in self.path:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {"events": received_payloads[-3:] if received_payloads else []}
                    ).encode()
                )
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):
            pass  # suppress output

    server = HTTPServer(("localhost", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    yield {"url": f"http://localhost:{port}", "payloads": received_payloads}
    server.shutdown()


@pytest.fixture()
def mock_k8s_helm_secret():
    """Mock kubernetes CoreV1Api fixture returning fake Helm release Secrets."""
    import base64
    import gzip
    import json

    def _make_secret(release_name: str, chart_version: str):
        metadata = {"chart": {"metadata": {"version": chart_version}}}
        raw = json.dumps(metadata).encode()
        compressed = gzip.compress(raw)
        encoded = base64.b64encode(base64.b64encode(compressed)).decode()
        secret = MagicMock()
        secret.metadata.name = f"sh.helm.release.v1.{release_name}.v1"
        secret.metadata.labels = {"owner": "helm", "name": release_name}
        secret.data = {"release": encoded}
        return secret

    releases = {
        "upstream-prometheus": "25.27.0",
        "upstream-grafana": "8.5.1",
        "upstream-otel-collector": "0.78.0",
        "upstream-marquez": "6.7.0",
    }
    secrets = [_make_secret(name, ver) for name, ver in releases.items()]

    mock_api = MagicMock()
    mock_list = MagicMock()
    mock_list.items = secrets
    mock_api.list_namespaced_secret.return_value = mock_list
    yield mock_api


@pytest.fixture(autouse=True)
def reset_prometheus_registry():
    """Reset prometheus_client registry between tests to prevent cross-test pollution."""
    import prometheus_client

    # Collect all existing collectors to restore after test
    collectors_before = set(prometheus_client.REGISTRY._names_to_collectors.copy().keys())
    yield
    # Unregister any collectors added during the test
    to_remove = [
        c
        for name, c in list(prometheus_client.REGISTRY._names_to_collectors.items())
        if name not in collectors_before
    ]
    seen = set()
    for collector in to_remove:
        if id(collector) not in seen:
            seen.add(id(collector))
            try:
                prometheus_client.REGISTRY.unregister(collector)
            except Exception:
                pass
