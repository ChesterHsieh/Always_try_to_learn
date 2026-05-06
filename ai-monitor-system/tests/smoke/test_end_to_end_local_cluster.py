"""End-to-end smoke test: bootstrap + run SLA + coverage validation (Task 12.2)."""

import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPTS = Path(__file__).parent.parent.parent / "deploy" / "scripts"
BOOTSTRAP_SLA_SECONDS = 600
RUN_SLA_SECONDS = 300


def test_end_to_end_local_cluster_assets_ready() -> None:
    required_paths = [
        "deploy/scripts/bootstrap-local.sh",
        "deploy/scripts/run-pipeline.sh",
        "deploy/scripts/check-monitoring-coverage.sh",
        "deploy/scripts/run-smoke-test.sh",
        "deploy/helm/values.local-minimal.yaml",
    ]
    for rel_path in required_paths:
        assert Path(rel_path).exists(), f"missing required path: {rel_path}"


@pytest.mark.usefixtures("require_cluster")
def test_bootstrap_within_sla():
    start = time.monotonic()
    result = subprocess.run(
        ["bash", str(SCRIPTS / "bootstrap-local.sh")],
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start
    print(f"bootstrap: {elapsed:.1f}s (SLA: {BOOTSTRAP_SLA_SECONDS}s)")
    assert result.returncode == 0, f"bootstrap failed: {result.stderr}"
    assert elapsed <= BOOTSTRAP_SLA_SECONDS


@pytest.mark.usefixtures("require_cluster")
def test_pipeline_run_within_sla():
    start = time.monotonic()
    result = subprocess.run(
        ["bash", str(SCRIPTS / "run-pipeline.sh")],
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start
    print(f"pipeline run: {elapsed:.1f}s (SLA: {RUN_SLA_SECONDS}s)")
    assert result.returncode == 0, f"pipeline run failed: {result.stderr}"
    assert elapsed <= RUN_SLA_SECONDS


def test_coverage_report_structure_with_mocks(mock_k8s_helm_secret):
    """Verify coverage report structure and semver chart versions via mocks."""
    from utils.coverage import run_coverage

    def mock_get(url, *args, **kwargs):
        m = MagicMock()
        m.raise_for_status = MagicMock()
        if "/-/healthy" in url:
            m.status_code = 200
            m.json.return_value = {}
        elif "api/v1/rules" in url:
            m.status_code = 200
            m.json.return_value = {"data": {"groups": [{"name": "pipeline-failure", "rules": []}]}}
        elif "/api/health" in url:
            m.status_code = 200
            m.json.return_value = {"status": "up"}
        elif "namespaces" in url:
            m.status_code = 200
            m.json.return_value = {"namespaces": [{"name": "ai_monitor_system"}]}
        elif "lineage" in url:
            m.status_code = 200
            m.json.return_value = {"events": [{"runId": "abc"}]}
        else:
            m.status_code = 200
            m.json.return_value = {}
        return m

    with patch("requests.get", side_effect=mock_get):
        with patch("utils.coverage._get_k8s_api", return_value=mock_k8s_helm_secret):
            report = run_coverage(
                namespace="ai-monitor-system",
                marquez_url="http://marquez:9555",
                prometheus_url="http://prometheus:9090",
                grafana_url="http://grafana:3000",
            )

    import re

    semver = re.compile(r"^\d+\.\d+\.\d+")
    for component, version in report["components"].items():
        assert semver.match(version), f"Non-semver version '{version}' for {component}"
