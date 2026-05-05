"""Smoke test: monitoring coverage CLI direct invocation (Task 12.3)."""

from pathlib import Path
from unittest.mock import MagicMock, patch


def test_us3_coverage_assets_present() -> None:
    assert Path("deploy/scripts/check-monitoring-coverage.sh").exists()
    assert Path("docs/onboarding-monitoring.md").exists()
    assert Path("docs/runbook.md").exists()


def test_coverage_cli_pass_count_and_chart_versions(mock_k8s_helm_secret):
    """Run coverage CLI with mocked backends; assert >= 5 pass checks and 4 chart versions."""
    from pipeline.coverage import run_coverage

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
        with patch("pipeline.coverage._get_k8s_api", return_value=mock_k8s_helm_secret):
            report = run_coverage(
                namespace="ai-monitor-system",
                marquez_url="http://marquez:9555",
                prometheus_url="http://prometheus:9090",
                grafana_url="http://grafana:3000",
            )

    pass_count = sum(1 for c in report["validation_checks"] if c["status"] == "pass")
    assert pass_count >= 5, f"Expected >= 5 pass checks, got {pass_count}"
    assert len(report["components"]) == 4, f"Expected 4 components, got {len(report['components'])}"

    import re

    semver = re.compile(r"^\d+\.\d+\.\d+")
    for component, version in report["components"].items():
        assert semver.match(version), f"Invalid semver '{version}' for {component}"
