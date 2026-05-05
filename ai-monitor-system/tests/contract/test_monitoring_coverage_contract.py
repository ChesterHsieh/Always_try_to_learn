"""Contract tests: CoverageReport structure, components, chart_version (Task 10.5)."""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_option_b_chart_registry_exists() -> None:
    assert Path("docs/chart-version-matrix.md").exists()


def test_coverage_report_components_contain_four_charts(mock_k8s_helm_secret):
    """CoverageReport.components must include all 4 upstream charts with semver versions."""
    from pipeline.coverage import run_coverage

    mock_prom = MagicMock()
    mock_prom.get.return_value.status_code = 200
    mock_prom.get.return_value.json.return_value = {
        "data": {"groups": [{"name": "pipeline-failure", "rules": []}]}
    }

    mock_grafana = MagicMock()
    mock_grafana.get.return_value.status_code = 200
    mock_grafana.get.return_value.json.return_value = {"status": "up"}

    mock_marquez_ns = MagicMock()
    mock_marquez_ns.status_code = 200
    mock_marquez_ns.json.return_value = {"namespaces": [{"name": "ai_monitor_system"}]}

    mock_marquez_events = MagicMock()
    mock_marquez_events.status_code = 200
    mock_marquez_events.json.return_value = {"events": [{"runId": "r1"}]}

    def mock_get(url, *args, **kwargs):
        if "prometheus" in url or ":9090" in url:
            if "api/v1/rules" in url:
                return mock_prom.get.return_value
            return mock_prom.get.return_value
        if "grafana" in url or ":3000" in url:
            return mock_grafana.get.return_value
        if "marquez" in url or ":9555" in url:
            if "namespaces" in url:
                return mock_marquez_ns
            if "lineage" in url:
                return mock_marquez_events
            return mock_marquez_ns
        raise ValueError(f"Unexpected URL: {url}")

    semver_pattern = re.compile(r"^\d+\.\d+\.\d+")
    expected_components = {
        "upstream-prometheus",
        "upstream-grafana",
        "upstream-otel-collector",
        "upstream-marquez",
    }

    with patch("requests.get", side_effect=mock_get):
        with patch("pipeline.coverage._get_k8s_api", return_value=mock_k8s_helm_secret):
            report = run_coverage(
                namespace="ai-monitor-system",
                marquez_url="http://marquez:9555",
                prometheus_url="http://prometheus:9090",
                grafana_url="http://grafana:3000",
            )

    assert "components" in report
    assert "validation_checks" in report
    assert "last_verified_at" in report
    assert "profile_name" in report

    # last_verified_at must be ISO 8601
    import datetime

    ts = report["last_verified_at"]
    datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))

    # All 4 components present with semver versions
    for component in expected_components:
        assert component in report["components"], f"Missing component: {component}"
        version = report["components"][component]
        assert version and semver_pattern.match(version), (
            f"Component '{component}' has invalid version: '{version}'"
        )

    # At least 5 validation checks
    assert len(report["validation_checks"]) >= 5


def test_coverage_report_validation_checks_structure(mock_k8s_helm_secret):
    from pipeline.coverage import run_coverage

    def mock_get(url, *args, **kwargs):
        m = MagicMock()
        if "api/v1/rules" in url:
            m.status_code = 200
            m.json.return_value = {"data": {"groups": [{"name": "pipeline-failure", "rules": []}]}}
        elif "grafana" in url or ":3000" in url:
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

    for check in report["validation_checks"]:
        assert "name" in check
        assert "status" in check
        assert check["status"] in ("pass", "warn", "fail")
        assert "detail" in check
