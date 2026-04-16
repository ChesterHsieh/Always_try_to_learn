from pathlib import Path


def test_required_monitoring_stack_configs_exist() -> None:
    assert Path("monitoring/otel/collector-config.yaml").exists()
    assert Path("monitoring/prometheus/prometheus.yml").exists()
    assert Path("monitoring/grafana/datasources.yaml").exists()


def test_option_b_chart_registry_exists() -> None:
    assert Path("docs/chart-version-matrix.md").exists()
