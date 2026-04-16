def test_telemetry_freshness_rule_name_present() -> None:
    rules_path = "monitoring/alerts/stack-health-rules.yaml"
    with open(rules_path, encoding="utf-8") as handle:
        content = handle.read()
    assert "OTelCollectorDown" in content
    assert "TelemetryFreshnessHigh" in content
    assert "pipeline_telemetry_freshness_seconds > 120" in content
