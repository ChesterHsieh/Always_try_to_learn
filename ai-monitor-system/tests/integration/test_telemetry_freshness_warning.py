def test_telemetry_freshness_rule_name_present() -> None:
    with open("monitoring/alerts/stack-health-rules.yaml", encoding="utf-8") as handle:
        content = handle.read()
    assert "OTelCollectorDown" in content
