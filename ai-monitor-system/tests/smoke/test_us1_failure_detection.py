import json
from pathlib import Path


def test_us1_dashboard_and_alert_artifacts_present() -> None:
    dashboard_path = Path("monitoring/dashboards/pipeline-health.json")
    alert_rules_path = Path("monitoring/alerts/pipeline-failure-rules.yaml")

    assert dashboard_path.exists()
    assert alert_rules_path.exists()

    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    panel_exprs = [
        target.get("expr", "")
        for panel in dashboard.get("panels", [])
        for target in panel.get("targets", [])
    ]
    assert any('pipeline_run_total{status="failed"' in expr for expr in panel_exprs)

    alert_rules = alert_rules_path.read_text(encoding="utf-8")
    assert "PipelineRunFailed" in alert_rules
    assert "for: 1m" in alert_rules
