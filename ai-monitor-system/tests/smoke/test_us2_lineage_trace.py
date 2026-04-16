import json
from pathlib import Path


def test_us2_lineage_dashboard_has_correlation_panels() -> None:
    dashboard_path = Path("monitoring/dashboards/lineage-view.json")
    assert dashboard_path.exists()

    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    panel_titles = [panel.get("title", "") for panel in dashboard.get("panels", [])]
    assert "Telemetry Freshness" in panel_titles
    assert "Lineage Events" in panel_titles
