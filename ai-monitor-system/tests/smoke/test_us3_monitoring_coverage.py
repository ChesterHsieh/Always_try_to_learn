from pathlib import Path


def test_us3_coverage_assets_present() -> None:
    assert Path("deploy/scripts/check-monitoring-coverage.sh").exists()
    assert Path("docs/onboarding-monitoring.md").exists()
    assert Path("docs/runbook.md").exists()
