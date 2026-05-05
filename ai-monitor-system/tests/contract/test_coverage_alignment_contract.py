"""Contract tests: three-way alignment gate (Task 4.2).

Validates that:
1. Each KNOWN_CATEGORY has a matching scenario file in scenarios/.
2. Each KNOWN_CATEGORY has a runbook anchor (## failure-<category>) in docs/runbook.md.
3. pipeline-failure-rules.yaml defines all required alert names.
4. Production helm values do NOT contain INJECT_FAILURE.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from pipeline.failure_classifier import KNOWN_CATEGORIES

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
RUNBOOK_PATH = PROJECT_ROOT / "docs" / "runbook.md"
ALERT_RULES_PATH = PROJECT_ROOT / "monitoring" / "alerts" / "pipeline-failure-rules.yaml"
PROD_VALUES_PATH = PROJECT_ROOT / "deploy" / "helm" / "values.yaml"

# Alert names that must exist in pipeline-failure-rules.yaml
REQUIRED_ALERT_NAMES = {
    "PipelineRunFailed",
    "PipelineSparkDriverError",
    "PipelineLineageEmissionFailed",
    "PipelineTelemetryUnavailable",
    "PipelineRunTimeout",
}


def test_each_known_category_has_scenario_file() -> None:
    """Every KNOWN_CATEGORY must have at least one scenario YAML in scenarios/."""
    scenario_categories: set[str] = set()
    for path in SCENARIOS_DIR.glob("*.yaml"):
        raw = yaml.safe_load(path.read_text())
        cat = raw.get("expected_failure_category")
        if cat:
            scenario_categories.add(cat)

    missing = KNOWN_CATEGORIES - scenario_categories
    assert not missing, f"No scenario file covers these failure categories: {sorted(missing)}"


def test_each_known_category_has_runbook_anchor() -> None:
    """docs/runbook.md must contain a ## failure-<category> anchor for each KNOWN_CATEGORY."""
    runbook_text = RUNBOOK_PATH.read_text()
    missing = []
    for category in sorted(KNOWN_CATEGORIES):
        anchor = f"failure-{category}"
        # Match markdown heading + anchor patterns: ## ... {#failure-x} or ## failure-x
        pattern = rf"failure-{re.escape(category)}"
        if not re.search(pattern, runbook_text):
            missing.append(anchor)

    assert not missing, (
        f"runbook.md is missing anchors for: {missing}. "
        "Add a section with '## failure-<category>' or '{{#failure-<category>}}' for each."
    )


def test_required_alert_names_present_in_rules_yaml() -> None:
    """pipeline-failure-rules.yaml must define all required alert names."""
    rules = yaml.safe_load(ALERT_RULES_PATH.read_text())
    defined_alerts = {
        rule["alert"] for group in rules.get("groups", []) for rule in group.get("rules", [])
    }
    missing = REQUIRED_ALERT_NAMES - defined_alerts
    assert not missing, (
        f"pipeline-failure-rules.yaml is missing these alert definitions: {sorted(missing)}"
    )


def test_production_values_do_not_contain_inject_failure() -> None:
    """Production helm values.yaml must not expose INJECT_FAILURE env var."""
    prod_values_text = PROD_VALUES_PATH.read_text()
    assert "INJECT_FAILURE" not in prod_values_text, (
        "Production values.yaml contains INJECT_FAILURE — "
        "this env gate must only appear in test/local overlays."
    )
