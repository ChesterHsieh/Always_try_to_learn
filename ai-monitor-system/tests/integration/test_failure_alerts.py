"""Integration tests: alert rule YAML structure and annotation contract (Task 11.2)."""

from pathlib import Path

import yaml

from pipeline.failure_classifier import classify_failure


def test_failure_classifier_for_missing_input() -> None:
    err = FileNotFoundError("missing")
    assert classify_failure(err) == "input_not_found"


def test_failure_classifier_for_directory_path() -> None:
    err = IsADirectoryError("not a file")
    assert classify_failure(err) == "invalid_path"


def test_failure_alert_rule_contains_required_context() -> None:
    content = Path("monitoring/alerts/pipeline-failure-rules.yaml").read_text(encoding="utf-8")
    assert "PipelineRunFailed" in content
    assert "severity: critical" in content


def _load_failure_rules() -> dict:
    return yaml.safe_load(
        Path("monitoring/alerts/pipeline-failure-rules.yaml").read_text(encoding="utf-8")
    )


def test_failure_alert_has_required_annotations() -> None:
    rules = _load_failure_rules()
    group = rules["groups"][0]
    for rule in group["rules"]:
        annotations = rule.get("annotations", {})
        assert "summary" in annotations, f"Rule {rule['alert']!r} missing 'summary' annotation"
        assert "dashboard_link" in annotations, (
            f"Rule {rule['alert']!r} missing 'dashboard_link' annotation"
        )
        assert "runbook_link" in annotations, (
            f"Rule {rule['alert']!r} missing 'runbook_link' annotation"
        )


def test_failure_alert_dashboard_link_has_pipeline_name_param() -> None:
    rules = _load_failure_rules()
    rule = rules["groups"][0]["rules"][0]
    dashboard_link = rule["annotations"]["dashboard_link"]
    assert "pipeline_name" in dashboard_link or "var-pipeline_name" in dashboard_link


def test_failure_alert_annotations_do_not_reference_labels_run_id() -> None:
    """Alert annotations must not reference $labels.run_id (cardinality contract)."""
    rules = _load_failure_rules()
    for group in rules["groups"]:
        for rule in group["rules"]:
            annotations = rule.get("annotations", {})
            for key, value in annotations.items():
                assert "$labels.run_id" not in str(value), (
                    f"Annotation '{key}' references $labels.run_id — "
                    "this would cause metric label cardinality explosion. "
                    "Use dashboard drilldown instead."
                )


def test_failure_alert_uses_increase_expr() -> None:
    """Alert expr should use increase() to avoid false positives on restart."""
    rules = _load_failure_rules()
    rule = rules["groups"][0]["rules"][0]
    assert "increase(" in rule["expr"]


def test_failure_alert_for_duration_is_30s() -> None:
    """Alert 'for' must be 30s per design spec (short window for scenario validation)."""
    rules = _load_failure_rules()
    for rule in rules["groups"][0]["rules"]:
        assert rule.get("for") == "30s", (
            f"Rule {rule['alert']!r} has for={rule.get('for')!r}; expected '30s'"
        )
