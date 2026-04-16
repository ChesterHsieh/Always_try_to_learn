from pipeline.failure_classifier import classify_failure


def test_failure_classifier_for_missing_input() -> None:
    err = FileNotFoundError("missing")
    assert classify_failure(err) == "input_not_found"


def test_failure_classifier_for_directory_path() -> None:
    err = IsADirectoryError("not a file")
    assert classify_failure(err) == "invalid_path"


def test_failure_alert_rule_contains_required_context() -> None:
    with open("monitoring/alerts/pipeline-failure-rules.yaml", encoding="utf-8") as handle:
        content = handle.read()
    assert "PipelineRunFailed" in content
    assert 'status="failed"' in content
    assert "severity: critical" in content
