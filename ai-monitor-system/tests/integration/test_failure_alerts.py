from pipeline.failure_classifier import classify_failure


def test_failure_classifier_for_missing_input() -> None:
    err = FileNotFoundError("missing")
    assert classify_failure(err) == "input_not_found"
