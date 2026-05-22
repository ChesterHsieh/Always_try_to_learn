"""Contract tests: scenario YAML schema validation (Task 1.1)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from utils.scenario_schema import (
    KNOWN_FAILURE_CATEGORIES,
    SUPPORTED_INJECTIONS,
    Scenario,
    ScenarioValidationError,
    load_scenario,
    load_scenario_from_text,
)


def _write(tmp_path: Path, text: str, name: str = "case.yaml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(text).lstrip())
    return path


def test_load_success_scenario_with_required_fields() -> None:
    text = """
    name: success-baseline
    description: healthy run
    pipeline:
      input_records: 1000
      inject_failure: none
    expected_run_status: succeeded
    expected_failure_category: null
    expected_alerts: []
    probes:
      - id: records_processed
        cmd: prom-query
        args:
          query: pipeline_records_processed_total
          gte: 1
          within: 60
    """
    scenario = load_scenario_from_text(text)
    assert isinstance(scenario, Scenario)
    assert scenario.name == "success-baseline"
    assert scenario.expected_run_status == "succeeded"
    assert scenario.expected_failure_category is None
    assert scenario.expected_alerts == ()
    assert scenario.pipeline.inject_failure == "none"


def test_load_failure_scenario_with_expected_category_and_alerts() -> None:
    text = """
    name: input-not-found
    description: missing input file
    pipeline:
      input_records: 0
      inject_failure: input_not_found
    expected_run_status: failed
    expected_failure_category: input_not_found
    expected_alerts:
      - PipelineRunFailed
    probes:
      - id: failure_recorded
        cmd: prom-query
        args:
          query: 'pipeline_failures_total{failure_category="input_not_found"}'
          gte: 1
          within: 60
    """
    scenario = load_scenario_from_text(text)
    assert scenario.expected_run_status == "failed"
    assert scenario.expected_failure_category == "input_not_found"
    assert scenario.expected_alerts == ("PipelineRunFailed",)
    assert scenario.pipeline.inject_failure == "input_not_found"


def test_missing_required_field_raises_with_field_name() -> None:
    text = """
    name: incomplete
    description: missing several fields
    pipeline:
      input_records: 100
      inject_failure: none
    """
    with pytest.raises(ScenarioValidationError) as exc:
        load_scenario_from_text(text)
    msg = str(exc.value)
    for field in ("expected_run_status", "expected_failure_category", "expected_alerts", "probes"):
        assert field in msg, f"missing field {field!r} not reported in {msg!r}"


def test_failure_status_requires_non_null_failure_category() -> None:
    text = """
    name: bad-failure
    description: failed run without failure_category
    pipeline:
      input_records: 0
      inject_failure: input_not_found
    expected_run_status: failed
    expected_failure_category: null
    expected_alerts:
      - PipelineRunFailed
    probes: []
    """
    with pytest.raises(ScenarioValidationError, match="failure_category"):
        load_scenario_from_text(text)


def test_succeeded_status_requires_null_failure_category() -> None:
    text = """
    name: bad-success
    description: succeeded but with failure_category set
    pipeline:
      input_records: 1
      inject_failure: none
    expected_run_status: succeeded
    expected_failure_category: input_not_found
    expected_alerts: []
    probes: []
    """
    with pytest.raises(ScenarioValidationError, match="succeeded"):
        load_scenario_from_text(text)


def test_unknown_failure_category_rejected() -> None:
    text = """
    name: bad-cat
    description: x
    pipeline:
      input_records: 0
      inject_failure: input_not_found
    expected_run_status: failed
    expected_failure_category: not_a_real_category
    expected_alerts: [PipelineRunFailed]
    probes: []
    """
    with pytest.raises(ScenarioValidationError, match="expected_failure_category"):
        load_scenario_from_text(text)


def test_unknown_inject_failure_rejected() -> None:
    text = """
    name: bad-inject
    description: x
    pipeline:
      input_records: 0
      inject_failure: not_a_real_injection
    expected_run_status: failed
    expected_failure_category: input_not_found
    expected_alerts: [PipelineRunFailed]
    probes: []
    """
    with pytest.raises(ScenarioValidationError, match="inject_failure"):
        load_scenario_from_text(text)


def test_invalid_run_status_rejected() -> None:
    text = """
    name: bad-status
    description: x
    pipeline:
      input_records: 0
      inject_failure: none
    expected_run_status: maybe
    expected_failure_category: null
    expected_alerts: []
    probes: []
    """
    with pytest.raises(ScenarioValidationError, match="expected_run_status"):
        load_scenario_from_text(text)


def test_load_scenario_from_path(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        name: from-disk
        description: x
        pipeline:
          input_records: 1
          inject_failure: none
        expected_run_status: succeeded
        expected_failure_category: null
        expected_alerts: []
        probes: []
        """,
    )
    scenario = load_scenario(p)
    assert scenario.name == "from-disk"


def test_constants_exported() -> None:
    assert "input_not_found" in KNOWN_FAILURE_CATEGORIES
    assert "none" in SUPPORTED_INJECTIONS
    assert KNOWN_FAILURE_CATEGORIES.issubset(SUPPORTED_INJECTIONS - {"none"}) or (
        KNOWN_FAILURE_CATEGORIES <= SUPPORTED_INJECTIONS
    )
