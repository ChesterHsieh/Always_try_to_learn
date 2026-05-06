"""Scenario YAML schema parsing and validation.

Single source of truth for the failure-scenario contract consumed by
`scripts/run-scenario.sh`, `deploy/scripts/check-monitoring-coverage.sh`,
and the integration test suite.

A scenario describes a pipeline run that the monitoring stack must observe
and classify correctly. The schema is intentionally narrow: every required
field has a downstream consumer that fails fast when the field is absent or
inconsistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Mapping

import yaml

from pipeline.failure_classifier import KNOWN_CATEGORIES

KNOWN_FAILURE_CATEGORIES: Final[frozenset[str]] = KNOWN_CATEGORIES

SUPPORTED_INJECTIONS: Final[frozenset[str]] = frozenset(
    {"none", "schema_mismatch", *KNOWN_CATEGORIES}
)

RunStatus = Literal["succeeded", "failed"]
_VALID_RUN_STATUSES: Final[frozenset[str]] = frozenset({"succeeded", "failed"})


class ScenarioValidationError(ValueError):
    """Raised when a scenario YAML fails schema validation.

    Carries every offending field in the message so the runner / coverage
    check can surface a complete list to the operator on first failure.
    """


@dataclass(frozen=True)
class Probe:
    id: str
    cmd: str
    args: Mapping[str, Any]


@dataclass(frozen=True)
class PreRun:
    """Optional baseline run executed before the main run.

    Used by multi-run scenarios (e.g. schema-mismatch) to seed the lineage
    backend with a prior dataset version before the failing run is observed.
    """

    schema_version: str
    inject_failure: str = "none"


@dataclass(frozen=True)
class PipelineSpec:
    input_records: int
    inject_failure: str
    schema_version: str = "v1"
    pre_runs: tuple[PreRun, ...] = ()


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    pipeline: PipelineSpec
    expected_run_status: RunStatus
    expected_failure_category: str | None
    expected_alerts: tuple[str, ...]
    probes: tuple[Probe, ...]


_REQUIRED_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "name",
    "description",
    "pipeline",
    "expected_run_status",
    "expected_failure_category",
    "expected_alerts",
    "probes",
)
_REQUIRED_PIPELINE_FIELDS: Final[tuple[str, ...]] = ("input_records", "inject_failure")


def _check_required(
    payload: Mapping[str, Any],
    fields: tuple[str, ...],
    *,
    location: str,
) -> list[str]:
    return [f"{location}.{f}" for f in fields if f not in payload]


def _validate(payload: Mapping[str, Any]) -> Scenario:
    if not isinstance(payload, Mapping):
        raise ScenarioValidationError("scenario root must be a mapping")

    missing = _check_required(payload, _REQUIRED_TOP_LEVEL_FIELDS, location="<root>")
    pipeline_block = payload.get("pipeline")
    if isinstance(pipeline_block, Mapping):
        missing += _check_required(pipeline_block, _REQUIRED_PIPELINE_FIELDS, location="pipeline")
    elif "pipeline" in payload:
        raise ScenarioValidationError("'pipeline' must be a mapping")

    if missing:
        raise ScenarioValidationError(
            "scenario is missing required fields: " + ", ".join(sorted(missing))
        )

    assert isinstance(pipeline_block, Mapping)  # for type narrowing

    inject_failure = pipeline_block["inject_failure"]
    if inject_failure not in SUPPORTED_INJECTIONS:
        raise ScenarioValidationError(
            f"pipeline.inject_failure {inject_failure!r} is not one of "
            f"{sorted(SUPPORTED_INJECTIONS)}"
        )

    expected_run_status = payload["expected_run_status"]
    if expected_run_status not in _VALID_RUN_STATUSES:
        raise ScenarioValidationError(
            f"expected_run_status {expected_run_status!r} must be one of "
            f"{sorted(_VALID_RUN_STATUSES)}"
        )

    expected_failure_category = payload["expected_failure_category"]
    if expected_run_status == "failed":
        if expected_failure_category is None:
            raise ScenarioValidationError(
                "expected_run_status='failed' requires a non-null expected_failure_category"
            )
        if expected_failure_category not in KNOWN_FAILURE_CATEGORIES:
            raise ScenarioValidationError(
                f"expected_failure_category {expected_failure_category!r} is "
                f"not in KNOWN_CATEGORIES"
            )
    else:  # succeeded
        if expected_failure_category is not None:
            raise ScenarioValidationError(
                "expected_run_status='succeeded' requires expected_failure_category to be null"
            )

    raw_alerts = payload["expected_alerts"]
    if not isinstance(raw_alerts, list) or not all(isinstance(a, str) for a in raw_alerts):
        raise ScenarioValidationError("expected_alerts must be a list of strings")

    raw_probes = payload["probes"]
    if not isinstance(raw_probes, list):
        raise ScenarioValidationError("probes must be a list")
    probes: list[Probe] = []
    for idx, item in enumerate(raw_probes):
        if not isinstance(item, Mapping):
            raise ScenarioValidationError(f"probes[{idx}] must be a mapping")
        for f in ("id", "cmd"):
            if f not in item:
                raise ScenarioValidationError(f"probes[{idx}].{f} is required")
        probes.append(Probe(id=str(item["id"]), cmd=str(item["cmd"]), args=item.get("args", {})))

    raw_pre_runs = pipeline_block.get("pre_runs", []) or []
    if not isinstance(raw_pre_runs, list):
        raise ScenarioValidationError("pipeline.pre_runs must be a list when present")
    pre_runs: list[PreRun] = []
    for idx, item in enumerate(raw_pre_runs):
        if not isinstance(item, Mapping):
            raise ScenarioValidationError(f"pipeline.pre_runs[{idx}] must be a mapping")
        sv = item.get("schema_version")
        if not isinstance(sv, str) or not sv:
            raise ScenarioValidationError(
                f"pipeline.pre_runs[{idx}].schema_version is required (non-empty string)"
            )
        pre_inject = item.get("inject_failure", "none")
        if pre_inject not in SUPPORTED_INJECTIONS:
            raise ScenarioValidationError(
                f"pipeline.pre_runs[{idx}].inject_failure {pre_inject!r} is not one of "
                f"{sorted(SUPPORTED_INJECTIONS)}"
            )
        pre_runs.append(PreRun(schema_version=sv, inject_failure=str(pre_inject)))

    schema_version = pipeline_block.get("schema_version", "v1")
    if not isinstance(schema_version, str) or not schema_version:
        raise ScenarioValidationError(
            "pipeline.schema_version must be a non-empty string when present"
        )

    return Scenario(
        name=str(payload["name"]),
        description=str(payload["description"]),
        pipeline=PipelineSpec(
            input_records=int(pipeline_block["input_records"]),
            inject_failure=str(inject_failure),
            schema_version=schema_version,
            pre_runs=tuple(pre_runs),
        ),
        expected_run_status=expected_run_status,  # type: ignore[arg-type]
        expected_failure_category=expected_failure_category,
        expected_alerts=tuple(raw_alerts),
        probes=tuple(probes),
    )


def load_scenario_from_text(text: str) -> Scenario:
    payload = yaml.safe_load(text)
    return _validate(payload)


def load_scenario(path: str | Path) -> Scenario:
    return load_scenario_from_text(Path(path).read_text())
