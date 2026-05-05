"""Contract tests: failure_injection interface skeleton (Task 1.3).

Verifies the stage-aware injection contract defined in design.md:
- `none` is a no-op in every stage.
- Each known category fires only at its assigned stage; calling it from a
  wrong stage is a no-op (so callers can safely place injection points at
  every pipeline boundary without worrying about double-firing).
- Unknown categories fail fast with ValueError.
"""

from __future__ import annotations

import pytest

from pipeline.failure_classifier import KNOWN_CATEGORIES, classify_failure
from pipeline.failure_injection import (
    INJECTION_STAGES,
    STAGE_FOR_CATEGORY,
    SUPPORTED_INJECTIONS,
    maybe_inject,
)

ALL_STAGES = ("pre_spark", "during_spark", "post_spark")


def test_supported_injections_includes_none_and_all_known() -> None:
    assert "none" in SUPPORTED_INJECTIONS
    assert KNOWN_CATEGORIES.issubset(SUPPORTED_INJECTIONS)
    assert "schema_mismatch" in SUPPORTED_INJECTIONS


def test_injection_stages_constant() -> None:
    assert set(INJECTION_STAGES) == set(ALL_STAGES)


def test_stage_for_category_covers_every_supported_non_none() -> None:
    expected = SUPPORTED_INJECTIONS - {"none"}
    assert set(STAGE_FOR_CATEGORY) == expected
    for stage in STAGE_FOR_CATEGORY.values():
        assert stage in INJECTION_STAGES


@pytest.mark.parametrize("stage", ALL_STAGES)
def test_none_is_noop_in_every_stage(stage: str) -> None:
    maybe_inject("none", stage=stage)  # type: ignore[arg-type]
    maybe_inject(None, stage=stage)  # type: ignore[arg-type]


def test_unknown_category_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not_a_real_category"):
        maybe_inject("not_a_real_category", stage="pre_spark")


def test_unknown_stage_raises_value_error() -> None:
    with pytest.raises(ValueError, match="stage"):
        maybe_inject("input_not_found", stage="not_a_stage")  # type: ignore[arg-type]


@pytest.mark.parametrize("category", sorted(SUPPORTED_INJECTIONS - {"none"}))
def test_wrong_stage_is_noop(category: str) -> None:
    """Calling a category from a stage other than its assigned one must
    be a no-op so we can drop maybe_inject() at every pipeline boundary.
    """
    assigned = STAGE_FOR_CATEGORY[category]
    for stage in ALL_STAGES:
        if stage == assigned:
            continue
        maybe_inject(category, stage=stage)  # type: ignore[arg-type]


@pytest.mark.parametrize("category", sorted(SUPPORTED_INJECTIONS - {"none"}))
def test_correct_stage_raises_classifiable_exception(category: str) -> None:
    """Each category must raise an exception that classify_failure maps
    back to the same category (or to spark_driver_error for schema_mismatch).
    """
    assigned = STAGE_FOR_CATEGORY[category]
    with pytest.raises(BaseException) as exc_info:
        maybe_inject(category, stage=assigned)
    actual = classify_failure(exc_info.value)
    expected = "spark_driver_error" if category == "schema_mismatch" else category
    assert actual == expected, (
        f"injection {category!r} at stage {assigned!r} produced "
        f"exception classified as {actual!r}, expected {expected!r}"
    )
