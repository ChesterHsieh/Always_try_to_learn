"""Regression for the 3x3 {random, v1, v2} head-to-head win-rate matrix.

Spec: hl-gobblet-fsm-controller-v2, requirement "3×3 交互對打結果矩陣". Asserts the
matrix is deterministic (same input -> same matrix) and the key monotonicity /
parity properties hold: both FSM controllers crush random; v2 is at least as
strong as v1 against random; v2 is not weaker than v1 head-to-head than a parity
floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The matrix builder lives with the experiment script; put it on the path.
_EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "hl-gobblet"
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

from matchup_matrix import build_matrix  # type: ignore[import-not-found]  # noqa: E402

# 30 seeds x 6 ordered pairs x 2 sides keeps the matrix regression fast while the
# coarse thresholds below stay well-separated from the measured values.
_SEEDS = range(30)
_OPENING_PLIES = 4
_PARITY_FLOOR = 0.35
_VS_RANDOM_EPS = 0.07


def test_matrix_is_deterministic():
    """Scenario: 產生 3×3 矩陣 — same seeds/opening reproduce the same matrix."""
    m1 = build_matrix(_SEEDS, _OPENING_PLIES)
    m2 = build_matrix(_SEEDS, _OPENING_PLIES)
    assert m1 == m2


def test_matrix_monotonicity_and_thresholds():
    """Scenario: 矩陣單調性與門檻回歸."""
    m = build_matrix(_SEEDS, _OPENING_PLIES)
    v1_vs_random = m[("v1", "random")]
    v2_vs_random = m[("v2", "random")]
    v2_vs_v1 = m[("v2", "v1")]

    # Both FSM controllers decisively beat random.
    assert v1_vs_random > 0.8, f"v1 vs random {v1_vs_random:.0%} not decisive"
    assert v2_vs_random > 0.8, f"v2 vs random {v2_vs_random:.0%} not decisive"
    # v2 is at least as strong as v1 against random (not weakened by its extras).
    assert v2_vs_random >= v1_vs_random - _VS_RANDOM_EPS
    # v2 is not clearly weaker than v1 head to head.
    assert v2_vs_v1 >= _PARITY_FLOOR, f"v2 vs v1 {v2_vs_v1:.0%} below parity floor"
