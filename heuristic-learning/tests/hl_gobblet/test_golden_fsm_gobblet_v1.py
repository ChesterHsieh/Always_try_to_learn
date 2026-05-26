"""Golden-trace regression for FsmGobbletV1 (spec: 棋力門檻與 Golden Trace 回歸).

Freezes FsmGobbletV1's seed=0 fsm-vs-random decision trace. A change to the
controller (or to a building block it relies on) that alters the move sequence
fails this test, forcing the author to declare the behaviour change deliberate
and regenerate the golden via tests/hl_gobblet/_gen_golden.py. Compares only the
semantic fields stored in the golden.

To confirm the guard bites, temporarily perturb a rule in the controller and
re-run this test — it must go RED.
"""

from __future__ import annotations

import json
from pathlib import Path

from _gen_golden import GOLDEN, SEED, run_trace  # type: ignore[import-not-found]


def test_fsm_gobblet_v1_matches_golden_trace() -> None:
    golden = json.loads(Path(GOLDEN).read_text(encoding="utf-8"))
    live = run_trace(SEED)
    assert len(live) == len(golden), f"trace length drifted: {len(live)} != {len(golden)}"
    for i, (got, exp) in enumerate(zip(live, golden)):
        assert got == exp, f"trace record {i} drifted: {got} != {exp}"
