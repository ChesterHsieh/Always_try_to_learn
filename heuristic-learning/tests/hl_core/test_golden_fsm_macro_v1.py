"""Golden-trace regression for fsm_macro_v1 (spec: 回歸測試與 Golden Trace).

Freezes fsm_macro_v1's seed=0 decision trace. A change to hl_core or the
controller that alters the trace fails this test, forcing the author to declare
the behaviour change deliberate and regenerate the golden via
tests/hl_core/_gen_golden.py. Compares only semantic fields (Design D4).

To confirm the guard bites, temporarily perturb a rule in fsm_macro_v1 and re-run
this test — it must go RED (verified during task 9.2).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import gymnasium as gym

from hl_lander.controllers.fsm_macro_v1 import FsmMacroLanderV1

GOLDEN = Path(__file__).resolve().parent / "golden" / "fsm_macro_v1.seed0.json"


def _live_trace(seed: int = 0) -> list[dict]:
    env = gym.make("LunarLander-v3")
    try:
        policy = FsmMacroLanderV1()
        policy.reset(seed)
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, _r, terminated, truncated, _ = env.step(int(policy.act(obs)))
        return [dataclasses.asdict(rec) for rec in policy.decision_trace()]
    finally:
        env.close()


def test_fsm_macro_v1_matches_golden_trace() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    live = _live_trace(0)
    assert len(live) == len(golden), f"trace length drifted: {len(live)} != {len(golden)}"
    for i, (got, exp) in enumerate(zip(live, golden)):
        assert got == exp, f"trace record {i} drifted: {got} != {exp}"
