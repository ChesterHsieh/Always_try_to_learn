"""Regenerate the fsm_macro_v1 seed=0 golden trace.

Run only when fsm_macro_v1's behaviour is intentionally changed:
    ./.venv/bin/python tests/hl_core/_gen_golden.py

Stores only the semantic TraceRecord fields (step/state/rule/macro_active/action)
so the golden is robust to behaviour-neutral refactors (Design D4).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

# Bootstrap src/ onto the path before importing the project packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import gymnasium as gym  # noqa: E402

from hl_lander.controllers.fsm_macro_v1 import FsmMacroLanderV1  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden" / "fsm_macro_v1.seed0.json"


def run_trace(seed: int = 0) -> list[dict]:
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


def main() -> None:
    records = run_trace(0)
    GOLDEN.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} records to {GOLDEN}")


if __name__ == "__main__":
    main()
