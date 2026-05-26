"""Regenerate the FsmGobbletV1 seed=0 golden decision trace.

Run only when FsmGobbletV1's behaviour is intentionally changed:
    ./.venv/bin/python tests/hl_gobblet/_gen_golden.py

Plays a fixed fsm(P0)-vs-random(P1) game at seed 0 and stores the semantic
GobbletTraceRecord fields, so the golden is robust to behaviour-neutral refactors
yet trips on any real change to the controller's move sequence.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

# Bootstrap src/ onto the path before importing the project packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hl_gobblet.controllers import FsmGobbletV1  # noqa: E402
from hl_gobblet.opponents import RandomOpponent  # noqa: E402
from hl_gobblet.rules import DEFAULT_MAX_MOVES, apply_move, status_of  # noqa: E402
from hl_gobblet.state import Player, initial_state  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden" / "fsm_gobblet_v1.seed0.json"
SEED = 0


def run_trace(seed: int = SEED) -> list[dict]:
    """Play fsm(P0) vs random(P1) and return the FSM's decision trace as dicts."""
    fsm = FsmGobbletV1()
    fsm.reset(seed)
    rnd = RandomOpponent(seed=seed + 1)
    rnd.reset(seed + 1)
    s = initial_state(seed)
    for _ in range(DEFAULT_MAX_MOVES + 1):
        if status_of(s).done:
            break
        mover = fsm if s.current is Player.P0 else rnd
        s = apply_move(s, mover.act(s))
    return [dataclasses.asdict(rec) for rec in fsm.decision_trace()]


def main() -> None:
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    records = run_trace(SEED)
    GOLDEN.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} records to {GOLDEN}")


if __name__ == "__main__":
    main()
