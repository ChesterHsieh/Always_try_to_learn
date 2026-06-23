"""Print the 3x3 head-to-head win-rate matrix for {random, v1, v2}.

Capability: hl-gobblet-fsm-controller-v2, requirement "3×3 交互對打結果矩陣". For
every ordered pair of controllers it plays a fixed set of seeds, each side once
(alternating who moves at handoff), from a seeded random opening so the
deterministic FSM controllers produce a distribution of distinct games rather
than a single replayed line. Prints win-rate of the ROW controller against the
COLUMN controller.

This script only computes and prints the matrix; it writes no files. It mirrors
the watch_match.py sys.path bootstrap convention.

Usage:
    ./.venv/bin/python experiments/hl-gobblet/matchup_matrix.py
    ./.venv/bin/python experiments/hl-gobblet/matchup_matrix.py --seeds 200 --opening 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Let `import hl_gobblet...` work when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hl_gobblet.controllers import FsmGobbletV1, FsmGobbletV2  # noqa: E402
from hl_gobblet.opponents import RandomOpponent  # noqa: E402
from hl_gobblet.rules import DEFAULT_MAX_MOVES, apply_move, legal_moves, status_of  # noqa: E402
from hl_gobblet.state import Player, initial_state  # noqa: E402

# Default number of seeded random opening plies before the controllers take over.
DEFAULT_OPENING_PLIES = 4
NAMES = ("random", "v1", "v2")


def _factory(name: str):
    """Return a fresh controller for one game, seeded off the game seed."""
    if name == "random":
        return lambda seed: _reset(RandomOpponent(seed=seed + 1000), seed + 1000)
    if name == "v1":
        return lambda seed: _reset(FsmGobbletV1(), seed)
    if name == "v2":
        return lambda seed: _reset(FsmGobbletV2(), seed)
    raise SystemExit(f"unknown controller '{name}' (available: {', '.join(NAMES)})")


def _reset(opp, seed: int):
    opp.reset(seed)
    return opp


def _random_opening(seed: int, plies: int):
    rng = np.random.default_rng(seed)
    s = initial_state(seed)
    for _ in range(plies):
        if status_of(s).done:
            return None
        moves = legal_moves(s)
        s = apply_move(s, moves[int(rng.integers(len(moves)))])
    return None if status_of(s).done else s


def _play(p0_factory, p1_factory, seed: int, opening_plies: int):
    start = _random_opening(seed, opening_plies)
    if start is None:
        return None
    p0 = p0_factory(seed)
    p1 = p1_factory(seed)
    s = start
    for _ in range(DEFAULT_MAX_MOVES + 1):
        st = status_of(s)
        if st.done:
            return st
        mover = p0 if s.current is Player.P0 else p1
        s = apply_move(s, mover.act(s))
    return status_of(s)


def winrate(a: str, b: str, seeds, opening_plies: int = DEFAULT_OPENING_PLIES) -> float:
    """Win rate of controller `a` against `b`, alternating sides, over `seeds`."""
    fa, fb = _factory(a), _factory(b)
    wins = games = 0
    for seed in seeds:
        st = _play(fa, fb, seed, opening_plies)
        if st is not None:
            games += 1
            if not st.draw and st.winner is Player.P0:
                wins += 1
        st = _play(fb, fa, seed, opening_plies)
        if st is not None:
            games += 1
            if not st.draw and st.winner is Player.P1:
                wins += 1
    return wins / games if games else 0.0


def build_matrix(seeds, opening_plies: int = DEFAULT_OPENING_PLIES) -> dict[tuple[str, str], float]:
    """Compute the full {random, v1, v2} x {random, v1, v2} win-rate matrix
    (diagonal omitted: a controller vs itself is ~0.5 by symmetry)."""
    matrix: dict[tuple[str, str], float] = {}
    for a in NAMES:
        for b in NAMES:
            if a == b:
                continue
            matrix[(a, b)] = winrate(a, b, seeds, opening_plies)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="3x3 head-to-head win-rate matrix.")
    parser.add_argument("--seeds", type=int, default=100, help="number of seeds (default: 100)")
    parser.add_argument(
        "--opening",
        type=int,
        default=DEFAULT_OPENING_PLIES,
        help=f"seeded random opening plies (default: {DEFAULT_OPENING_PLIES})",
    )
    args = parser.parse_args()

    seeds = range(args.seeds)
    matrix = build_matrix(seeds, args.opening)

    print(f"3x3 win-rate matrix (row beats col), seeds=0..{args.seeds - 1}, opening={args.opening}")
    header = "         " + "".join(f"{n:>10}" for n in NAMES)
    print(header)
    for a in NAMES:
        cells = []
        for b in NAMES:
            cells.append("   --   " if a == b else f"{matrix[(a, b)]:>8.1%}")
        print(f"{a:>8} " + "".join(f"{c:>10}" for c in cells))


if __name__ == "__main__":
    main()
