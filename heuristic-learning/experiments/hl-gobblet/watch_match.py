"""Watch two AIs play a full game of 奇迹连连 (Gobblet Gobblers) in the terminal.

This script only "watches" — it renders an AI-vs-AI game step by step with rich;
it writes no report and computes no statistics. It mirrors the lander
play_gui.py convention (sys.path bootstrap, argparse, opponent factory) and reuses
hl_gobblet.render so the on-screen logic matches the snapshot-tested pure helpers.

Opponents: `random` (uniform legal move) and `fsm` (FsmGobbletV1, a gradient-free
FSM controller that switches between aggressive/defensive modes).

Usage:
    ./.venv/bin/python experiments/hl-gobblet/watch_match.py \\
        --p0 fsm --p1 random --seed 0 --delay 0.6
    # step through one ply at a time (press Enter):
    ./.venv/bin/python experiments/hl-gobblet/watch_match.py --step
    # play the touch-move variant:
    ./.venv/bin/python experiments/hl-gobblet/watch_match.py --reveal-loses
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Let `import hl_gobblet...` work when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rich.console import Console  # noqa: E402
from rich.live import Live  # noqa: E402

from hl_gobblet.render import render_panel  # noqa: E402
from hl_gobblet.rules import DEFAULT_MAX_MOVES, apply_move, status_of  # noqa: E402
from hl_gobblet.state import Player, initial_state  # noqa: E402


def _opponent_factory(name: str, seed: int, *, reveal_loses: bool = False):
    if name == "random":
        from hl_gobblet.opponents import RandomOpponent

        return RandomOpponent(seed=seed)
    if name == "fsm":
        from hl_gobblet.controllers import FsmGobbletV1

        # Pass the match variant so the controller judges wins/threats consistently.
        return FsmGobbletV1(reveal_loses=reveal_loses)
    if name == "fsm-v2":
        from hl_gobblet.controllers import FsmGobbletV2

        return FsmGobbletV2(reveal_loses=reveal_loses)
    raise SystemExit(f"unknown opponent '{name}' (available: random, fsm, fsm-v2)")


def _result_str(st) -> str:
    if st.draw:
        return "draw (move cap reached)"
    if st.winner is Player.P0:
        return "P0 wins"
    if st.winner is Player.P1:
        return "P1 wins"
    return "unfinished"


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch two AIs play Gobblet Gobblers.")
    parser.add_argument(
        "--p0", default="random", help="opponent for P0: random|fsm|fsm-v2 (default: random)"
    )
    parser.add_argument(
        "--p1", default="random", help="opponent for P1: random|fsm|fsm-v2 (default: random)"
    )
    parser.add_argument("--seed", type=int, default=0, help="game seed (default: 0)")
    parser.add_argument(
        "--reveal-loses",
        action="store_true",
        help="enable the touch-move variant (lifting that reveals opponent's line loses)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=DEFAULT_MAX_MOVES,
        help=f"ply cap before a draw (default: {DEFAULT_MAX_MOVES})",
    )
    parser.add_argument("--delay", type=float, default=0.6, help="seconds between plies")
    parser.add_argument(
        "--step",
        action="store_true",
        help="advance one ply per Enter keypress instead of auto-playing",
    )
    args = parser.parse_args()

    console = Console()
    # Distinct seeds per side so two 'random' AIs don't mirror each other.
    p0 = _opponent_factory(args.p0, seed=args.seed, reveal_loses=args.reveal_loses)
    p1 = _opponent_factory(args.p1, seed=args.seed + 1, reveal_loses=args.reveal_loses)
    p0.reset(args.seed)
    p1.reset(args.seed + 1)

    state = initial_state(args.seed)
    before = state
    last_move = None
    title = f"Gobblet Gobblers — P0 ({args.p0}) vs P1 ({args.p1})  seed={args.seed}"

    with Live(
        render_panel(state, before, last_move, title=title),
        console=console,
        refresh_per_second=30,
    ) as live:
        for _ in range(args.max_moves + 1):
            st = status_of(state, max_moves=args.max_moves)
            if st.done:
                live.update(
                    render_panel(state, before, last_move, title=title, result=_result_str(st))
                )
                break

            mover = p0 if state.current is Player.P0 else p1
            move = mover.act(state)
            before = state
            state = apply_move(state, move, reveal_loses=args.reveal_loses)
            last_move = move

            st = status_of(state, max_moves=args.max_moves)
            result = _result_str(st) if st.done else None
            live.update(render_panel(state, before, last_move, title=title, result=result))

            if st.done:
                break
            if args.step:
                console.input("[dim]press Enter for next ply...[/dim]")
            else:
                time.sleep(args.delay)


if __name__ == "__main__":
    main()
