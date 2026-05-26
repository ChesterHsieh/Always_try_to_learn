"""Play Gobblet Gobblers against the FsmGobbletV1 AI from the terminal.

Unlike watch_match.py (which only *watches* two AIs), this is interactive: on
your turn it lists every legal move, numbered, and you type the number to play
it; on the AI's turn FsmGobbletV1 picks a move and the script shows which move it
chose AND whether it was in `aggressive` or `defensive` mode (read from the AI's
decision_trace), so you can see what the controller is "thinking".

It reuses hl_gobblet.render so the on-screen board matches the snapshot-tested
pure helpers, and hl_gobblet.rules for legal-move generation / transitions /
win detection — no game logic is duplicated here.

Usage:
    ./.venv/bin/python experiments/hl-gobblet/play_human.py
    # you play P1 (second) instead of P0 (first):
    ./.venv/bin/python experiments/hl-gobblet/play_human.py --human-as 1
    # touch-move variant + a fixed seed:
    ./.venv/bin/python experiments/hl-gobblet/play_human.py --reveal-loses --seed 3

Pieces read as "<SIZE><owner>", e.g. L0 = P0 large, M1 = P1 medium, '*' = a piece
hidden underneath. Cells are numbered c0..c8 (row-major, top-left to bottom-right).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Let `import hl_gobblet...` work when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rich.console import Console  # noqa: E402

from hl_gobblet.controllers import FsmGobbletV1  # noqa: E402
from hl_gobblet.moves import Move  # noqa: E402
from hl_gobblet.render import move_description, render_panel  # noqa: E402
from hl_gobblet.rules import (  # noqa: E402
    DEFAULT_MAX_MOVES,
    apply_move,
    legal_moves,
    status_of,
)
from hl_gobblet.state import GobbletState, Player, initial_state  # noqa: E402


def _result_str(st) -> str:
    if st.draw:
        return "draw (move cap reached)"
    if st.winner is Player.P0:
        return "P0 wins"
    if st.winner is Player.P1:
        return "P1 wins"
    return "unfinished"


def _prompt_human_move(console: Console, state: GobbletState) -> Move:
    """List the legal moves (numbered) and read a valid choice from stdin.

    Re-prompts on any invalid input (non-numeric, out of range) instead of
    crashing; an empty line or 'q' aborts the game.
    """
    moves = legal_moves(state)
    console.print("[bold]Your legal moves:[/bold]")
    for i, move in enumerate(moves):
        console.print(f"  [cyan]{i:>2}[/cyan]  {move_description(state, move)}")

    while True:
        raw = console.input("[bold]pick a move number[/bold] (q to quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            raise SystemExit("aborted by player")
        if not raw.isdigit():
            console.print("[red]please type a move number from the list above[/red]")
            continue
        idx = int(raw)
        if not 0 <= idx < len(moves):
            console.print(f"[red]out of range — pick 0..{len(moves) - 1}[/red]")
            continue
        return moves[idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Gobblet Gobblers against the FsmGobbletV1 AI."
    )
    parser.add_argument(
        "--human-as",
        type=int,
        choices=(0, 1),
        default=0,
        help="which side you play: 0 = P0 (moves first), 1 = P1 (default: 0)",
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
    args = parser.parse_args()

    console = Console()
    human = Player.P0 if args.human_as == 0 else Player.P1
    ai_side = human.other

    # The AI judges wins/threats under the same variant the game is played with.
    ai = FsmGobbletV1(reveal_loses=args.reveal_loses)
    ai.reset(args.seed)

    state = initial_state(args.seed)
    before = state
    last_move = None
    title = f"Gobblet Gobblers — you = P{int(human)} vs FSM AI = P{int(ai_side)}"

    console.print(
        f"[dim]You are [bold]P{int(human)}[/bold]; the FSM AI is "
        f"[bold]P{int(ai_side)}[/bold]. reveal_loses={args.reveal_loses}.[/dim]"
    )

    for _ in range(args.max_moves + 1):
        st = status_of(state, max_moves=args.max_moves)
        if st.done:
            break

        console.print(render_panel(state, before, last_move, title=title))

        if state.current is human:
            move = _prompt_human_move(console, state)
        else:
            move = ai.act(state)
            rec = ai.decision_trace()[-1]
            console.print(
                f"[magenta]AI ({rec.state}/{rec.rule})[/magenta] plays: "
                f"{move_description(state, move)}"
            )

        before = state
        state = apply_move(state, move, reveal_loses=args.reveal_loses)
        last_move = move

    st = status_of(state, max_moves=args.max_moves)
    console.print(render_panel(state, before, last_move, title=title, result=_result_str(st)))
    console.print(f"[bold green]Game over: {_result_str(st)}[/bold green]")


if __name__ == "__main__":
    main()
