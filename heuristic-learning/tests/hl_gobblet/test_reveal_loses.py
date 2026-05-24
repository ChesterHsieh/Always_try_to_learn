"""Tests for the reveal_loses (touch-move) advanced variant.

Spec: hl-gobblet-env, requirement "reveal_loses 進階變體".

We build positions directly with GobbletState.with_ so the lift instant can be
set up precisely. Setup (P0 to move):
  cell 0: P1 small UNDER a P0 large   (top = P0)
  cell 1: P1 small                    (top = P1)
  cell 2: P1 small                    (top = P1)
Top row tops are P0, P1, P1. If P0 lifts the large off cell 0, the P1 small
underneath is revealed and the top row becomes P1, P1, P1 — the opponent's line
appears at the lift instant.
"""

from __future__ import annotations

from hl_gobblet.moves import Move
from hl_gobblet.rules import apply_move, line_winner, status_of
from hl_gobblet.state import EMPTY_CELL, GobbletState, Player


def _lift_instant_setup() -> GobbletState:
    P0, P1 = Player.P0, Player.P1
    board = [EMPTY_CELL] * 9
    board[0] = (P1, None, P0)  # P1 small underneath, P0 large on top
    board[1] = (P1, None, None)  # P1 small
    board[2] = (P1, None, None)  # P1 small
    # Reserves are irrelevant to the MOVE under test; use plausible counts.
    return GobbletState(
        board=tuple(board),
        reserve=((1, 2, 1), (0, 2, 2)),
        current=P0,
        move_count=6,
    )


def test_reveal_loses_enabled_lifter_loses():
    """Scenario: 拿起即揭露對方連線判負（啟用）.

    P0 moves the large off cell 0 to an empty cell (5). With reveal_loses on, the
    lift instant exposes P1's top row -> P0 loses immediately.
    """
    s = _lift_instant_setup()
    # Sanity: before the move, nobody has a line (P0 large caps cell 0).
    assert line_winner(s.board) is None

    result = apply_move(s, Move.move(0, 5), reveal_loses=True)
    st = status_of(result)
    assert st.done
    assert st.winner is Player.P1  # the opponent (not the lifter) wins
    assert not st.draw


def test_reveal_loses_disabled_judged_after_landing():
    """Scenario: 同樣局面在官方規則下不判負（關閉）.

    Same lift, but with reveal_loses off the line is only judged after landing.
    P0 lands the large on cell 1 (covering P1's small there), so the final top
    row is P1(revealed), P0, P1 — no line — and the game continues.
    """
    s = _lift_instant_setup()
    result = apply_move(s, Move.move(0, 1), reveal_loses=False)
    st = status_of(result)
    assert not st.done  # no win after landing
    # Final board reflects the official outcome:
    from hl_gobblet.state import top_owner

    assert top_owner(result.board[0]) is Player.P1  # revealed small
    assert top_owner(result.board[1]) is Player.P0  # large landed here, covering P1
    assert top_owner(result.board[2]) is Player.P1
    assert line_winner(result.board) is None


def test_disabled_does_not_penalize_even_when_landing_elsewhere():
    """With reveal_loses off, moving the large to an empty cell (5) re-exposes the
    P1 line on the final board — and THAT is a legitimate official win for P1
    (the line exists after the move completes), not a touch-move penalty."""
    s = _lift_instant_setup()
    result = apply_move(s, Move.move(0, 5), reveal_loses=False)
    st = status_of(result)
    # Official rules: after landing, top row is P1,P1,P1 -> P1 wins on the board.
    assert st.done and st.winner is Player.P1
