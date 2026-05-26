"""Tests for the FSM mode-switch threat assessment.

Spec: hl-gobblet-fsm-controller, requirement "FSM 兩模式與切換條件".

Covers the three switch cases: I can win now, the opponent can win now, and no
immediate threat. Also checks the assessment never mutates the inspected state
and honours the reveal_loses variant.
"""

from __future__ import annotations

from hl_gobblet.controllers._assess import (
    i_can_win,
    opp_can_win,
    opponent_winning_lines,
)
from hl_gobblet.moves import Move
from hl_gobblet.rules import apply_move, status_of
from hl_gobblet.state import EMPTY_CELL, GobbletState, Player, Size, initial_state


def _cell(**sizes):
    """Build a Cell tuple from SIZE_NAME=owner kwargs (e.g. SMALL=Player.P0)."""
    slots = [None, None, None]
    for name, owner in sizes.items():
        slots[int(getattr(Size, name))] = owner
    return (slots[0], slots[1], slots[2])


def _two_in_row_for(player: Player):
    """Build a position where `player` owns cells 0 and 1 (top row) by small
    pieces and it is `player`'s turn — so they are one PLACE from the top row."""
    s = initial_state()
    if player is Player.P0:
        s = apply_move(s, Move.place(Size.SMALL, 0))  # P0 @0
        s = apply_move(s, Move.place(Size.SMALL, 6))  # P1 elsewhere
        s = apply_move(s, Move.place(Size.SMALL, 1))  # P0 @1 -> P1 to move
        s = apply_move(s, Move.place(Size.SMALL, 7))  # P1 elsewhere -> P0 to move
    else:
        s = apply_move(s, Move.place(Size.SMALL, 6))  # P0 elsewhere
        s = apply_move(s, Move.place(Size.SMALL, 0))  # P1 @0
        s = apply_move(s, Move.place(Size.SMALL, 7))  # P0 elsewhere
        s = apply_move(s, Move.place(Size.SMALL, 1))  # P1 @1 -> P0 to move
    return s


def test_i_can_win_true_when_one_move_completes_line():
    """Scenario: 我方有殺 — current player owns two of a line and can claim the third."""
    s = _two_in_row_for(Player.P0)  # P0 to move, owns 0 & 1
    assert s.current is Player.P0
    assert i_can_win(s) is True


def test_i_can_win_false_at_opening():
    """No one is one move from a line at the opening."""
    s = initial_state()
    assert i_can_win(s) is False


def test_opp_can_win_true_when_opponent_threatens():
    """Scenario: 對手有立即威脅 — opponent owns two tops of a line and can claim the gap.

    P1 owns cells 0 & 1; it is P0 to move, so P1 (the opponent) is one move from
    winning the top row on their next turn.
    """
    s = _two_in_row_for(Player.P1)  # P0 to move, P1 owns 0 & 1
    assert s.current is Player.P0
    assert opp_can_win(s) is True
    threats = opponent_winning_lines(s)
    assert any(t.line == (0, 1, 2) and t.cell == 2 for t in threats)


def test_opp_can_win_false_with_no_threat():
    """Scenario: 無立即威脅 — opening position has no standing threats."""
    s = initial_state()
    assert opp_can_win(s) is False
    assert opponent_winning_lines(s) == ()


def test_assess_does_not_mutate_state():
    """The assessment functions are read-only over the position."""
    s = _two_in_row_for(Player.P1)
    before = s
    i_can_win(s)
    opp_can_win(s)
    opponent_winning_lines(s)
    assert s == before  # frozen dataclass equality: nothing changed


def _reveal_loses_trap_state() -> GobbletState:
    """P0 to move. The MOVE 5->2 completes P0's top row officially, but lifting
    P0's large off cell 5 first reveals P1's middle row (3,4,5) on the small
    layer — so under reveal_loses that very move makes P0 lose instead of win.

    Constructed directly (not via play) for a precise, deterministic position.
    """
    board = (
        _cell(SMALL=Player.P0),  # 0  P0
        _cell(SMALL=Player.P0),  # 1  P0
        EMPTY_CELL,  # 2  target of the MOVE
        _cell(SMALL=Player.P1),  # 3  P1
        _cell(SMALL=Player.P1),  # 4  P1
        _cell(SMALL=Player.P1, LARGE=Player.P0),  # 5  P1 small hidden under P0 large
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
    )
    return GobbletState(
        board=board,
        reserve=((0, 2, 1), (0, 2, 2)),
        current=Player.P0,
        move_count=8,
    )


def test_i_can_win_judged_under_active_variant():
    """Scenario: reveal_loses 一致 — i_can_win evaluates candidate moves under the
    active variant via apply_move(..., reveal_loses=...).

    The MOVE 5->2 completes P0's top row under official rules, but lifting P0's
    large off cell 5 first reveals P1's middle row, so under reveal_loses that
    same move makes P0 lose. i_can_win must reflect the variant when scoring this
    move — it is a P0 win officially and a P0 loss under reveal_loses.
    """
    s = _reveal_loses_trap_state()
    win_move = Move.move(5, 2)
    assert status_of(apply_move(s, win_move, reveal_loses=False)).winner is Player.P0
    assert status_of(apply_move(s, win_move, reveal_loses=True)).winner is Player.P1
    # i_can_win never raises and is variant-consistent (P0 still has the clean
    # PLACE-on-2 win under either flag here, so both are True; the key invariant
    # is that it routes every candidate through the active variant).
    assert i_can_win(s, reveal_loses=False) is True
    assert i_can_win(s, reveal_loses=True) is True


def test_i_can_win_false_when_only_win_self_destructs_under_reveal_loses():
    """A position whose ONLY one-move win is the self-destructing MOVE: i_can_win
    is True officially but False under reveal_loses (no clean PLACE win exists)."""
    # Same trap, but P0 cannot PLACE on cell 2 to win because P0 has no small in
    # reserve and the only line P0 can complete is via the 5->2 MOVE.
    base = _reveal_loses_trap_state()
    # Remove P0's ability to win any other way: drop P0 small reserve to 0 (already
    # 0 in the trap) and ensure cells 0,1 are the only near-line. The trap already
    # has reserve P0 small = 0, so PLACE on 2 with a small is impossible; a MEDIUM
    # or LARGE PLACE on empty cell 2 also completes the top row, so deny those too.
    s = base.with_(reserve=((0, 0, 0), base.reserve[1]))
    # Now P0 has no reserve at all; the only line-completing move is the 5->2 MOVE.
    assert i_can_win(s, reveal_loses=False) is True
    assert i_can_win(s, reveal_loses=True) is False
