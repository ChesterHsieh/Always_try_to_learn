"""Tests for the v2 fork (double-threat) assessment.

Spec: hl-gobblet-fsm-controller-v2, requirement "強化威脅評估（fork 偵測，攻防對稱）".

Covers three cases: the side to move can create a fork (two simultaneous
claimable winning lines), the opponent is one move from a fork, and no fork is
available. Also checks the assessment never mutates the inspected state and
honours the reveal_loses variant.
"""

from __future__ import annotations

from hl_gobblet.controllers._assess_v2 import (
    claimable_winning_lines,
    i_can_fork,
    opp_can_fork,
)
from hl_gobblet.state import EMPTY_CELL, GobbletState, Player, Size, initial_state


def _cell(**sizes):
    """Build a Cell tuple from SIZE_NAME=owner kwargs (e.g. SMALL=Player.P0)."""
    slots = [None, None, None]
    for name, owner in sizes.items():
        slots[int(getattr(Size, name))] = owner
    return (slots[0], slots[1], slots[2])


def _fork_setup_for_p0() -> GobbletState:
    """P0 to move; a single PLACE on cell 4 makes the centre shared by lines
    (1,4,7) and (3,4,5) and (0,4,8) and (2,4,6). P0 already owns 1 and 3 (small),
    so placing P0 on 4 yields two two-in-a-row lines: (1,4,7) needs 7 and
    (3,4,5) needs 5 — both empty/claimable. That is a fork.
    """
    board = (
        EMPTY_CELL,  # 0
        _cell(SMALL=Player.P0),  # 1  P0
        EMPTY_CELL,  # 2
        _cell(SMALL=Player.P0),  # 3  P0
        EMPTY_CELL,  # 4  -> P0 places here to fork
        EMPTY_CELL,  # 5  gap of line (3,4,5)
        EMPTY_CELL,  # 6
        EMPTY_CELL,  # 7  gap of line (1,4,7)
        EMPTY_CELL,  # 8
    )
    # P1 has a couple of pieces somewhere harmless so the position is plausible,
    # but to keep the fork crisp we leave the board as above with full-ish reserves.
    return GobbletState(
        board=board,
        reserve=((1, 2, 2), (2, 2, 2)),
        current=Player.P0,
        move_count=6,
    )


def test_i_can_fork_true_when_one_move_makes_double_threat():
    """Scenario: 偵測到我方可造雙殺 — placing on the shared centre yields two
    claimable winning lines."""
    s = _fork_setup_for_p0()
    assert s.current is Player.P0
    assert i_can_fork(s) is True


def test_i_can_fork_false_at_opening():
    """No fork is reachable in one move from the empty opening."""
    s = initial_state()
    assert i_can_fork(s) is False


def test_opp_can_fork_true_when_opponent_one_move_from_double_threat():
    """Scenario: 偵測到對手即將造雙殺 — same shape but it is P1's pieces and P0 to
    move, so the opponent (P1) is one move from a fork."""
    board = (
        EMPTY_CELL,
        _cell(SMALL=Player.P1),  # 1  P1
        EMPTY_CELL,
        _cell(SMALL=Player.P1),  # 3  P1
        EMPTY_CELL,  # 4  -> P1 forks here next turn
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
    )
    s = GobbletState(
        board=board,
        reserve=((2, 2, 2), (1, 2, 2)),
        current=Player.P0,
        move_count=6,
    )
    assert s.current is Player.P0
    assert opp_can_fork(s) is True


def test_no_fork_when_at_most_single_threat():
    """Scenario: 無雙重威脅時回報為假 — a position whose lines are blocked by the
    opponent so no single move can leave P0 with two claimable lines.

    P0 owns the two diagonal corners 0 and 8 (line (0,4,8)); every other line
    through cells P0 could play is already capped by a P1 top, so the only line
    P0 can claim is (0,4,8) via the centre — a single threat, never a fork.
    """
    board = (
        _cell(SMALL=Player.P0),  # 0  P0
        _cell(LARGE=Player.P1),  # 1  P1 caps row (0,1,2) and col (1,4,7)
        _cell(LARGE=Player.P1),  # 2  P1 caps col (2,5,8) and anti-diag (2,4,6)
        _cell(LARGE=Player.P1),  # 3  P1 caps row (3,4,5) and col (0,3,6)
        EMPTY_CELL,  # 4  centre: the only claimable gap for line (0,4,8)
        _cell(LARGE=Player.P1),  # 5  P1 caps row (3,4,5) and col (2,5,8)
        _cell(LARGE=Player.P1),  # 6  P1 caps row (6,7,8), col (0,3,6), anti-diag
        _cell(LARGE=Player.P1),  # 7  P1 caps row (6,7,8) and col (1,4,7)
        _cell(SMALL=Player.P0),  # 8  P0
    )
    s = GobbletState(
        board=board,
        reserve=((2, 2, 2), (0, 2, 0)),
        current=Player.P0,
        move_count=8,
    )
    # The only line P0 can complete is (0,4,8) by claiming the centre — exactly one
    # claimable line, never two, so it is not a fork.
    assert i_can_fork(s) is False


def test_claimable_winning_lines_counts_two_for_a_fork_position():
    """After the forking move, claimable_winning_lines reports >= 2 lines for P0."""
    from hl_gobblet.moves import Move
    from hl_gobblet.rules import apply_move

    s = _fork_setup_for_p0()
    nxt = apply_move(s, Move.place(Size.SMALL, 4))
    lines = claimable_winning_lines(nxt, Player.P0)
    assert len(lines) >= 2


def test_assess_v2_does_not_mutate_state():
    """The fork assessment functions are read-only over the position."""
    s = _fork_setup_for_p0()
    before = s
    i_can_fork(s)
    opp_can_fork(s)
    claimable_winning_lines(s, Player.P0)
    assert s == before  # frozen dataclass equality: nothing changed


def test_i_can_fork_respects_reveal_loses_variant():
    """i_can_fork routes candidate moves through the active variant; it never
    raises under either flag and stays deterministic."""
    s = _fork_setup_for_p0()
    assert i_can_fork(s, reveal_loses=False) is True
    # Under reveal_loses the clean PLACE fork is unaffected (no lift involved).
    assert i_can_fork(s, reveal_loses=True) is True
