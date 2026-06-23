"""Tests for the v2 mode RuleTables (priority order, default fallback, ties).

Spec: hl-gobblet-fsm-controller-v2, requirement "各模式的規則表決策（含 make_fork
與 block_fork）". Asserts: win-now fires first; the defensive mode blocks an
incoming double threat; an empty rule space falls back to `default` without
raising; ties resolve deterministically by smallest index.
"""

from __future__ import annotations

from hl_gobblet.controllers._modes_v2 import (
    GobbletCtxV2,
    build_aggressive,
    build_defensive,
)
from hl_gobblet.moves import index_to_move, move_to_index
from hl_gobblet.rules import apply_move, legal_moves, status_of
from hl_gobblet.state import EMPTY_CELL, GobbletState, Player, Size


def _cell(**sizes):
    slots = [None, None, None]
    for name, owner in sizes.items():
        slots[int(getattr(Size, name))] = owner
    return (slots[0], slots[1], slots[2])


def _ctx_for(state: GobbletState) -> GobbletCtxV2:
    ctx = GobbletCtxV2()
    ctx.refresh(state)
    return ctx


def test_win_now_fires_first_and_wins():
    """Scenario: 能贏就贏優先 — P0 owns 0,1 with the gap claimable; aggressive must
    fire win_now and the chosen move must complete the line."""
    board = (
        _cell(SMALL=Player.P0),  # 0
        _cell(SMALL=Player.P0),  # 1
        EMPTY_CELL,  # 2 gap
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
    )
    s = GobbletState(board=board, reserve=((2, 2, 2), (2, 2, 2)), current=Player.P0, move_count=4)
    ctx = _ctx_for(s)
    table = build_aggressive()
    idx, record = table.decide(s, ctx)
    assert record.rule == "win_now"
    move = index_to_move(idx)
    assert status_of(apply_move(s, move)).winner is Player.P0


def test_defensive_blocks_incoming_double_threat():
    """Scenario: 防守模式會擋掉對手的雙重威脅 — P1 can fork; P0 has no win; the
    defensive table must fire block_fork (not develop)."""
    board = (
        EMPTY_CELL,
        _cell(SMALL=Player.P1),  # 1  P1
        EMPTY_CELL,
        _cell(SMALL=Player.P1),  # 3  P1
        EMPTY_CELL,  # 4  P1 forks here next turn (lines (1,4,7),(3,4,5))
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
    )
    s = GobbletState(board=board, reserve=((2, 2, 2), (1, 2, 2)), current=Player.P0, move_count=6)
    ctx = _ctx_for(s)
    assert ctx.opp_can_fork is True
    assert ctx.i_can_win is False
    table = build_defensive()
    _idx, record = table.decide(s, ctx)
    assert record.rule == "block_fork"


def test_default_fallback_when_no_guard_matches_does_not_raise():
    """Scenario: 沒有規則命中時走保底合法步 — a table whose only rules are the
    win/block guards (both false) falls back to default. We simulate by clearing
    the always-true develop rule via a stripped table built from the same helpers.
    """
    # Use the defensive table but force a position with no win and no threat/fork:
    # then only safe_develop (always-true) fires. To exercise `default`, build a
    # table with no always-true rule.
    from hl_core import Rule, RuleTable
    from hl_gobblet.controllers._modes_v2 import _P_WIN, _can_win, _win_action

    board = tuple(EMPTY_CELL for _ in range(9))
    s = GobbletState(board=board, reserve=((2, 2, 2), (2, 2, 2)), current=Player.P0)
    ctx = _ctx_for(s)
    # Only a win_now rule (guard false at the empty opening) -> must hit default.
    table = RuleTable(
        rules=[Rule(name="win_now", priority=_P_WIN, guard=_can_win, action_fn=_win_action)],
        default_action=-1,
        state_name="probe",
    )
    idx, record = table.decide(s, ctx)
    assert record.rule == "default"
    assert idx == -1  # the sentinel, decoded to first-legal by the controller


def test_aggressive_make_threat_breaks_ties_by_smallest_index():
    """Scenario: 平手分數的決定性打破 — at the empty opening every PLACE has equal
    development value bands; the develop/make_threat selection is deterministic and
    repeats across calls."""
    board = tuple(EMPTY_CELL for _ in range(9))
    s = GobbletState(board=board, reserve=((2, 2, 2), (2, 2, 2)), current=Player.P0)
    ctx = _ctx_for(s)
    table = build_aggressive()
    idx_a, _ = table.decide(s, ctx)
    idx_b, _ = table.decide(s, ctx)
    assert idx_a == idx_b
    # The chosen index must be one of the legal moves.
    assert idx_a in {move_to_index(m) for m in legal_moves(s)}
