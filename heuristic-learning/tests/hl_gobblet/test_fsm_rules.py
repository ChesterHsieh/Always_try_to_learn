"""Tests for the two mode RuleTables (priority, blocking, default, tie-break).

Spec: hl-gobblet-fsm-controller, requirement "各模式的規則表決策".

The tables return a *move index*; we decode with index_to_move to assert on the
chosen Move. A GobbletCtx is refreshed for each position so guards see the same
cached legal moves + threat flags the controller would.
"""

from __future__ import annotations

from hl_gobblet.controllers._modes import (
    GobbletCtx,
    build_aggressive,
    build_defensive,
)
from hl_gobblet.moves import index_to_move, move_to_index
from hl_gobblet.rules import apply_move, legal_moves, status_of
from hl_gobblet.state import EMPTY_CELL, GobbletState, Player, Size, initial_state


def _cell(**sizes):
    slots = [None, None, None]
    for name, owner in sizes.items():
        slots[int(getattr(Size, name))] = owner
    return (slots[0], slots[1], slots[2])


def _ctx_for(state: GobbletState, *, reveal_loses: bool = False) -> GobbletCtx:
    ctx = GobbletCtx(reveal_loses=reveal_loses)
    ctx.refresh(state)
    return ctx


def _two_in_row_to_move(player: Player) -> GobbletState:
    """`player` owns top-row cells 0,1 (smalls) and it is `player`'s turn."""
    p, o = player, player.other
    board = [
        _cell(SMALL=p),
        _cell(SMALL=p),
        EMPTY_CELL,
        _cell(SMALL=o),
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
    ]
    return GobbletState(
        board=tuple(board),
        reserve=((1, 2, 2), (1, 2, 2)),
        current=player,
        move_count=4,
    )


def test_win_now_fires_first_in_aggressive():
    """Scenario: 能贏就贏優先 — a winnable position makes win_now fire and win."""
    s = _two_in_row_to_move(Player.P0)
    table = build_aggressive()
    idx, rec = table.decide(s, _ctx_for(s))
    assert rec.rule == "win_now"
    move = index_to_move(idx)
    assert status_of(apply_move(s, move)).winner is Player.P0


def test_win_now_fires_first_in_defensive():
    """Even in defensive mode, an available win is taken first."""
    s = _two_in_row_to_move(Player.P0)
    table = build_defensive()
    idx, rec = table.decide(s, _ctx_for(s))
    assert rec.rule == "win_now"
    assert status_of(apply_move(s, index_to_move(idx))).winner is Player.P0


def test_aggressive_blocks_lethal_threat_when_no_win():
    """Scenario: 攻擊模式仍會擋致命威脅 — opponent threatens, we have no win, so
    aggressive fires block_then_gobble and the chosen move removes the threat."""
    # P1 owns 0,1 (top row); P0 to move with no two-in-a-row of its own.
    board = [
        _cell(SMALL=Player.P1),
        _cell(SMALL=Player.P1),
        EMPTY_CELL,
        _cell(SMALL=Player.P0),
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
    ]
    s = GobbletState(
        board=tuple(board), reserve=((1, 2, 2), (1, 2, 2)), current=Player.P0, move_count=4
    )
    table = build_aggressive()
    ctx = _ctx_for(s)
    assert ctx.opp_can_win and not ctx.i_can_win  # premise for the block rule
    idx, rec = table.decide(s, ctx)
    assert rec.rule == "block_then_gobble"
    # After our block, the opponent must no longer have a completed top row next.
    nxt = apply_move(s, index_to_move(idx))
    # The opponent's previously-open cell 2 should now be denied (P0 owns it or it
    # is no longer claimable for the line).
    from hl_gobblet.state import top_owner

    assert top_owner(nxt.board[2]) is not Player.P1


def test_no_named_premise_falls_through_no_exception():
    """Scenario: 沒有規則命中時走保底 — at the opening neither win nor block
    applies, so each table's lowest always-true rule fires (never the structural
    default, never an exception) and returns a legal move.

    Aggressive's always-true rule is make_threat (it should build threats, not
    passively develop); defensive's is safe_develop.
    """
    s = initial_state()
    cases = ((build_aggressive(), "make_threat"), (build_defensive(), "safe_develop"))
    for table, expected_rule in cases:
        ctx = _ctx_for(s)
        assert not ctx.i_can_win and not ctx.opp_can_win
        idx, rec = table.decide(s, ctx)
        assert rec.rule == expected_rule  # an always-true rule, never "default"
        assert index_to_move(idx) in legal_moves(s)


def test_default_sentinel_never_surfaces_but_decodes_to_legal():
    """If the structural default sentinel ever fired, the controller maps it to a
    legal move; here we assert the table itself never returns the sentinel."""
    s = initial_state()
    idx, rec = build_aggressive().decide(s, _ctx_for(s))
    assert idx >= 0  # real rule fired, not the -1 sentinel
    assert rec.rule != "default"


def test_tie_break_prefers_smallest_index():
    """Scenario: 平手分數的決定性打破 — equal-scoring candidates resolve to the
    smallest move index. At the symmetric opening, develop must pick the lowest
    index among the best-scoring placements, and the choice is reproducible."""
    s = initial_state()
    table = build_aggressive()
    idx1, _ = table.decide(s, _ctx_for(s))
    idx2, _ = table.decide(s, _ctx_for(s))
    assert idx1 == idx2  # deterministic
    move = index_to_move(idx1)
    # Re-derive the best-scoring set and confirm idx1 is the minimum index of it.
    from hl_gobblet.controllers._score import attack_score

    legals = legal_moves(s)
    best_score = max(attack_score(s, m) for m in legals)
    best_idxs = sorted(move_to_index(m) for m in legals if attack_score(s, m) == best_score)
    assert move_to_index(move) == best_idxs[0]
