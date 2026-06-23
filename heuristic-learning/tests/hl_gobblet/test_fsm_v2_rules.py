"""Tests for the v2 candidate-move scoring (fork bonus + gobble tactics).

Spec: hl-gobblet-fsm-controller-v2, requirements "gobble 戰術得失評估" and
"各模式的規則表決策"(平手分數的決定性打破). Asserts: covering an opponent top to
remove a threat scores higher; a MOVE that reveals an opponent line is penalised;
a forking move wins on the fork bonus; ties resolve to the smallest index.
"""

from __future__ import annotations

from hl_gobblet.controllers import _score_v2
from hl_gobblet.moves import Move, move_to_index
from hl_gobblet.state import EMPTY_CELL, GobbletState, Player, Size


def _cell(**sizes):
    slots = [None, None, None]
    for name, owner in sizes.items():
        slots[int(getattr(Size, name))] = owner
    return (slots[0], slots[1], slots[2])


def test_gobble_removing_threat_scores_higher_than_idle_develop():
    """Scenario: 覆蓋對手大子解除威脅獲得加分 — P1 owns 0,1 (threatening row
    (0,1,2)); P0 to move. Covering cell 0 with a larger piece removes the threat
    and must score higher than placing a piece on an idle empty cell."""
    board = (
        _cell(SMALL=Player.P1),  # 0  P1 (part of threat (0,1,2))
        _cell(SMALL=Player.P1),  # 1  P1 (part of threat (0,1,2))
        EMPTY_CELL,  # 2  the gap P1 would claim
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
    )
    s = GobbletState(
        board=board,
        reserve=((2, 2, 2), (0, 2, 2)),
        current=Player.P0,
        move_count=4,
    )
    gobble = Move.place(Size.MEDIUM, 0)  # cover P1's small on cell 0 -> kills threat
    idle = Move.place(Size.SMALL, 8)  # harmless corner, removes no threat
    assert _score_v2.attack_score(s, gobble) > _score_v2.attack_score(s, idle)


def test_move_revealing_opponent_line_is_penalised():
    """Scenario: 搬移後揭露對手線的步被懲罰 — P0 has a large on cell 2 hiding P1's
    small there; lifting it (MOVE 2->x) reveals P1's completed row (0,1,2) and
    loses under the touch-move judgement, so it must score well below a safe move."""
    board = (
        _cell(SMALL=Player.P1),  # 0  P1
        _cell(SMALL=Player.P1),  # 1  P1
        _cell(SMALL=Player.P1, LARGE=Player.P0),  # 2  P1 small hidden under P0 large
        EMPTY_CELL,
        _cell(SMALL=Player.P0),  # 4  a safe P0 piece to relocate instead
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
    )
    s = GobbletState(
        board=board,
        reserve=((1, 1, 0), (0, 2, 2)),
        current=Player.P0,
        move_count=8,
    )
    reveal_move = Move.move(2, 3)  # lifting cell 2 reveals P1's row (0,1,2)
    safe_move = Move.move(4, 3)  # relocating the centre piece reveals nothing
    assert _score_v2.attack_score(s, reveal_move) < _score_v2.attack_score(s, safe_move)


def test_forking_move_beats_single_threat_move_via_fork_bonus():
    """Scenario: 造 fork 的步因 _FORK_BONUS 勝出 — placing on the shared centre
    creates two claimable lines; it must outscore a move that makes only one."""
    board = (
        EMPTY_CELL,
        _cell(SMALL=Player.P0),  # 1  P0
        EMPTY_CELL,
        _cell(SMALL=Player.P0),  # 3  P0
        EMPTY_CELL,  # 4  fork point: lines (1,4,7) and (3,4,5)
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
        EMPTY_CELL,
    )
    s = GobbletState(
        board=board,
        reserve=((2, 2, 2), (2, 2, 2)),
        current=Player.P0,
        move_count=6,
    )
    fork = Move.place(Size.SMALL, 4)  # two claimable lines after this
    single = Move.place(Size.SMALL, 8)  # corner, no double threat
    assert _score_v2.attack_score(s, fork) > _score_v2.attack_score(s, single)


def test_best_by_breaks_ties_by_smallest_index():
    """Scenario: 平手取最小索引 — with a constant scorer, best_by picks the move
    with the smallest move_to_index."""
    board = tuple(EMPTY_CELL for _ in range(9))
    s = GobbletState(board=board, reserve=((2, 2, 2), (2, 2, 2)), current=Player.P0)
    from hl_gobblet.rules import legal_moves

    cands = legal_moves(s)

    def flat(_state, _move, *, reveal_loses=False):
        return 0

    chosen = _score_v2.best_by(s, cands, flat)
    assert move_to_index(chosen) == min(move_to_index(m) for m in cands)
