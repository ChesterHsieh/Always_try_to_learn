"""Tests for legal moves, transitions, and official win detection.

Spec: 合法步生成, 回合推進, 勝負判定（官方規則）.
"""

from __future__ import annotations

import pytest

from hl_gobblet.moves import Move
from hl_gobblet.rules import (
    apply_move,
    legal_moves,
    line_winner,
    status_of,
)
from hl_gobblet.state import Player, Size, initial_state, top_owner


def test_place_on_empty_is_legal():
    """Scenario: 放子到空格 — placing a medium from reserve onto an empty cell."""
    s = initial_state()
    assert Move.place(Size.MEDIUM, 4) in legal_moves(s)


def test_larger_can_gobble_smaller_either_owner():
    """Scenario: 大子吃小子 — a larger piece may cover a smaller one (any owner)."""
    s = initial_state()
    s = apply_move(s, Move.place(Size.SMALL, 0))  # P0 small on cell 0
    # P1 to move: P1 may place a medium/large over P0's small on cell 0.
    moves = legal_moves(s)
    assert Move.place(Size.MEDIUM, 0) in moves
    assert Move.place(Size.LARGE, 0) in moves


def test_cannot_cover_equal_or_larger():
    """Scenario: 不可疊放於同size或更大子之上."""
    s = initial_state()
    s = apply_move(s, Move.place(Size.MEDIUM, 0))  # P0 medium on cell 0
    moves = legal_moves(s)  # P1 to move
    assert Move.place(Size.SMALL, 0) not in moves  # smaller can't cover
    assert Move.place(Size.MEDIUM, 0) not in moves  # equal can't cover
    assert Move.place(Size.LARGE, 0) in moves  # larger can


def test_move_reveals_piece_underneath():
    """Scenario: 移動大子揭露底下的子.

    Build: P0 small @0, then P1 large gobbles it @0. Later P1 relocates that
    large to an empty cell, which must re-expose the P0 small underneath.
    """
    s = initial_state()
    s = apply_move(s, Move.place(Size.SMALL, 0))  # P0 small @0   (P1 to move)
    s = apply_move(s, Move.place(Size.LARGE, 0))  # P1 large over P0 small @0 (P0)
    assert top_owner(s.board[0]) is Player.P1
    assert s.board[0][Size.SMALL] is Player.P0  # P0 small hidden underneath

    s = apply_move(s, Move.place(Size.SMALL, 1))  # P0 filler @1  (P1 to move)
    # P1 relocates the large from cell 0 to empty cell 5, revealing P0 small.
    s = apply_move(s, Move.move(0, 5))
    assert top_owner(s.board[0]) is Player.P0  # revealed P0 small
    assert top_owner(s.board[5]) is Player.P1  # large landed on 5


def test_apply_illegal_move_rejected():
    """Scenario: 拒絕非法動作."""
    s = initial_state()
    s = apply_move(s, Move.place(Size.MEDIUM, 0))  # P0 medium @0
    # P1 to move: placing a small on cell 0 is illegal (can't cover medium).
    with pytest.raises(ValueError):
        apply_move(s, Move.place(Size.SMALL, 0))


def test_three_in_a_row_wins():
    """Scenario: 連成三線獲勝 — P0 completes the top row."""
    s = initial_state()
    s = apply_move(s, Move.place(Size.SMALL, 0))  # P0 @0
    s = apply_move(s, Move.place(Size.SMALL, 3))  # P1 @3
    s = apply_move(s, Move.place(Size.SMALL, 1))  # P0 @1
    s = apply_move(s, Move.place(Size.SMALL, 4))  # P1 @4
    s = apply_move(s, Move.place(Size.MEDIUM, 2))  # P0 @2 -> top row 0,1,2
    assert line_winner(s.board) is Player.P0
    st = status_of(s)
    assert st.done and st.winner is Player.P0 and not st.draw


def test_max_moves_draw():
    """Scenario: 達步數上限平局."""
    s = initial_state()
    # Fabricate a non-winning position at the cap.
    s = s.with_(move_count=4)
    assert not status_of(s, max_moves=10).done
    s = s.with_(move_count=10)
    st = status_of(s, max_moves=10)
    assert st.done and st.draw and st.winner is None
