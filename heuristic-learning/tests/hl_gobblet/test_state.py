"""Tests for the Gobblet state model (spec: 盤面狀態模型)."""

from __future__ import annotations

from hl_gobblet.moves import Move
from hl_gobblet.rules import apply_move
from hl_gobblet.state import (
    BOARD_CELLS,
    EMPTY_CELL,
    Player,
    Size,
    initial_state,
    top_owner,
)


def test_initial_state_is_empty_full_reserve_p0_to_move():
    """Scenario: 初始局面 — empty board, 2 of each size each side, P0 to move."""
    s = initial_state(seed=0)
    assert len(s.board) == BOARD_CELLS
    assert all(cell == EMPTY_CELL for cell in s.board)
    assert s.reserve == ((2, 2, 2), (2, 2, 2))
    assert s.current is Player.P0
    assert s.move_count == 0


def test_apply_move_does_not_mutate_original():
    """Scenario: 狀態不可變 — applying a move returns a new object, original intact."""
    s0 = initial_state()
    board_before = s0.board
    reserve_before = s0.reserve

    s1 = apply_move(s0, Move.place(Size.SMALL, 0))

    assert s1 is not s0
    # original is untouched
    assert s0.board is board_before
    assert s0.board[0] == EMPTY_CELL
    assert s0.reserve == reserve_before
    assert s0.current is Player.P0
    # new state reflects the move
    assert top_owner(s1.board[0]) is Player.P0
    assert s1.current is Player.P1


def test_top_owner_after_gobble_belongs_to_coverer():
    """Scenario: 疊放後最上層決定歸屬 — P0 large over P1 small -> cell is P0's."""
    s = initial_state()
    s = apply_move(s, Move.place(Size.SMALL, 4))  # P0 small on center
    # Move a P1 piece elsewhere is needed to keep turns; instead place P1 small
    # somewhere then have P0 gobble. Simplest: P1 places small on cell 0,
    # then P0 places large over P1's small on cell 0.
    s = apply_move(s, Move.place(Size.SMALL, 0))  # P1 small on cell 0
    s = apply_move(s, Move.place(Size.LARGE, 0))  # P0 large over P1 small on cell 0

    cell = s.board[0]
    assert cell[Size.SMALL] is Player.P1  # P1 small still recorded underneath
    assert cell[Size.LARGE] is Player.P0  # P0 large on top
    assert top_owner(cell) is Player.P0  # cell counts for P0 in line detection
