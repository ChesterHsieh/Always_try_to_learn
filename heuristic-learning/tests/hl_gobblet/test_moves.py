"""Tests for the Move value object and move<->index encoding (spec: 合法步生成)."""

from __future__ import annotations

import pytest

from hl_gobblet.moves import (
    Move,
    MoveKind,
    action_space_size,
    index_to_move,
    move_to_index,
)
from hl_gobblet.state import BOARD_CELLS, Size


def test_action_space_size_is_fixed_99():
    # 3 sizes * 9 cells (PLACE) + 9 * 8 ordered pairs (MOVE) = 27 + 72 = 99
    assert action_space_size() == 99


def test_every_index_round_trips():
    """Scenario: 動作索引雙向可逆 — decode then encode is identity for all indices."""
    seen = set()
    for i in range(action_space_size()):
        move = index_to_move(i)
        assert move_to_index(move) == i
        seen.add(move)
    # all indices map to distinct moves
    assert len(seen) == action_space_size()


def test_every_move_round_trips():
    """Encode then decode is identity for every constructible legal-shaped move."""
    moves = []
    for size in Size:
        for cell in range(BOARD_CELLS):
            moves.append(Move.place(size, cell))
    for src in range(BOARD_CELLS):
        for dst in range(BOARD_CELLS):
            if src != dst:
                moves.append(Move.move(src, dst))
    assert len(moves) == action_space_size()
    for move in moves:
        assert index_to_move(move_to_index(move)) == move


def test_place_and_move_partition_index_ranges():
    place_indices = [i for i in range(action_space_size()) if index_to_move(i).kind is MoveKind.PLACE]
    move_indices = [i for i in range(action_space_size()) if index_to_move(i).kind is MoveKind.MOVE]
    assert place_indices == list(range(27))
    assert move_indices == list(range(27, 99))


def test_move_with_equal_cells_rejected():
    with pytest.raises(ValueError):
        Move.move(3, 3)


def test_index_out_of_range_rejected():
    with pytest.raises(ValueError):
        index_to_move(action_space_size())
    with pytest.raises(ValueError):
        index_to_move(-1)
