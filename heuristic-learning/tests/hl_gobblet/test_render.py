"""Tests for the pure render helpers (spec: CLI 對戰觀戰器, rendering part)."""

from __future__ import annotations

from hl_gobblet.moves import Move
from hl_gobblet.render import (
    board_text,
    cell_label,
    move_description,
    reserve_text,
)
from hl_gobblet.state import EMPTY_CELL, GobbletState, Player, Size


def _state_with_gobble() -> GobbletState:
    """cell 0: P1 small under a P0 large (top = P0, hidden piece below)."""
    board = [EMPTY_CELL] * 9
    board[0] = (Player.P1, None, Player.P0)  # P1 small underneath, P0 large on top
    return GobbletState(
        board=tuple(board),
        reserve=((2, 2, 1), (2, 2, 2)),
        current=Player.P1,
        move_count=2,
    )


def test_cell_label_shows_owner_size_and_hidden_marker():
    """Scenario: 渲染含疊放與被吃的局面.

    P0 large over P1 small -> top is "L0" (P0's large) with a hidden-piece marker.
    """
    s = _state_with_gobble()
    assert cell_label(s.board[0]) == "L0*"  # L0 = P0 large, * = piece hidden below
    assert cell_label(s.board[1]) == "·"  # empty


def test_cell_label_distinguishes_owners_without_case():
    """The owner digit (0/1) discriminates sides regardless of colour."""
    board = [EMPTY_CELL] * 9
    board[0] = (None, None, Player.P0)  # P0 large
    board[1] = (None, None, Player.P1)  # P1 large
    s = GobbletState(board=tuple(board), reserve=((2, 2, 1), (2, 2, 1)),
                     current=Player.P0, move_count=2)
    assert cell_label(s.board[0]) == "L0"  # P0
    assert cell_label(s.board[1]) == "L1"  # P1


def test_board_text_marks_gobbled_cell_for_coverer():
    s = _state_with_gobble()
    text = board_text(s)
    first_row = text.splitlines()[0]
    assert "L0*" in first_row  # the gobbled cell is shown as P0's large + hidden


def test_reserve_text_lists_remaining_pieces():
    s = _state_with_gobble()
    rtext = reserve_text(s)
    assert "reserve P0: S S M M L" in rtext  # P0 used one large
    assert "reserve P1: S S M M L L" in rtext


def test_move_description_reports_reveal():
    """Scenario: 顯示移動揭露了什麼.

    Moving the P0 large off cell 0 reveals the P1 small underneath; the
    description must say so.
    """
    before = _state_with_gobble().with_(current=Player.P0)  # P0 to move its large
    move = Move.move(0, 5)
    desc = move_description(before, move)
    assert "P0 MOVE c0 -> c5" in desc
    assert "revealed S(P1)" in desc


def test_move_description_place_and_start():
    s = _state_with_gobble().with_(current=Player.P0)
    assert move_description(s, Move.place(Size.MEDIUM, 4)) == "P0 PLACE M -> c4"
    assert move_description(s, None) == "(game start)"
