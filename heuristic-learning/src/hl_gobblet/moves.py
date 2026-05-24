"""Move value object and a stable, fixed full-action-space encoding.

Spec: hl-gobblet-env, requirement "合法步生成" (the move<->index part).

Design D2: two move kinds unified behind one frozen `Move`:
  - PLACE(size, to_cell): drop a new piece of `size` from reserve onto `to_cell`.
  - MOVE(from_cell, to_cell): relocate the top piece of `from_cell` to `to_cell`.

The action space is FIXED (does not change with the position), so an index
always decodes to the same Move regardless of board state. Legality is decided
separately by rules.legal_moves; this module only defines the universe of moves
and a bijection move<->index, which is what action masking needs.

Layout (deterministic order):
  indices [0, 27)        PLACE: size * 9 + to_cell           (3 sizes * 9 cells)
  indices [27, 27 + 72)  MOVE:  from_cell * 8 + slot(to)     (9 * 8 ordered pairs)
where `slot(to)` enumerates the 8 destination cells != from_cell in ascending
order, so (from, to) with from != to maps to a unique index.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .state import BOARD_CELLS, Size

_N_SIZES = len(Size)  # 3
_PLACE_COUNT = _N_SIZES * BOARD_CELLS  # 27
_MOVE_PER_FROM = BOARD_CELLS - 1  # 8 destinations per source
_MOVE_COUNT = BOARD_CELLS * _MOVE_PER_FROM  # 72
_ACTION_SPACE_SIZE = _PLACE_COUNT + _MOVE_COUNT  # 99


class MoveKind(Enum):
    PLACE = "place"
    MOVE = "move"


@dataclass(frozen=True)
class Move:
    """A single action. For PLACE, `size` and `to_cell` are set and `from_cell`
    is None. For MOVE, `from_cell` and `to_cell` are set and `size` is None."""

    kind: MoveKind
    to_cell: int
    size: Optional[Size] = None
    from_cell: Optional[int] = None

    @staticmethod
    def place(size: Size, to_cell: int) -> "Move":
        return Move(kind=MoveKind.PLACE, to_cell=to_cell, size=Size(size))

    @staticmethod
    def move(from_cell: int, to_cell: int) -> "Move":
        if from_cell == to_cell:
            raise ValueError("MOVE from_cell must differ from to_cell")
        return Move(kind=MoveKind.MOVE, to_cell=to_cell, from_cell=from_cell)


def action_space_size() -> int:
    """Total number of distinct moves in the fixed action space (constant)."""
    return _ACTION_SPACE_SIZE


def _dest_slot(from_cell: int, to_cell: int) -> int:
    """Position of `to_cell` among the 8 cells != from_cell (ascending order)."""
    return to_cell if to_cell < from_cell else to_cell - 1


def _slot_to_dest(from_cell: int, slot: int) -> int:
    """Inverse of _dest_slot."""
    return slot if slot < from_cell else slot + 1


def move_to_index(move: Move) -> int:
    """Encode a Move to its stable index in [0, action_space_size())."""
    if move.kind is MoveKind.PLACE:
        assert move.size is not None
        return int(move.size) * BOARD_CELLS + move.to_cell
    # MOVE
    assert move.from_cell is not None
    slot = _dest_slot(move.from_cell, move.to_cell)
    return _PLACE_COUNT + move.from_cell * _MOVE_PER_FROM + slot


def index_to_move(index: int) -> Move:
    """Decode an index back to its Move (inverse of move_to_index)."""
    if not 0 <= index < _ACTION_SPACE_SIZE:
        raise ValueError(f"action index {index} out of range [0, {_ACTION_SPACE_SIZE})")
    if index < _PLACE_COUNT:
        size = Size(index // BOARD_CELLS)
        to_cell = index % BOARD_CELLS
        return Move.place(size, to_cell)
    rel = index - _PLACE_COUNT
    from_cell = rel // _MOVE_PER_FROM
    slot = rel % _MOVE_PER_FROM
    to_cell = _slot_to_dest(from_cell, slot)
    return Move.move(from_cell, to_cell)
