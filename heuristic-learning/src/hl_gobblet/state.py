"""Immutable board state for Gobblet Gobblers.

Spec: hl-gobblet-env, requirement "盤面狀態模型".

Design D1: each of the 9 cells is a fixed-length size-indexed stack — a tuple of
length 3 where index = Size value (0=SMALL, 1=MEDIUM, 2=LARGE) holding the owner
(None / P0 / P1) of the piece occupying that size slot on that cell. Because
pieces are strictly size-increasing as you stack, this representation can never
encode an illegal stack (a small on top of a large), and the "top" of a cell is
just the largest occupied size. Reserves track how many of each size each player
still holds off the board. Everything is frozen + tuple-based, so applying a move
returns a brand-new state and never mutates the original.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Optional

BOARD_CELLS = 9  # 3x3
PIECES_PER_SIZE = 2  # each player starts with 2 of each size


class Player(IntEnum):
    P0 = 0
    P1 = 1

    @property
    def other(self) -> "Player":
        return Player.P1 if self is Player.P0 else Player.P0


class Size(IntEnum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2


# One cell = owners stacked by size; index by Size. None means that size is empty.
Cell = tuple[Optional[Player], Optional[Player], Optional[Player]]
EMPTY_CELL: Cell = (None, None, None)


@dataclass(frozen=True)
class GobbletState:
    """A complete, immutable Gobblet Gobblers position.

    board:   9 cells, each a size-indexed owner tuple (row-major, cell 0..8).
    reserve: reserve[player][size] = count of that size still off the board.
    current: whose turn it is.
    move_count: number of plies played so far (used for the draw cap).
    """

    board: tuple[Cell, ...]
    reserve: tuple[tuple[int, int, int], tuple[int, int, int]]
    current: Player
    move_count: int = 0

    def with_(
        self,
        *,
        board: Optional[tuple[Cell, ...]] = None,
        reserve: Optional[tuple[tuple[int, int, int], tuple[int, int, int]]] = None,
        current: Optional[Player] = None,
        move_count: Optional[int] = None,
    ) -> "GobbletState":
        """Return a copy with selected fields replaced (never mutates self)."""
        return replace(
            self,
            board=self.board if board is None else board,
            reserve=self.reserve if reserve is None else reserve,
            current=self.current if current is None else current,
            move_count=self.move_count if move_count is None else move_count,
        )


def initial_state(seed: int = 0) -> GobbletState:
    """Build the opening position.

    Gobblet's opening is fixed (empty board, full reserves, P0 to move), so the
    seed does not change it — it is accepted only to match the env reset(seed)
    contract and keep call sites uniform.
    """
    del seed  # opening position is deterministic; seed kept for API symmetry
    full = (PIECES_PER_SIZE, PIECES_PER_SIZE, PIECES_PER_SIZE)
    return GobbletState(
        board=tuple(EMPTY_CELL for _ in range(BOARD_CELLS)),
        reserve=(full, full),
        current=Player.P0,
        move_count=0,
    )


def top_size(cell: Cell) -> Optional[Size]:
    """Largest occupied size on a cell, or None if the cell is empty."""
    for size in (Size.LARGE, Size.MEDIUM, Size.SMALL):
        if cell[size] is not None:
            return size
    return None


def top_owner(cell: Cell) -> Optional[Player]:
    """Owner of the topmost (largest) piece on a cell, or None if empty."""
    size = top_size(cell)
    return None if size is None else cell[size]
