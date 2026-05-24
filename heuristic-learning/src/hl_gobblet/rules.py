"""Rules engine: legal-move generation, pure-function transitions, win detection.

Spec: hl-gobblet-env, requirements "合法步生成", "回合推進", "勝負判定（官方規則）",
"reveal_loses 進階變體".

All transitions are pure: apply_move(state, move) returns a new GobbletState and
never mutates its input (Design D3). Win detection uses the topmost owner of each
cell. Two timing modes:
  - Official rules (default): a line is only checked AFTER the move completes.
  - reveal_loses variant: on a MOVE, the instant the source piece is lifted (top
    removed, destination not yet changed) is checked first; if that instant
    reveals the opponent already lined up three, the lifting player loses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .moves import Move, MoveKind
from .state import (
    BOARD_CELLS,
    Cell,
    GobbletState,
    Player,
    Size,
    top_owner,
    top_size,
)

DEFAULT_MAX_MOVES = 60  # ply cap; reaching it without a line is a draw

# The eight winning lines (row-major cell indices 0..8).
LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),  # rows
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),  # cols
    (0, 4, 8),
    (2, 4, 6),  # diagonals
)


@dataclass(frozen=True)
class Status:
    """Outcome of a position after a move."""

    done: bool
    winner: Optional[Player] = None  # None winner + done => draw
    draw: bool = False


def _can_cover(cell: Cell, size: Size) -> bool:
    """True if a piece of `size` may land on `cell` (empty or strictly smaller top)."""
    t = top_size(cell)
    return t is None or t < size


def legal_moves(state: GobbletState) -> tuple[Move, ...]:
    """All legal moves for state.current, in a deterministic order.

    Order: PLACE moves first (size ascending, then cell ascending), then MOVE
    moves (from-cell ascending, then to-cell ascending). Deterministic ordering
    makes opponents and tests reproducible.
    """
    player = state.current
    moves: list[Move] = []

    # PLACE: a size the player still has in reserve, onto a coverable cell.
    for size in Size:
        if state.reserve[player][size] <= 0:
            continue
        for cell_idx in range(BOARD_CELLS):
            if _can_cover(state.board[cell_idx], size):
                moves.append(Move.place(size, cell_idx))

    # MOVE: the player's own top piece relocated onto another coverable cell.
    for src in range(BOARD_CELLS):
        src_cell = state.board[src]
        if top_owner(src_cell) is not player:
            continue
        src_size = top_size(src_cell)
        assert src_size is not None
        for dst in range(BOARD_CELLS):
            if dst == src:
                continue
            if _can_cover(state.board[dst], src_size):
                moves.append(Move.move(src, dst))

    return tuple(moves)


def line_winner(board: tuple[Cell, ...]) -> Optional[Player]:
    """Return the player owning a full line by top piece, or None."""
    for a, b, c in LINES:
        owner = top_owner(board[a])
        if owner is not None and top_owner(board[b]) is owner and top_owner(board[c]) is owner:
            return owner
    return None


def _set_top(cell: Cell, size: Size, owner: Optional[Player]) -> Cell:
    """Return a copy of `cell` with the `size` slot set to `owner`."""
    lst = list(cell)
    lst[size] = owner
    return (lst[0], lst[1], lst[2])


def _place_on_board(
    board: tuple[Cell, ...], cell_idx: int, size: Size, owner: Player
) -> tuple[Cell, ...]:
    new_cell = _set_top(board[cell_idx], size, owner)
    return board[:cell_idx] + (new_cell,) + board[cell_idx + 1 :]


def _lift_from_board(board: tuple[Cell, ...], cell_idx: int) -> tuple[tuple[Cell, ...], Size]:
    """Remove the top piece of cell_idx; return (new board, lifted size)."""
    src_cell = board[cell_idx]
    size = top_size(src_cell)
    if size is None:
        raise ValueError(f"cannot lift from empty cell {cell_idx}")
    new_cell = _set_top(src_cell, size, None)
    new_board = board[:cell_idx] + (new_cell,) + board[cell_idx + 1 :]
    return new_board, size


def status_of(state: GobbletState, max_moves: int = DEFAULT_MAX_MOVES) -> Status:
    """Outcome of a (post-move) position: win, draw at the ply cap, or ongoing."""
    winner = line_winner(state.board)
    if winner is not None:
        return Status(done=True, winner=winner)
    if state.move_count >= max_moves:
        return Status(done=True, winner=None, draw=True)
    return Status(done=False)


def apply_move(
    state: GobbletState,
    move: Move,
    *,
    reveal_loses: bool = False,
) -> GobbletState:
    """Apply a legal move and return the resulting state (never mutates input).

    Raises ValueError for any move not in legal_moves(state). On a MOVE with
    reveal_loses=True, the lift instant is annotated on the returned state via a
    sentinel: if lifting reveals the opponent's line, current player loses — this
    is surfaced by status carrying winner=opponent. We encode that by completing
    the move normally but, when the lift instant is a loss, the move_count is
    advanced and the board is left so that status_of reports the opponent's win.
    """
    if move not in legal_moves(state):
        raise ValueError(f"illegal move {move!r} for current position")

    player = state.current

    if move.kind is MoveKind.PLACE:
        assert move.size is not None
        size = move.size
        new_board = _place_on_board(state.board, move.to_cell, size, player)
        new_reserve = _decrement_reserve(state.reserve, player, size)
        return state.with_(
            board=new_board,
            reserve=new_reserve,
            current=player.other,
            move_count=state.move_count + 1,
        )

    # MOVE
    assert move.from_cell is not None
    lifted_board, size = _lift_from_board(state.board, move.from_cell)

    if reveal_loses:
        # The instant the piece is lifted: destination unchanged, source revealed.
        revealed_winner = line_winner(lifted_board)
        if revealed_winner is not None and revealed_winner is not player:
            # Lifting exposed the opponent's three-in-a-row -> lifting player loses.
            # Freeze the lifted board so status_of reports the opponent as winner;
            # advance the ply count and hand the (now decided) game to the opponent.
            return state.with_(
                board=lifted_board,
                current=player.other,
                move_count=state.move_count + 1,
            )

    final_board = _place_on_board(lifted_board, move.to_cell, size, player)
    return state.with_(
        board=final_board,
        current=player.other,
        move_count=state.move_count + 1,
    )


def _decrement_reserve(
    reserve: tuple[tuple[int, int, int], tuple[int, int, int]],
    player: Player,
    size: Size,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    counts = list(reserve[player])
    counts[size] -= 1
    new_player_reserve = (counts[0], counts[1], counts[2])
    if player is Player.P0:
        return (new_player_reserve, reserve[1])
    return (reserve[0], new_player_reserve)
