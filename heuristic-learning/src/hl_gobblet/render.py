"""Pure rendering helpers: GobbletState + last move -> text / rich renderables.

Spec: hl-gobblet-env, requirement "CLI 對戰觀戰器" (the rendering part).

NO I/O here. Functions return strings or rich renderables; the viewer script owns
all printing and refresh timing. Keeping the "who gobbled whom / what was
revealed" logic in pure functions lets test_render.py assert on text snapshots
without a terminal (Design D6).

Symbols: each cell shows "<SIZE><owner>" — size by letter (S/M/L) and owner by
digit (0/1), e.g. "L0" = P0's large, "M1" = P1's medium. The owner digit makes
the two sides distinguishable even without colour; the rich view additionally
puts P0 on a blue background and P1 on a red background. A trailing '*' marks a
piece hidden underneath, e.g. "L0*".
"""

from __future__ import annotations

from typing import Optional

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .moves import Move, MoveKind
from .state import Cell, GobbletState, Player, Size, top_owner, top_size

_SIZE_LETTER = {Size.SMALL: "S", Size.MEDIUM: "M", Size.LARGE: "L"}
_OWNER_NUM = {Player.P0: "0", Player.P1: "1"}
# Dual encoding: a strong background block PLUS a readable owner number, so the
# two sides are distinguishable even when colour is stripped (pipes, no-TTY).
_OWNER_CELL_STYLE = {Player.P0: "bold white on blue", Player.P1: "bold white on red"}
_OWNER_TEXT_STYLE = {Player.P0: "bold blue", Player.P1: "bold red"}


def _has_hidden_piece(cell: Cell) -> bool:
    """True if anything sits below the top piece of this cell."""
    t = top_size(cell)
    if t is None:
        return False
    return any(cell[s] is not None for s in Size if s < t)


def cell_label(cell: Cell) -> str:
    """Plain-text label for a cell's top piece, '·' for empty.

    Format is "<SIZE><owner>" — e.g. "L0" = P0's large, "M1" = P1's medium. The
    trailing owner digit (0/1) discriminates the two sides without relying on
    colour. A trailing '*' marks a piece hidden underneath.
    """
    owner = top_owner(cell)
    if owner is None:
        return "·"
    size = top_size(cell)
    assert size is not None
    label = _SIZE_LETTER[size] + _OWNER_NUM[owner]
    if _has_hidden_piece(cell):
        label += "*"
    return label


def board_text(state: GobbletState) -> str:
    """A 3x3 plain-text board (rows separated by newlines), cells space-padded."""
    rows = []
    for r in range(3):
        cells = [cell_label(state.board[r * 3 + c]) for c in range(3)]
        rows.append(" ".join(f"{c:<2}" for c in cells).rstrip())
    return "\n".join(rows)


def reserve_text(state: GobbletState) -> str:
    """Two lines: each player's remaining off-board pieces by size."""
    lines = []
    for player in (Player.P0, Player.P1):
        parts = []
        for size in Size:
            parts += [_SIZE_LETTER[size]] * state.reserve[player][size]
        tag = f"P{_OWNER_NUM[player]}"
        lines.append(f"reserve {tag}: {' '.join(parts) if parts else '(none)'}")
    return "\n".join(lines)


def _cell_name(idx: int) -> str:
    return f"c{idx}"


def move_description(
    state_before: GobbletState,
    move: Optional[Move],
) -> str:
    """Human-readable description of `move` evaluated against the pre-move state.

    For a MOVE that uncovers a piece, the description names what was revealed
    underneath the source cell.
    """
    if move is None:
        return "(game start)"
    mover = "P0" if state_before.current is Player.P0 else "P1"
    if move.kind is MoveKind.PLACE:
        assert move.size is not None
        return f"{mover} PLACE {_SIZE_LETTER[move.size]} -> {_cell_name(move.to_cell)}"
    # MOVE: describe source, destination, and what (if anything) is revealed.
    assert move.from_cell is not None
    src_cell = state_before.board[move.from_cell]
    top = top_size(src_cell)
    revealed = None
    if top is not None:
        for s in (Size.LARGE, Size.MEDIUM, Size.SMALL):
            if s < top and src_cell[s] is not None:
                owner = src_cell[s]
                revealed = f"{_SIZE_LETTER[s]}({'P0' if owner is Player.P0 else 'P1'})"
                break
    desc = f"{mover} MOVE {_cell_name(move.from_cell)} -> {_cell_name(move.to_cell)}"
    if revealed is not None:
        desc += f" (revealed {revealed})"
    return desc


# --- rich renderables (thin wrappers over the text helpers) -------------------


def _rich_cell(cell: Cell) -> Text:
    """A coloured cell block: P0 on a blue background, P1 on a red background.

    The label still carries the owner digit so the block reads correctly even if
    the terminal drops the background colour.
    """
    owner = top_owner(cell)
    if owner is None:
        return Text(" ·  ", style="dim")
    label = cell_label(cell)  # e.g. "L0", "M1*"
    # Pad to a fixed 4-wide field so the background block is a tidy rectangle.
    return Text(f" {label:<3}", style=_OWNER_CELL_STYLE[owner])


def board_table(state: GobbletState) -> Table:
    """rich.Table rendering of the 3x3 board (coloured background by owner)."""
    table = Table(show_header=False, show_edge=True, pad_edge=False, padding=0)
    for _ in range(3):
        table.add_column(justify="center", width=4)
    for r in range(3):
        table.add_row(*[_rich_cell(state.board[r * 3 + c]) for c in range(3)])
    return table


def render_panel(
    state: GobbletState,
    state_before: Optional[GobbletState],
    last_move: Optional[Move],
    *,
    title: str = "Gobblet Gobblers",
    result: Optional[str] = None,
) -> Panel:
    """Full rich panel: legend + board + reserves + last-move line (+ result)."""
    before = state_before if state_before is not None else state
    legend = Text.assemble(
        ("  P0  ", _OWNER_CELL_STYLE[Player.P0]),
        ("  vs  ", "dim"),
        ("  P1  ", _OWNER_CELL_STYLE[Player.P1]),
        ("   L/M/S = large/medium/small, * = piece hidden underneath", "dim"),
    )
    body = Group(
        legend,
        Text(""),
        board_table(state),
        Text(""),
        Text(reserve_text(state)),
        Text(""),
        Text(f"last: {move_description(before, last_move)}", style="yellow"),
        *([Text(f"result: {result}", style="bold green")] if result else []),
    )
    return Panel(body, title=title, expand=False)
