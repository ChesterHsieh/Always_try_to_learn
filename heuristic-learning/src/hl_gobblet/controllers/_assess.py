"""Threat-assessment pure functions used by the FSM mode switch.

Capability: hl-gobblet-fsm-controller. These functions never mutate the state
they inspect — they only read it and, where they look one ply ahead, do so via
the immutable `apply_move`. They power the FSM transition between `aggressive`
and `defensive` and are reused inside the rule tables.

Definitions:
  i_can_win(state, reveal_loses)  — does the side to move have a move that wins
                                    immediately (under the active variant)?
  opponent_winning_lines(state)   — lines the opponent is one legal top-claim
                                    away from completing, with the empty-or-
                                    coverable cell that must be denied.
  opp_can_win(state)              — is there at least one such line?

These are intentionally a single shallow ply (no tree search): "can I win in one
move" is exact via apply_move; "can the opponent win next" is a fast standing-
threat heuristic that does not simulate the opponent's reply tree (HL red line).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..rules import LINES, apply_move, legal_moves, status_of
from ..state import GobbletState, Player, Size, top_owner, top_size


def i_can_win(state: GobbletState, *, reveal_loses: bool = False) -> bool:
    """True if state.current has a legal move that ends the game in their favour."""
    player = state.current
    for move in legal_moves(state):
        nxt = apply_move(state, move, reveal_loses=reveal_loses)
        st = status_of(nxt)
        if st.done and st.winner is player:
            return True
    return False


def _opponent_can_claim(state: GobbletState, cell_idx: int, opponent: Player) -> bool:
    """True if `opponent` could, on their own turn, make `cell_idx`'s top theirs.

    Mirrors the legality of legal_moves but evaluated for `opponent` (who is NOT
    state.current): either PLACE a reserved size that covers the cell's current
    top, or MOVE one of their own top pieces (strictly larger than the cell's
    top) from another cell onto it. We never need the cell already theirs — that
    is handled by the two-of-three line check before calling this.
    """
    cell = state.board[cell_idx]
    cur_top = top_size(cell)

    # PLACE: a reserved size strictly larger than the current top (or any size if empty).
    for size in Size:
        if state.reserve[opponent][size] <= 0:
            continue
        if cur_top is None or cur_top < size:
            return True

    # MOVE: an opponent-owned top piece elsewhere that is large enough to cover.
    for src in range(len(state.board)):
        if src == cell_idx:
            continue
        if top_owner(state.board[src]) is not opponent:
            continue
        src_size = top_size(state.board[src])
        assert src_size is not None
        if cur_top is None or cur_top < src_size:
            return True

    return False


@dataclass(frozen=True)
class LineThreat:
    """A line the opponent is one legal claim away from completing.

    line       the three cell indices.
    cell       the single cell that must be denied (the one not yet owned by the
               opponent, which the opponent can legally claim next turn).
    """

    line: tuple[int, int, int]
    cell: int


def opponent_winning_lines(state: GobbletState) -> tuple[LineThreat, ...]:
    """Lines where the opponent owns two tops and can legally claim the third.

    The "opponent" is state.current.other (the side NOT to move). Deterministic
    order: lines in LINES order. Only lines with exactly one non-opponent cell
    that the opponent can claim next turn are returned.
    """
    opponent = state.current.other
    threats: list[LineThreat] = []
    for line in LINES:
        owners = [top_owner(state.board[c]) for c in line]
        opp_count = sum(1 for o in owners if o is opponent)
        if opp_count != 2:
            continue
        # The single cell not owned by the opponent.
        gap_cells = [c for c, o in zip(line, owners) if o is not opponent]
        if len(gap_cells) != 1:
            continue
        gap = gap_cells[0]
        if _opponent_can_claim(state, gap, opponent):
            threats.append(LineThreat(line=line, cell=gap))
    return tuple(threats)


def opp_can_win(state: GobbletState) -> bool:
    """True if the opponent has at least one one-move winning threat."""
    return len(opponent_winning_lines(state)) > 0
