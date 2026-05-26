"""Candidate-move scoring pure functions for the rule tables.

Capability: hl-gobblet-fsm-controller. Every score is computed from the state
that *results* from applying a candidate move (via the immutable apply_move), so
the scorers are exact and side-effect-free. Selection is deterministic: ties are
broken by ascending move_to_index, so the same position always yields the same
choice.

All weights are named constants (HL red line: no learned parameters).
"""

from __future__ import annotations

from ..moves import Move, move_to_index
from ..rules import LINES, apply_move
from ..state import GobbletState, Player, Size, top_owner, top_size

# --- scoring weights (named constants) ------------------------------------------
_WIN_LINE = 100  # a line we already completed (should be caught by win_now anyway)
_TWO_LINE = 10  # a line where we own two tops and the third is open/coverable
_ONE_LINE = 1  # a line where we own exactly one top and the rest is open
_CENTER_BONUS = 3  # occupying the centre cell (4) controls four lines
_RESERVE_KEEP = 1  # prefer keeping larger reserve pieces in hand (per large/medium kept)
_EXPOSE_PENALTY = 2  # penalty per own large piece sitting on the board (reversible exposure)

_CENTER_CELL = 4


def _line_potential(board: tuple, player: Player) -> int:
    """Heuristic potential of `player`'s top ownership across all eight lines.

    Sums per-line value: a line is only scored while it is still winnable for the
    player (no enemy top blocking all three cells in a way that can't be covered
    is approximated by 'no opponent top present' — a fast, deterministic proxy).
    A line with two player tops and no opponent top scores _TWO_LINE; with one,
    _ONE_LINE. Two such two-tops lines means a double threat and naturally sums.
    """
    opponent = player.other
    total = 0
    for a, b, c in LINES:
        owners = (top_owner(board[a]), top_owner(board[b]), top_owner(board[c]))
        if any(o is opponent for o in owners):
            continue  # opponent occupies the line: not a clean threat for us
        mine = sum(1 for o in owners if o is player)
        if mine == 3:
            total += _WIN_LINE
        elif mine == 2:
            total += _TWO_LINE
        elif mine == 1:
            total += _ONE_LINE
    return total


def _reserve_value(state: GobbletState, player: Player) -> int:
    """Reward keeping flexible (larger) pieces in reserve; small reserve is cheap."""
    res = state.reserve[player]
    return _RESERVE_KEEP * (res[Size.MEDIUM] + res[Size.LARGE])


def _exposure_penalty(board: tuple, player: Player) -> int:
    """Penalise own large pieces left on the board (they can be lifted/relocated
    by us but also reveal whatever they cover)."""
    penalty = 0
    for cell in board:
        if top_owner(cell) is player and top_size(cell) is Size.LARGE:
            penalty += _EXPOSE_PENALTY
    return penalty


def attack_score(state: GobbletState, move: Move, *, reveal_loses: bool = False) -> int:
    """Score a candidate move for the aggressive mode (higher = better).

    Combines our own line potential after the move, centre control, and a small
    reserve-keeping reward, minus large-piece exposure. Computed on the resulting
    position so it reflects real consequences.
    """
    nxt = apply_move(state, move, reveal_loses=reveal_loses)
    player = state.current
    score = _line_potential(nxt.board, player)
    if top_owner(nxt.board[_CENTER_CELL]) is player:
        score += _CENTER_BONUS
    score += _reserve_value(nxt, player)
    score -= _exposure_penalty(nxt.board, player)
    return score


def develop_score(state: GobbletState, move: Move, *, reveal_loses: bool = False) -> int:
    """Score a candidate move for safe development (conservative).

    Like attack_score but weights exposure avoidance and reserve-keeping more, so
    the controller does not fling large pieces onto the board early where they can
    be relocated to reveal lines.
    """
    nxt = apply_move(state, move, reveal_loses=reveal_loses)
    player = state.current
    score = _line_potential(nxt.board, player)
    if top_owner(nxt.board[_CENTER_CELL]) is player:
        score += _CENTER_BONUS
    score += 2 * _reserve_value(nxt, player)
    score -= 2 * _exposure_penalty(nxt.board, player)
    return score


def blocks_threat_count(state: GobbletState, move: Move, *, reveal_loses: bool = False) -> int:
    """How many opponent one-move threats this candidate removes.

    Compares the opponent's standing winning lines before and after the move; a
    move that denies more threats scores higher. Imported lazily to avoid a cycle
    with _assess (which imports nothing from here).
    """
    from ._assess import opponent_winning_lines

    before = len(opponent_winning_lines(state))
    nxt = apply_move(state, move, reveal_loses=reveal_loses)
    # After our move it is the opponent's turn; their threats are now assessed from
    # the opponent's perspective. opponent_winning_lines is defined relative to the
    # side NOT to move, so on `nxt` (opponent to move) it reports OUR threats. We
    # instead recompute the opponent's standing threats explicitly on nxt.board.
    after = _standing_threats_for(nxt, state.current.other)
    return max(0, before - after)


def _standing_threats_for(state: GobbletState, who: Player) -> int:
    """Count lines where `who` owns two tops and the third is empty/coverable by
    anyone (a conservative 'still dangerous' proxy used after our move)."""
    count = 0
    for a, b, c in LINES:
        owners = (top_owner(state.board[a]), top_owner(state.board[b]), top_owner(state.board[c]))
        if sum(1 for o in owners if o is who) != 2:
            continue
        gap = [cell for cell, o in zip((a, b, c), owners) if o is not who]
        if len(gap) != 1:
            continue
        # coverable by `who` if empty or top strictly smaller than some piece `who`
        # could place; approximate as 'not already blocked by a large of the other side'.
        gap_cell = state.board[gap[0]]
        if top_size(gap_cell) is Size.LARGE and top_owner(gap_cell) is not who:
            continue
        count += 1
    return count


def best_by(
    state: GobbletState,
    candidates: tuple[Move, ...],
    scorer,
    *,
    reveal_loses: bool = False,
) -> Move:
    """Pick the highest-scoring candidate, breaking ties by ascending index.

    `candidates` MUST be non-empty and all legal. Deterministic: the key is
    (-score, move_to_index), so equal scores always resolve to the smallest index.
    """
    if not candidates:
        raise ValueError("best_by called with no candidates")
    return min(
        candidates,
        key=lambda m: (-scorer(state, m, reveal_loses=reveal_loses), move_to_index(m)),
    )


def winning_moves(
    state: GobbletState, *, reveal_loses: bool = False
) -> tuple[Move, ...]:
    """All legal moves that immediately win for the side to move (deterministic)."""
    from ..rules import legal_moves, status_of

    player = state.current
    out = []
    for move in legal_moves(state):
        st = status_of(apply_move(state, move, reveal_loses=reveal_loses))
        if st.done and st.winner is player:
            out.append(move)
    return tuple(out)
