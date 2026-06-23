"""v2 candidate-move scoring: v1's potential plus fork bonus and gobble tactics.

Capability: hl-gobblet-fsm-controller-v2. Every score is computed from the state
that *results* from applying a candidate move (via the immutable apply_move), so
the scorers are exact and side-effect-free. Selection is deterministic: ties are
broken by ascending move_to_index.

Beyond v1's line potential / centre / reserve / exposure terms (reused unchanged
from _score), v2 adds three Gobblet-specific terms — all named constants (HL red
line: no learned parameters):

  _FORK_BONUS    rewards a move that leaves us with >= 2 claimable winning lines
                 (a double threat the opponent cannot all block next turn).
  _GOBBLE_VALUE  rewards covering an opponent top so that one of the opponent's
                 standing one-move threats is removed.
  _REVEAL_PEN    penalises a MOVE whose source-lift reveals a piece that hands the
                 opponent an immediate one-move win (the gobble downside).
"""

from __future__ import annotations

from ..moves import Move, MoveKind, move_to_index
from ..rules import LINES, apply_move
from ..state import GobbletState, Player, top_owner
from . import _score
from ._assess import opponent_winning_lines
from ._assess_v2 import claimable_winning_lines

# --- v2 scoring weights (named constants) ---------------------------------------
_FORK_BONUS = 25  # a move leaving us with >= 2 claimable winning lines
_GOBBLE_VALUE = 8  # per opponent one-move threat removed by covering their top
_REVEAL_PEN = 40  # a MOVE that lifts to reveal an opponent's immediate win
_BLOCK_FORK_PEN = 30  # per claimable winning line the opponent still STANDS on after our defence
_GIVE_FORK_PEN = 35  # per claimable winning line we leave the opponent on the post-move board
_TWO_OPEN_LINE = 4  # an UNcontested line where we own exactly two tops (a live threat)
_ONE_OPEN_LINE = 1  # an uncontested line where we own exactly one top (a seed)
_PRESSURE_W = 3  # weight on (our pressure - opponent pressure) after the move

_FORK_LINE_THRESHOLD = 2


def _uncontested_pressure(board: tuple, player: Player) -> int:
    """Positional pressure for `player`: sum over lines with NO opponent top of a
    weight by how many tops `player` already owns (two = a live threat, one = a
    seed toward a future fork). Lines blocked by an opponent top score nothing.

    This is a denser positional signal than v1's line potential: it directly
    rewards spreading ownership across many still-open lines, which is what builds
    toward the double threats v1 does not plan for.
    """
    opponent = player.other
    total = 0
    for a, b, c in LINES:
        owners = (top_owner(board[a]), top_owner(board[b]), top_owner(board[c]))
        if any(o is opponent for o in owners):
            continue
        mine = sum(1 for o in owners if o is player)
        if mine == 2:
            total += _TWO_OPEN_LINE
        elif mine == 1:
            total += _ONE_OPEN_LINE
    return total


def _pressure_diff(nxt: GobbletState, me: Player) -> int:
    """(_PRESSURE_W) * (our uncontested pressure - the opponent's) on the post-move
    board. Positive when our move builds more multi-line pressure than it gives the
    opponent — the core 'win the development race to a fork' signal."""
    mine = _uncontested_pressure(nxt.board, me)
    theirs = _uncontested_pressure(nxt.board, me.other)
    return _PRESSURE_W * (mine - theirs)


def _fork_bonus(nxt: GobbletState, me) -> int:
    """+_FORK_BONUS if, after the move, `me` holds two or more claimable lines."""
    if len(claimable_winning_lines(nxt, me)) >= _FORK_LINE_THRESHOLD:
        return _FORK_BONUS
    return 0


def _gobble_value(state: GobbletState, move: Move, *, reveal_loses: bool) -> int:
    """Reward covering an opponent top that removes standing opponent threats.

    Counts how many of the opponent's one-move winning lines (from the v1
    standing-threat model) are removed by this move, scaled by _GOBBLE_VALUE.
    Applies to any move that lands on a cell currently topped by the opponent
    (i.e. a real gobble); a non-covering move scores 0 here and is judged by the
    line-potential terms instead.
    """
    target_top = top_owner(state.board[move.to_cell])
    if target_top is not state.current.other:
        return 0  # not covering an opponent top -> not a gobble
    before = len(opponent_winning_lines(state))
    nxt = apply_move(state, move, reveal_loses=reveal_loses)
    # On nxt the opponent is to move; their standing threats are this side's
    # `opponent_winning_lines` evaluated relative to the (now) side-not-to-move,
    # which is us. Recompute the OPPONENT's claimable lines explicitly instead.
    after = len(claimable_winning_lines(nxt, state.current.other))
    removed = max(0, before - after)
    return _GOBBLE_VALUE * removed


def _reveal_penalty(state: GobbletState, move: Move) -> int:
    """-_REVEAL_PEN if this MOVE lifts a piece and the resulting position lets the
    opponent win immediately (the lift revealed an opponent line or freed a path).

    Evaluated under reveal_loses=True so the lift instant itself is judged: a MOVE
    that self-destructs under the touch-move variant, or that completes our move
    but leaves the opponent a one-move win on a line we just uncovered, is a
    tactical blunder we want to avoid even when the variant is off.
    """
    if move.kind is not MoveKind.MOVE:
        return 0
    # Judge the lift instant explicitly via the touch-move variant.
    nxt = apply_move(state, move, reveal_loses=True)
    # If the touch-move lift already loses for us, that surfaces as the opponent
    # winning on nxt.
    from ..rules import status_of

    st = status_of(nxt)
    if st.done and st.winner is state.current.other:
        return -_REVEAL_PEN
    # Otherwise check whether, after our completed move, the opponent now has a
    # claimable winning line they did not have before (we uncovered it).
    before = len(claimable_winning_lines(state, state.current.other))
    after = len(claimable_winning_lines(nxt, state.current.other))
    if after > before:
        return -_REVEAL_PEN
    return 0


def _opponent_standing_danger(nxt: GobbletState) -> int:
    """Penalty for the standing danger our move leaves on the post-move board.

    Strictly one ply: `nxt` is the board AFTER our single move. We read the
    opponent's standing claimable winning lines on that static board (the same
    one-move-ahead model v1 uses for threats) and penalise by how many they hold —
    two or more is a fork we walked into. We do NOT enumerate the opponent's reply
    (that would be a second ply / tree search, forbidden by the HL red line).
    """
    opponent = nxt.current  # on `nxt` (after our move) the opponent is to move
    standing = len(claimable_winning_lines(nxt, opponent))
    return -_GIVE_FORK_PEN * standing


def attack_score(state: GobbletState, move: Move, *, reveal_loses: bool = False) -> int:
    """Aggressive-mode score (higher = better): v1 attack potential plus the v2
    fork bonus, gobble value, reveal penalty, and standing-danger penalty."""
    nxt = apply_move(state, move, reveal_loses=reveal_loses)
    me = state.current
    score = _score.attack_score(state, move, reveal_loses=reveal_loses)
    score += _fork_bonus(nxt, me)
    score += _gobble_value(state, move, reveal_loses=reveal_loses)
    score += _reveal_penalty(state, move)
    score += _opponent_standing_danger(nxt)
    score += _pressure_diff(nxt, me)
    return score


def develop_score(state: GobbletState, move: Move, *, reveal_loses: bool = False) -> int:
    """Conservative development score: v1 develop potential plus a (smaller) fork
    bonus, the reveal penalty, and the standing-danger penalty. No gobble incentive
    (development should not fling pieces onto opponent tops)."""
    nxt = apply_move(state, move, reveal_loses=reveal_loses)
    me = state.current
    score = _score.develop_score(state, move, reveal_loses=reveal_loses)
    score += _fork_bonus(nxt, me)
    score += _reveal_penalty(state, move)
    score += _opponent_standing_danger(nxt)
    score += _pressure_diff(nxt, me)
    return score


def fork_score(state: GobbletState, move: Move, *, reveal_loses: bool = False) -> int:
    """Score for the setup_fork mode: maximise reaching a double threat, then fall
    back to attack potential. A move that achieves the fork dominates."""
    nxt = apply_move(state, move, reveal_loses=reveal_loses)
    me = state.current
    score = _fork_bonus(nxt, me)
    score += _score.attack_score(state, move, reveal_loses=reveal_loses)
    score += _reveal_penalty(state, move)
    score += _opponent_standing_danger(nxt)
    return score


def block_score(state: GobbletState, move: Move, *, reveal_loses: bool = False) -> int:
    """Defensive score (higher = better): deny the opponent's standing threats and
    minimise the opponent's standing claimable lines left on the post-move board.

    Strictly one ply: we combine v1's `blocks_threat_count` (standing one-move
    threats removed) with a penalty proportional to how many claimable winning
    lines the opponent still STANDS on after our single move — driving the table
    toward the move that occupies/covers the shared cell of an incoming fork,
    rather than just the lowest index. No opponent-reply enumeration (HL red line).
    """
    nxt = apply_move(state, move, reveal_loses=reveal_loses)
    removed = _score.blocks_threat_count(state, move, reveal_loses=reveal_loses)
    opp_standing = len(claimable_winning_lines(nxt, nxt.current))
    score = _GOBBLE_VALUE * removed
    score -= _BLOCK_FORK_PEN * opp_standing
    score += _reveal_penalty(state, move)
    # Tie-break nudge toward our own development so equal-defence moves still build.
    score += _score.develop_score(state, move, reveal_loses=reveal_loses)
    return score


def best_by(state, candidates, scorer, *, reveal_loses: bool = False) -> Move:
    """Pick the highest-scoring candidate, ties broken by ascending index.

    Thin wrapper over _score.best_by so v2 selection stays identical in tie-break
    semantics to v1 (deterministic key (-score, move_to_index)).
    """
    if not candidates:
        raise ValueError("best_by called with no candidates")
    return min(
        candidates,
        key=lambda m: (-scorer(state, m, reveal_loses=reveal_loses), move_to_index(m)),
    )
