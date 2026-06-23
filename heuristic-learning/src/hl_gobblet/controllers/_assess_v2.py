"""v2 threat-assessment: one-ply fork (double-threat) detection.

Capability: hl-gobblet-fsm-controller-v2. These pure functions extend the v1
assessment (`i_can_win`/`opp_can_win`) with *fork* detection — the move that
leaves the mover with two simultaneous one-move winning threats, which the
opponent cannot all block in a single reply. They never mutate the state they
inspect; they look at most one ply ahead via the immutable `apply_move`, and they
do NOT expand the opponent's reply tree (HL red line: a single shallow ply only).

Definitions:
  claimable_winning_lines(state, who) — lines where `who` owns two tops and can
                                        legally claim the third next turn. This is
                                        the v1 LineThreat shape generalised to an
                                        arbitrary player (v1's opponent_winning_lines
                                        is exactly this for state.current.other).
  i_can_fork(state, reveal_loses)     — can the side to move reach (in one legal
                                        move) a position with >= 2 of their own
                                        claimable winning lines?
  opp_can_fork(state)                 — can the opponent, on their own next turn,
                                        reach >= 2 of their claimable winning lines?

The ">= 2 claimable lines == unstoppable" reading is a fast, deterministic
conservative proxy, not a proof; the win-rate gate against v1 validates it is
strong enough in practice.
"""

from __future__ import annotations

from ..rules import LINES, apply_move, legal_moves
from ..state import GobbletState, Player, top_owner
from ._assess import LineThreat, _opponent_can_claim

# A fork is "two or more simultaneous claimable winning lines" (the opponent
# cannot block them all in one reply). Named constant per the HL red line.
_FORK_LINE_THRESHOLD = 2


def claimable_winning_lines(state: GobbletState, who: Player) -> tuple[LineThreat, ...]:
    """Lines where `who` owns exactly two tops and can legally claim the third.

    Deterministic order (LINES order). Reuses v1's `_opponent_can_claim`, which is
    already parameterised by player, so the claim legality matches the v1 threat
    model exactly. Read-only: never mutates `state`.
    """
    threats: list[LineThreat] = []
    for line in LINES:
        owners = [top_owner(state.board[c]) for c in line]
        if sum(1 for o in owners if o is who) != 2:
            continue
        gap_cells = [c for c, o in zip(line, owners) if o is not who]
        if len(gap_cells) != 1:
            continue
        gap = gap_cells[0]
        if _opponent_can_claim(state, gap, who):
            threats.append(LineThreat(line=line, cell=gap))
    return tuple(threats)


def _count_claimable(state: GobbletState, who: Player) -> int:
    return len(claimable_winning_lines(state, who))


def i_can_fork(state: GobbletState, *, reveal_loses: bool = False) -> bool:
    """True if the side to move has a legal move after which it holds >= 2
    claimable winning lines (a double threat the opponent cannot all block).

    One shallow ply only: we apply each of our legal moves (immutably) and count
    our claimable lines on the resulting board. No opponent reply is simulated.
    """
    me = state.current
    for move in legal_moves(state):
        nxt = apply_move(state, move, reveal_loses=reveal_loses)
        if _count_claimable(nxt, me) >= _FORK_LINE_THRESHOLD:
            return True
    return False


def opp_can_fork(state: GobbletState) -> bool:
    """True if the opponent (side NOT to move) can, on their own next turn, reach
    a position with >= 2 of their claimable winning lines.

    We model "the opponent's next turn" as the position after our (null) pass,
    i.e. a board identical to `state` but with the opponent to move, then apply
    each of the opponent's legal moves. This is still a single opponent ply — no
    deeper tree (HL red line). Read-only over `state`.
    """
    opponent = state.current.other
    # Hand the move to the opponent without changing the board (a hypothetical
    # "what can they do next"); never mutates the input state.
    opp_to_move = state.with_(current=opponent)
    for move in legal_moves(opp_to_move):
        nxt = apply_move(opp_to_move, move)
        if _count_claimable(nxt, opponent) >= _FORK_LINE_THRESHOLD:
            return True
    return False
