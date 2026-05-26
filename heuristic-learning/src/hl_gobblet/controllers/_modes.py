"""The two named modes as hl_core RuleTables, plus the per-step decision context.

Capability: hl-gobblet-fsm-controller. Each mode is a priority-ordered
hl_core.RuleTable whose guards/actions are pure functions over (GobbletState,
GobbletCtx). action_fns return a *move index* (an int), which is exactly what
hl_core.RuleTable.decide expects (it calls int(action_fn(...))); the controller
decodes the index back to a Move. This lets us reuse hl_core unchanged while the
action space is structured Moves rather than scalar actions.

Modes (readable, named):
  aggressive  win_now -> block_then_gobble -> make_threat -> develop
  defensive   win_now -> block -> safe_develop

Every "parameter" is a named constant (HL red line). The tables hold no state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hl_core import Rule, RuleTable

from ..moves import Move, move_to_index
from ..rules import legal_moves
from ..state import GobbletState
from . import _assess, _score

AGGRESSIVE = "aggressive"
DEFENSIVE = "defensive"

# Priorities (lower fires first).
_P_WIN = 0
_P_BLOCK = 10
_P_THREAT = 20
_P_DEVELOP = 30

# Structural fallback index handed to RuleTable.default_action. It is never the
# real output because each table ends with an always-true develop rule; if it
# ever surfaced, the controller maps it to the first legal move (see fsm_gobblet_v1).
_DEFAULT_SENTINEL = -1


@dataclass
class GobbletCtx:
    """Per-step decision context passed to guards/actions (hl_core ctx slot).

    Mirrors hl_core's ctx duck-type: it carries step_index for trace records and,
    Gobblet-specific, the current legal moves, the cached threat assessment, and
    the active reveal_loses variant so guards/scorers stay pure and fast.
    """

    legal: tuple[Move, ...] = ()
    reveal_loses: bool = False
    i_can_win: bool = False
    opp_can_win: bool = False
    step_index: int = 0
    fsm_state: object | None = None
    _threats: tuple = field(default_factory=tuple)

    def refresh(self, state: GobbletState) -> None:
        """Recompute the cached legal moves and threat flags for `state`."""
        self.legal = legal_moves(state)
        self.i_can_win = _assess.i_can_win(state, reveal_loses=self.reveal_loses)
        self._threats = _assess.opponent_winning_lines(state)
        self.opp_can_win = len(self._threats) > 0


# --- shared action helpers (return a move INDEX) --------------------------------
def _first_legal_index(ctx: GobbletCtx) -> int:
    return move_to_index(ctx.legal[0])


def _win_action(state: GobbletState, ctx: GobbletCtx) -> int:
    wins = _score.winning_moves(state, reveal_loses=ctx.reveal_loses)
    return move_to_index(min(wins, key=move_to_index)) if wins else _first_legal_index(ctx)


def _block_action(state: GobbletState, ctx: GobbletCtx) -> int:
    """Pick the legal move that denies the most opponent threats (tie -> index)."""
    best = _score.best_by(
        state, ctx.legal, _score.blocks_threat_count, reveal_loses=ctx.reveal_loses
    )
    return move_to_index(best)


def _make_threat_action(state: GobbletState, ctx: GobbletCtx) -> int:
    best = _score.best_by(state, ctx.legal, _score.attack_score, reveal_loses=ctx.reveal_loses)
    return move_to_index(best)


def _develop_action(state: GobbletState, ctx: GobbletCtx) -> int:
    best = _score.best_by(state, ctx.legal, _score.develop_score, reveal_loses=ctx.reveal_loses)
    return move_to_index(best)


# --- guards (pure predicates) ---------------------------------------------------
def _can_win(_state: GobbletState, ctx: GobbletCtx) -> bool:
    return ctx.i_can_win


def _must_block(_state: GobbletState, ctx: GobbletCtx) -> bool:
    return ctx.opp_can_win


def _always(_state: GobbletState, _ctx: GobbletCtx) -> bool:
    return True


def build_aggressive() -> RuleTable:
    """Aggressive mode: still self-preserving (blocks a lethal threat) but biased
    toward building our own lines."""
    return RuleTable(
        rules=[
            Rule(name="win_now", priority=_P_WIN, guard=_can_win, action_fn=_win_action),
            Rule(
                name="block_then_gobble",
                priority=_P_BLOCK,
                guard=_must_block,
                action_fn=_block_action,
            ),
            Rule(
                name="make_threat",
                priority=_P_THREAT,
                guard=_always,
                action_fn=_make_threat_action,
            ),
            Rule(
                name="develop", priority=_P_DEVELOP, guard=_always, action_fn=_develop_action
            ),
        ],
        default_action=_DEFAULT_SENTINEL,
        state_name=AGGRESSIVE,
    )


def build_defensive() -> RuleTable:
    """Defensive mode: take a win if offered, otherwise deny the opponent's
    threats, otherwise develop conservatively."""
    return RuleTable(
        rules=[
            Rule(name="win_now", priority=_P_WIN, guard=_can_win, action_fn=_win_action),
            Rule(name="block", priority=_P_BLOCK, guard=_must_block, action_fn=_block_action),
            Rule(
                name="safe_develop",
                priority=_P_DEVELOP,
                guard=_always,
                action_fn=_develop_action,
            ),
        ],
        default_action=_DEFAULT_SENTINEL,
        state_name=DEFENSIVE,
    )
