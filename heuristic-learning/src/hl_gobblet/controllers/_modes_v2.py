"""v2 modes as hl_core RuleTables, plus the v2 per-step decision context.

Capability: hl-gobblet-fsm-controller-v2. Three named modes, each a
priority-ordered hl_core.RuleTable whose guards/actions are pure functions over
(GobbletState, GobbletCtxV2). action_fns return a *move index* (an int), exactly
what hl_core.RuleTable.decide expects; the controller decodes the index back to a
Move.

Modes (readable, named):
  aggressive  win_now -> block_then_gobble -> make_fork -> make_threat -> develop
  defensive   win_now -> block_fork        -> safe_develop
  setup_fork  win_now -> block             -> commit_fork -> develop

Every "parameter" is a named constant (HL red line). The tables hold no state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hl_core import Rule, RuleTable

from ..moves import Move, move_to_index
from ..rules import legal_moves
from ..state import GobbletState
from . import _assess, _assess_v2, _score, _score_v2

AGGRESSIVE = "aggressive"
DEFENSIVE = "defensive"
SETUP_FORK = "setup_fork"

# Priorities (lower fires first).
_P_WIN = 0
_P_BLOCK = 10
_P_FORK = 20
_P_THREAT = 30
_P_DEVELOP = 40

# Structural fallback index handed to RuleTable.default_action; never the real
# output because each table ends with an always-true develop rule.
_DEFAULT_SENTINEL = -1


@dataclass
class GobbletCtxV2:
    """Per-step decision context for v2 (hl_core ctx slot).

    Extends the v1 ctx shape with cached fork flags so the FSM transitions and the
    rule guards can read them without recomputing per rule.
    """

    legal: tuple[Move, ...] = ()
    reveal_loses: bool = False
    i_can_win: bool = False
    opp_can_win: bool = False
    i_can_fork: bool = False
    opp_can_fork: bool = False
    step_index: int = 0
    fsm_state: object | None = None
    _threats: tuple = field(default_factory=tuple)

    def refresh(self, state: GobbletState) -> None:
        """Recompute the cached legal moves, threat flags, and fork flags."""
        self.legal = legal_moves(state)
        self.i_can_win = _assess.i_can_win(state, reveal_loses=self.reveal_loses)
        self._threats = _assess.opponent_winning_lines(state)
        self.opp_can_win = len(self._threats) > 0
        self.i_can_fork = _assess_v2.i_can_fork(state, reveal_loses=self.reveal_loses)
        self.opp_can_fork = _assess_v2.opp_can_fork(state)


# --- shared action helpers (return a move INDEX) --------------------------------
def _first_legal_index(ctx: GobbletCtxV2) -> int:
    return move_to_index(ctx.legal[0])


def _win_action(state: GobbletState, ctx: GobbletCtxV2) -> int:
    wins = _score.winning_moves(state, reveal_loses=ctx.reveal_loses)
    return move_to_index(min(wins, key=move_to_index)) if wins else _first_legal_index(ctx)


def _block_action(state: GobbletState, ctx: GobbletCtxV2) -> int:
    """Pick the legal move that denies the most opponent threats (tie -> index)."""
    best = _score.best_by(
        state, ctx.legal, _score.blocks_threat_count, reveal_loses=ctx.reveal_loses
    )
    return move_to_index(best)


def _block_fork_action(state: GobbletState, ctx: GobbletCtxV2) -> int:
    """Defensive block that also breaks an incoming fork: pick the move minimising
    the opponent's standing threats AND their reachable double threat next turn."""
    best = _score_v2.best_by(state, ctx.legal, _score_v2.block_score, reveal_loses=ctx.reveal_loses)
    return move_to_index(best)


def _make_fork_action(state: GobbletState, ctx: GobbletCtxV2) -> int:
    best = _score_v2.best_by(state, ctx.legal, _score_v2.fork_score, reveal_loses=ctx.reveal_loses)
    return move_to_index(best)


def _make_threat_action(state: GobbletState, ctx: GobbletCtxV2) -> int:
    best = _score_v2.best_by(
        state, ctx.legal, _score_v2.attack_score, reveal_loses=ctx.reveal_loses
    )
    return move_to_index(best)


def _develop_action(state: GobbletState, ctx: GobbletCtxV2) -> int:
    best = _score_v2.best_by(
        state, ctx.legal, _score_v2.develop_score, reveal_loses=ctx.reveal_loses
    )
    return move_to_index(best)


# --- guards (pure predicates) ---------------------------------------------------
def _can_win(_state: GobbletState, ctx: GobbletCtxV2) -> bool:
    return ctx.i_can_win


def _must_block(_state: GobbletState, ctx: GobbletCtxV2) -> bool:
    return ctx.opp_can_win


def _should_block_fork(_state: GobbletState, ctx: GobbletCtxV2) -> bool:
    """Defensive block fires on a single one-move threat OR an incoming fork."""
    return ctx.opp_can_win or ctx.opp_can_fork


def _can_fork(_state: GobbletState, ctx: GobbletCtxV2) -> bool:
    return ctx.i_can_fork


def _always(_state: GobbletState, _ctx: GobbletCtxV2) -> bool:
    return True


def build_aggressive() -> RuleTable:
    """Aggressive: win, else block a lethal threat, else build a fork, else make a
    threat, else develop."""
    return RuleTable(
        rules=[
            Rule(name="win_now", priority=_P_WIN, guard=_can_win, action_fn=_win_action),
            Rule(
                name="block_then_gobble",
                priority=_P_BLOCK,
                guard=_must_block,
                action_fn=_block_action,
            ),
            Rule(name="make_fork", priority=_P_FORK, guard=_can_fork, action_fn=_make_fork_action),
            Rule(
                name="make_threat",
                priority=_P_THREAT,
                guard=_always,
                action_fn=_make_threat_action,
            ),
            Rule(name="develop", priority=_P_DEVELOP, guard=_always, action_fn=_develop_action),
        ],
        default_action=_DEFAULT_SENTINEL,
        state_name=AGGRESSIVE,
    )


def build_defensive() -> RuleTable:
    """Defensive: win if offered, else deny the opponent's threats AND incoming
    forks (block_fork), else develop conservatively."""
    return RuleTable(
        rules=[
            Rule(name="win_now", priority=_P_WIN, guard=_can_win, action_fn=_win_action),
            Rule(
                name="block_fork",
                priority=_P_BLOCK,
                guard=_should_block_fork,
                action_fn=_block_fork_action,
            ),
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


def build_setup_fork() -> RuleTable:
    """setup_fork: win if offered, else stay safe (block any live threat), else
    commit the forking move, else develop."""
    return RuleTable(
        rules=[
            Rule(name="win_now", priority=_P_WIN, guard=_can_win, action_fn=_win_action),
            Rule(name="block", priority=_P_BLOCK, guard=_must_block, action_fn=_block_action),
            Rule(
                name="commit_fork",
                priority=_P_FORK,
                guard=_can_fork,
                action_fn=_make_fork_action,
            ),
            Rule(name="develop", priority=_P_DEVELOP, guard=_always, action_fn=_develop_action),
        ],
        default_action=_DEFAULT_SENTINEL,
        state_name=SETUP_FORK,
    )
