"""FsmGobbletV2 — an iterated, gradient-free Gobblet opponent built to beat v1.

Capability: hl-gobblet-fsm-controller-v2. Implements the Gobblet opponent
contract (`reset(seed)` + `act(state) -> Move`) using hl_core's
FiniteStateMachine + RuleTable building blocks, unchanged. It iterates on
FsmGobbletV1 by adding fork (double-threat) awareness and gobble tactics, and a
third named mode:

  aggressive  (initial) — build our own lines, set up forks, gobble to remove
              threats; still blocks a lethal opponent threat first.
  defensive             — when the opponent is one move from winning OR one move
              from a fork and we have no win, deny it first.
  setup_fork            — when there is no live threat and we can reach a double
              threat in one move, commit to the forking move.

Transitions (spec "FSM 模式與切換條件（含 setup_fork）"):
  -> aggressive  when i_can_win                                   (so win_now fires)
  -> defensive   when (opp_can_win or opp_can_fork) and not i_can_win
  -> setup_fork  when not (opp threat) and not i_can_win and i_can_fork
  -> aggressive  otherwise (no threat, no fork)

HL red line: no gradients, no neural nets, no tree search — only a single shallow
"win in one / opponent wins (or forks) next / I can fork in one" assessment.
Every parameter is a named constant.
"""

from __future__ import annotations

from dataclasses import dataclass

from hl_core import FiniteStateMachine, Transition

from ..moves import Move, index_to_move, move_to_index
from ..state import GobbletState
from ._modes_v2 import (
    AGGRESSIVE,
    DEFENSIVE,
    SETUP_FORK,
    GobbletCtxV2,
    build_aggressive,
    build_defensive,
    build_setup_fork,
)


@dataclass(frozen=True)
class GobbletTraceRecordV2:
    """One immutable per-ply decision record (semantic fields only, for golden trace)."""

    step: int
    state: str  # "aggressive" | "defensive" | "setup_fork"
    rule: str  # fired rule name, or "default"
    action_index: int  # move_to_index of the chosen move
    move_kind: str  # Move.kind.value
    to_cell: int
    size: int | None  # PLACE only
    from_cell: int | None  # MOVE only


def _opp_threatens(_o, c: GobbletCtxV2) -> bool:
    return c.opp_can_win or c.opp_can_fork


def _build_fsm() -> FiniteStateMachine:
    """Three states, each bound to its mode RuleTable; transitions on the threat
    and fork flags. Order matters: i_can_win wins outright, then threats force
    defence, then a reachable fork enters setup_fork, else back to aggressive."""
    # A win always routes to aggressive (win_now fires there). From any state we
    # express this as edges into aggressive guarded by i_can_win.
    to_aggressive_on_win = [
        Transition(src=src, dst=AGGRESSIVE, condition=lambda _o, c: c.i_can_win)
        for src in (DEFENSIVE, SETUP_FORK)
    ]
    # Threat (single or fork) with no win of our own -> defend.
    to_defensive = [
        Transition(
            src=src,
            dst=DEFENSIVE,
            condition=lambda o, c: _opp_threatens(o, c) and not c.i_can_win,
        )
        for src in (AGGRESSIVE, SETUP_FORK)
    ]
    # No live threat, no win, but we can fork in one move -> set it up.
    to_setup_fork = [
        Transition(
            src=src,
            dst=SETUP_FORK,
            condition=lambda o, c: not _opp_threatens(o, c)
            and not c.i_can_win
            and c.i_can_fork,
        )
        for src in (AGGRESSIVE, DEFENSIVE)
    ]
    # Threat resolved and no fork to set up -> resume aggression.
    from_defensive_to_aggressive = [
        Transition(
            src=DEFENSIVE,
            dst=AGGRESSIVE,
            condition=lambda o, c: not _opp_threatens(o, c) and not c.i_can_fork,
        )
    ]
    from_setup_fork_to_aggressive = [
        Transition(
            src=SETUP_FORK,
            dst=AGGRESSIVE,
            condition=lambda o, c: not _opp_threatens(o, c) and not c.i_can_fork,
        )
    ]
    transitions = (
        to_aggressive_on_win
        + to_defensive
        + to_setup_fork
        + from_defensive_to_aggressive
        + from_setup_fork_to_aggressive
    )
    return FiniteStateMachine(
        initial_state=AGGRESSIVE,
        deciders={
            AGGRESSIVE: build_aggressive(),
            DEFENSIVE: build_defensive(),
            SETUP_FORK: build_setup_fork(),
        },
        transitions=transitions,
    )


class FsmGobbletV2:
    """Iterated heuristic Gobblet opponent over an hl_core FSM of three RuleTables."""

    def __init__(self, *, reveal_loses: bool = False) -> None:
        self._fsm = _build_fsm()
        self._reveal_loses = reveal_loses
        self._fsm_state = self._fsm.new_state()
        self._ctx = GobbletCtxV2(reveal_loses=reveal_loses, fsm_state=self._fsm_state)
        self._trace: list[GobbletTraceRecordV2] = []

    def reset(self, seed: int) -> None:
        """Reset to the initial mode and clear the decision trace (seed kept for
        API symmetry; this controller is deterministic, so it is not consumed)."""
        del seed
        self._fsm_state = self._fsm.new_state()  # current = AGGRESSIVE, buffers cleared
        self._ctx = GobbletCtxV2(reveal_loses=self._reveal_loses, fsm_state=self._fsm_state)
        self._trace = []

    def act(self, state: GobbletState) -> Move:
        """Pick a legal move for state.current. Never mutates `state`."""
        self._ctx.refresh(state)
        if not self._ctx.legal:
            raise ValueError("FsmGobbletV2.act called with no legal moves")

        index, record = self._fsm.step(state, self._ctx, self._fsm_state)
        move = self._decode(index)

        self._trace.append(
            GobbletTraceRecordV2(
                step=self._ctx.step_index,
                state=record.state,
                rule=record.rule,
                action_index=move_to_index(move),
                move_kind=move.kind.value,
                to_cell=move.to_cell,
                size=None if move.size is None else int(move.size),
                from_cell=move.from_cell,
            )
        )
        self._ctx.step_index += 1
        return move

    def decision_trace(self) -> tuple[GobbletTraceRecordV2, ...]:
        """Immutable snapshot of the per-ply decision records (read-only)."""
        return tuple(self._trace)

    def _decode(self, index: int) -> Move:
        """Decode a returned move index to a Move; map the structural default
        sentinel (-1) to the first legal move so act() always returns a legal Move."""
        if index < 0:
            return self._ctx.legal[0]
        move = index_to_move(index)
        if move not in self._ctx.legal:
            return self._ctx.legal[0]
        return move
