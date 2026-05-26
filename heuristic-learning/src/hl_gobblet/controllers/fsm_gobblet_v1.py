"""FsmGobbletV1 — a gradient-free Gobblet opponent assembled from hl_core.

Capability: hl-gobblet-fsm-controller. Implements the Gobblet opponent contract
(`reset(seed)` + `act(state) -> Move`) using hl_core's FiniteStateMachine +
RuleTable building blocks, unchanged. It switches between two readable named
modes via a one-ply threat assessment:

  aggressive  (initial) — build our own lines / gobble to make threats; still
              blocks a lethal opponent threat when we have no win of our own.
  defensive             — when the opponent is one move from winning and we are
              not, deny the threat first.

Transitions (spec "FSM 兩模式與切換條件"):
  -> defensive  when opp_can_win and not i_can_win
  -> aggressive when i_can_win or not opp_can_win

The hl_core RuleTable action space is integer; here each action_fn returns a
*move index*, which act() decodes back to a Move. A read-only decision_trace()
records, per ply: step, mode name, fired rule, and the chosen Move (+ its index).

HL red line: no gradients, no neural nets, no tree search — only a single shallow
"win in one / opponent wins next" assessment. Every parameter is a named constant.
"""

from __future__ import annotations

from dataclasses import dataclass

from hl_core import FiniteStateMachine, Transition

from ..moves import Move, index_to_move, move_to_index
from ..state import GobbletState
from ._modes import AGGRESSIVE, DEFENSIVE, GobbletCtx, build_aggressive, build_defensive


@dataclass(frozen=True)
class GobbletTraceRecord:
    """One immutable per-ply decision record (semantic fields only, for golden trace)."""

    step: int
    state: str  # "aggressive" | "defensive"
    rule: str  # fired rule name, or "default"
    action_index: int  # move_to_index of the chosen move
    move_kind: str  # Move.kind.value
    to_cell: int
    size: int | None  # PLACE only
    from_cell: int | None  # MOVE only


def _build_fsm() -> FiniteStateMachine:
    """Two states, each bound to its mode RuleTable; transitions on the threat flags."""
    return FiniteStateMachine(
        initial_state=AGGRESSIVE,
        deciders={AGGRESSIVE: build_aggressive(), DEFENSIVE: build_defensive()},
        transitions=[
            # Opponent is one move from winning and we have no win -> defend.
            Transition(
                src=AGGRESSIVE,
                dst=DEFENSIVE,
                condition=lambda _o, c: c.opp_can_win and not c.i_can_win,
            ),
            # We can win, or there is no live threat -> attack.
            Transition(
                src=DEFENSIVE,
                dst=AGGRESSIVE,
                condition=lambda _o, c: c.i_can_win or not c.opp_can_win,
            ),
        ],
    )


class FsmGobbletV1:
    """Heuristic-style Gobblet opponent over an hl_core FSM of two RuleTables."""

    def __init__(self, *, reveal_loses: bool = False) -> None:
        self._fsm = _build_fsm()
        self._reveal_loses = reveal_loses
        self._fsm_state = self._fsm.new_state()
        self._ctx = GobbletCtx(reveal_loses=reveal_loses, fsm_state=self._fsm_state)
        self._trace: list[GobbletTraceRecord] = []

    def reset(self, seed: int) -> None:
        """Reset to the initial mode and clear the decision trace (seed kept for
        API symmetry; this controller is deterministic, so it is not consumed)."""
        del seed
        self._fsm_state = self._fsm.new_state()  # current = AGGRESSIVE, macro buffers cleared
        self._ctx = GobbletCtx(reveal_loses=self._reveal_loses, fsm_state=self._fsm_state)
        self._trace = []

    def act(self, state: GobbletState) -> Move:
        """Pick a legal move for state.current. Never mutates `state`."""
        # Refresh the cached legal moves + threat assessment for THIS position.
        self._ctx.refresh(state)
        if not self._ctx.legal:
            raise ValueError("FsmGobbletV1.act called with no legal moves")

        index, record = self._fsm.step(state, self._ctx, self._fsm_state)
        move = self._decode(index)

        self._trace.append(
            GobbletTraceRecord(
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

    def decision_trace(self) -> tuple[GobbletTraceRecord, ...]:
        """Immutable snapshot of the per-ply decision records (read-only)."""
        return tuple(self._trace)

    def _decode(self, index: int) -> Move:
        """Decode a returned move index to a Move; map the structural default
        sentinel (-1) to the first legal move so act() always returns a legal Move."""
        if index < 0:
            return self._ctx.legal[0]
        move = index_to_move(index)
        # Defensive guarantee: the chosen move must be legal in this position.
        if move not in self._ctx.legal:
            return self._ctx.legal[0]
        return move
