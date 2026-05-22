"""FiniteStateMachine — named states, each bound to a Decider sub-policy.

Spec: hl-procedural-policy, requirement "有限狀態機（Finite State Machine）".

Design D2 (composition over inheritance): a state's sub-policy is any Decider —
a RuleTable or a macro wrapper. step() first applies the transition table from
the current state, then delegates to the (possibly new) state's Decider. The
current state is a readable named variable held in the external FsmState
(Design D1) so reset() restores initial_state and clears macro progress in one
place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .macros import MacroState
from .trace import Decider, TraceRecord

Condition = Callable[[Any, Any], bool]


@dataclass(frozen=True)
class Transition:
    """A directed edge: when in `src` and `condition` holds, move to `dst`."""

    src: str
    dst: str
    condition: Condition


class FsmState:
    """Mutable per-episode FSM progress: current state + each state's MacroState."""

    def __init__(self, initial_state: str, macro_states: dict[str, MacroState]) -> None:
        self._initial = initial_state
        self.current = initial_state
        self.macro_states = macro_states

    def reset(self) -> None:
        self.current = self._initial
        for ms in self.macro_states.values():
            ms.reset()


class FiniteStateMachine:
    """An immutable FSM config; per-episode progress lives in FsmState."""

    def __init__(
        self,
        initial_state: str,
        deciders: dict[str, Decider],
        transitions: list[Transition],
    ) -> None:
        if initial_state not in deciders:
            raise ValueError(f"initial_state {initial_state!r} has no bound decider")
        self.initial_state = initial_state
        self.deciders = deciders
        self.transitions = tuple(transitions)

    def new_state(self) -> FsmState:
        """Create fresh per-episode progress, one MacroState per state name."""
        macro_states = {name: MacroState() for name in self.deciders}
        return FsmState(self.initial_state, macro_states)

    def step(self, observation: Any, ctx: Any, state: FsmState) -> tuple[int, TraceRecord]:
        for t in self.transitions:
            if t.src == state.current and t.condition(observation, ctx):
                state.current = t.dst
                self._on_enter(t.dst, ctx)
                break
        decider = self.deciders[state.current]
        return decider.decide(observation, ctx)

    def _on_enter(self, state_name: str, ctx: Any) -> None:
        """Notify a destination decider it was just entered (e.g. arm a macro)."""
        decider = self.deciders[state_name]
        on_enter = getattr(decider, "on_enter", None)
        if callable(on_enter):
            on_enter(ctx)
