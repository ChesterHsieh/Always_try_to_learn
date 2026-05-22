"""ProceduralPolicy — assembles rule tables / FSM / macros behind HeuristicPolicy.

Spec: hl-procedural-policy, requirement "ProceduralPolicy 組合介面與決策軌跡導出".

Implements the existing HeuristicPolicy contract (reset / act) and nothing more —
no update/learn/train. Per-episode execution progress (FSM current state, macro
cursors, step counter) lives in a single resettable runtime object cleared by
reset(seed), so a given seed reproduces the same trajectory (Design D1).

Each step produces a (action, TraceRecord) in the same call (Design D3); the
record is appended to a read-only buffer and only the action is returned to the
runner. decision_trace() returns an immutable tuple snapshot, so exporting it can
never change behaviour and the caller cannot mutate the buffer.

NOTE: this module imports only stdlib + numpy + sibling hl_core modules. It MUST
NOT import hl_lander or any concrete environment (Design / spec decoupling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .fsm import FiniteStateMachine, FsmState
from .macros import MacroAction, MacroState
from .trace import Decider, TraceRecord


@dataclass
class Ctx:
    """Cross-step context handed to guards / conditions / action_fns.

    Kept minimal (Design "Open Questions"): step index and previous action. The
    FSM runtime state rides along so macro-bound deciders can find their cursor.
    """

    step_index: int = 0
    prev_action: Optional[int] = None
    fsm_state: Optional[FsmState] = None


@dataclass
class MacroDecider:
    """Adapts a MacroAction to the Decider protocol for a macro-bound FSM state.

    The macro is armed once on state entry (on_enter). While active it emits the
    macro's next action; an interrupt stops it early. Once the macro is exhausted
    (or interrupted) it does NOT auto-restart — control passes to `fallback`
    (a Decider, e.g. a RuleTable) so the FSM does not get stuck firing forever.
    The MacroState is looked up from the FsmState by this decider's owning state
    name, so reset() clears it centrally.
    """

    macro: MacroAction
    state_name: str
    fallback: Optional[Decider] = None

    def on_enter(self, ctx: Ctx) -> None:
        ms = self._macro_state(ctx)
        if ms is not None:
            self.macro.start(ms)

    def decide(self, observation: Any, ctx: Ctx) -> tuple[int, TraceRecord]:
        ms = self._macro_state(ctx)
        if ms is not None and self.macro.should_interrupt(observation, ctx):
            self.macro.stop(ms)
        if ms is not None and self.macro.is_active(ms):
            action = self.macro.next_action(observation, ctx, ms)
            return action, TraceRecord(
                step=ctx.step_index,
                state=self.state_name,
                rule="<macro>",
                macro_active=True,
                action=action,
            )
        # Macro exhausted / interrupted / unavailable: hand off to fallback.
        if self.fallback is not None:
            action, base = self.fallback.decide(observation, ctx)
            return action, TraceRecord(
                step=ctx.step_index,
                state=self.state_name,
                rule=base.rule,
                macro_active=False,
                action=action,
            )
        return 0, TraceRecord(
            step=ctx.step_index,
            state=self.state_name,
            rule="default",
            macro_active=False,
            action=0,
        )

    def _macro_state(self, ctx: Ctx) -> Optional[MacroState]:
        if ctx.fsm_state is None:
            return None
        return ctx.fsm_state.macro_states.get(self.state_name)


@dataclass
class ProceduralPolicy:
    """A HeuristicPolicy assembled from hl_core building blocks.

    Provide exactly one root: an `fsm` (typical) or a flat root `decider`.
    """

    fsm: Optional[FiniteStateMachine] = None
    decider: Optional[Decider] = None
    _ctx: Ctx = field(init=False, default_factory=Ctx, repr=False)
    _fsm_state: Optional[FsmState] = field(init=False, default=None, repr=False)
    _trace: list[TraceRecord] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if (self.fsm is None) == (self.decider is None):
            raise ValueError("ProceduralPolicy needs exactly one of fsm= or decider=")

    def reset(self, seed: int) -> None:
        self._trace = []
        self._ctx = Ctx(step_index=0, prev_action=None)
        if self.fsm is not None:
            self._fsm_state = self.fsm.new_state()
            self._ctx.fsm_state = self._fsm_state

    def act(self, observation: np.ndarray) -> int:
        if self.fsm is not None:
            action, record = self.fsm.step(observation, self._ctx, self._fsm_state)
        else:
            assert self.decider is not None  # guaranteed by __post_init__
            action, record = self.decider.decide(observation, self._ctx)
        self._trace.append(record)
        self._ctx.step_index += 1
        self._ctx.prev_action = action
        return int(action)

    def decision_trace(self) -> tuple[TraceRecord, ...]:
        """Immutable snapshot of the per-step decision records (read-only)."""
        return tuple(self._trace)
