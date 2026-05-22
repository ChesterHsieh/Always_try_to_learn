"""Decision trace primitives — produced as a by-product of every decision.

Spec: hl-procedural-policy, requirements "ProceduralPolicy 組合介面與決策軌跡導出"
and "回歸測試與 Golden Trace".

Design D3: each Decider returns (action, TraceRecord) in the SAME call that makes
the decision, so the trace can never disagree with the real behaviour. TraceRecord
holds only semantic fields (no floats / timestamps) so golden-trace regression is
robust against behaviour-neutral refactors (Design D4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class TraceRecord:
    """One immutable decision record.

    step         zero-based step index within the episode
    state        name of the FSM state that owned the decision (or a sentinel
                 like "<root>" when the policy has no enclosing FSM)
    rule         name of the rule that fired ("default" when no guard matched,
                 "<macro>" while a macro is driving)
    macro_active whether a macro was producing the action this step
    action       the discrete action emitted
    """

    step: int
    state: str
    rule: str
    macro_active: bool
    action: int


@runtime_checkable
class Decider(Protocol):
    """A composable decision unit: rule table, FSM, or a macro wrapper.

    `ctx` carries cross-step context (e.g. step index, previous action) so the
    building blocks themselves stay free of mutable execution state — progress
    lives in the ProceduralPolicy runtime object (Design D1).
    """

    def decide(self, observation: Any, ctx: Any) -> tuple[int, TraceRecord]: ...
