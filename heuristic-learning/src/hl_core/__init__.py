"""hl_core — environment-agnostic procedural-policy building blocks.

Capability: hl-procedural-policy (Learning Beyond Gradients, "procedural-policy"
theme). Three composable, pure-code building blocks — rule tables, finite state
machines, macro-actions — plus a ProceduralPolicy that assembles them behind the
existing HeuristicPolicy contract and exports a read-only decision trace.

HL red line: no gradients, no neural-network weights, no MPC/CPG/CV here. Every
"parameter" is a named constant in source. Depends only on stdlib + numpy and
MUST NOT import any concrete environment (e.g. hl_lander).

"""

from __future__ import annotations

from .fsm import FiniteStateMachine, FsmState, Transition
from .macros import MacroAction, MacroState
from .policy import Ctx, MacroDecider, ProceduralPolicy
from .rules import Rule, RuleTable
from .trace import Decider, TraceRecord

__all__ = [
    "RuleTable",
    "Rule",
    "FiniteStateMachine",
    "Transition",
    "FsmState",
    "MacroAction",
    "MacroState",
    "ProceduralPolicy",
    "MacroDecider",
    "Ctx",
    "TraceRecord",
    "Decider",
]
