"""Procedural (gradient-free) opponents for GobbletEnv, assembled from hl_core.

Capability: hl-gobblet-fsm-controller. Mirrors the hl_lander/controllers/
convention: each controller implements the Gobblet opponent contract
(`reset(seed)` + `act(state) -> Move`) but, unlike the shipped RandomOpponent,
picks moves with a readable, code-only policy.

FsmGobbletV1 reuses hl_core's FiniteStateMachine + RuleTable building blocks to
switch between two named modes — `aggressive` and `defensive` — driven purely by
a one-ply threat assessment. No gradients, no neural-network weights, no tree
search (HL red line); every "parameter" is a named constant in source.
"""

from __future__ import annotations

from .fsm_gobblet_v1 import FsmGobbletV1

__all__ = ["FsmGobbletV1"]
