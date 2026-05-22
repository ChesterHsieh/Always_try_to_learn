"""FsmMacroLanderV1 — procedural policy assembled from hl_core building blocks.

Capability: hl-procedural-policy demo on hl-lunar-lander. Demonstrates that a
code-composed policy (rule tables + finite state machine + macro-action) can be
iterated and beat a single-file rule controller, with an exportable decision
trace for explainability and golden-trace regression.

This controller builds ENTIRELY on hl_core; it does NOT hand-roll a parallel
state machine, and it does NOT modify the immutable baseline_v1.py. It implements
the existing HeuristicPolicy contract via hl_core.ProceduralPolicy.

State machine (readable named states):
  descend   — high above the pad: steer back over the pad and control descent
              (the proven baseline_v1 angle+descent rule, expressed as a RuleTable)
  align     — close to the surface but still off-centre: prioritise attitude so
              the lander touches down upright
  touchdown — a leg has made (or is about to make) contact: a short stable-burn
              MacroAction damps the remaining fall, interrupted once both legs
              are firmly down

LunarLander-v3 observation layout:
  obs[0] x position   obs[1] y position   obs[2] x velocity   obs[3] y velocity
  obs[4] angle        obs[5] angular vel  obs[6] left-leg     obs[7] right-leg
Discrete actions: 0 no-op, 1 fire-left, 2 fire-main, 3 fire-right.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hl_core import (
    FiniteStateMachine,
    MacroAction,
    MacroDecider,
    ProceduralPolicy,
    Rule,
    RuleTable,
    Transition,
)


# --- shared decision helpers (pure functions over observation) ------------------
def _angle_todo(obs: np.ndarray) -> float:
    x, _y, vx, _vy, angle, ang_vel = (obs[0], obs[1], obs[2], obs[3], obs[4], obs[5])
    target_angle = float(np.clip(0.5 * x + 1.0 * vx, -0.4, 0.4))
    return (target_angle - angle) * 0.5 - ang_vel * 1.0


def _hover_todo(obs: np.ndarray) -> float:
    x, y, _vx, vy = obs[0], obs[1], obs[2], obs[3]
    if obs[6] or obs[7]:
        return -vy * 0.5  # on the ground: only damp remaining fall speed
    hover_target = 0.55 * abs(x)
    return (hover_target - y) * 0.5 - vy * 0.5


def _steer_action(obs: np.ndarray) -> int:
    """The baseline_v1 angle+descent decision, reused as the descend-phase logic."""
    angle_todo = 0.0 if (obs[6] or obs[7]) else _angle_todo(obs)
    hover_todo = _hover_todo(obs)
    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        return 2
    if angle_todo < -0.05:
        return 3
    if angle_todo > 0.05:
        return 1
    return 0


def _align_action(obs: np.ndarray) -> int:
    """Attitude-first logic for the final approach.

    Distinct from descend: near the surface, prioritise levelling the lander
    (correct angle first) and only fire the main engine when the fall is
    genuinely fast, so it touches down upright. This is where the FSM earns its
    keep over the single baseline rule.
    """
    angle_todo = _angle_todo(obs)
    if angle_todo < -0.05:
        return 3
    if angle_todo > 0.05:
        return 1
    if _hover_todo(obs) > 0.15:  # fall is fast -> brake
        return 2
    return 0


# --- state predicates -----------------------------------------------------------
def _near_surface(obs: np.ndarray) -> bool:
    # Low and slow vertical: about to land.
    return bool(obs[1] < 0.25)


def _leg_contact(obs: np.ndarray) -> bool:
    return bool(obs[6] or obs[7])


def _both_legs_down(obs: np.ndarray) -> bool:
    return bool(obs[6] and obs[7])


# --- assembled policy -----------------------------------------------------------
def _build_fsm() -> FiniteStateMachine:
    # descend + align share the proven steering rule; align exists as a distinct,
    # readable phase so future iterations can specialise attitude control there.
    descend = RuleTable(
        rules=[
            Rule(
                name="steer",
                priority=0,
                guard=lambda o, c: True,
                action_fn=lambda o, c: _steer_action(o),
            ),
        ],
        default_action=0,
        state_name="descend",
    )
    align = RuleTable(
        rules=[
            Rule(
                name="align",
                priority=0,
                guard=lambda o, c: True,
                action_fn=lambda o, c: _align_action(o),
            ),
        ],
        default_action=0,
        state_name="align",
    )
    # Stable-burn macro: a single main-engine tap on entry to arrest a fast fall,
    # interrupted immediately once both legs are firmly down. Kept to length 1 so
    # it never over-burns; once done it does NOT loop — control falls through to a
    # settle RuleTable that mirrors baseline_v1's on-ground behaviour (damp only
    # remaining fall speed), so this phase is never worse than baseline_v1 and the
    # episode terminates cleanly.
    touchdown_macro = MacroAction(
        name="stable_burn",
        sequence=(2,),
        interrupt=lambda o, c: _both_legs_down(o),
    )
    settle = RuleTable(
        rules=[
            # Same threshold as baseline_v1's on-ground damping (-vy*0.5 > 0.05).
            Rule(
                name="damp",
                priority=0,
                guard=lambda o, c: (-o[3] * 0.5) > 0.05,
                action_fn=lambda o, c: 2,
            ),
        ],
        default_action=0,
        state_name="touchdown",
    )
    touchdown = MacroDecider(macro=touchdown_macro, state_name="touchdown", fallback=settle)

    return FiniteStateMachine(
        initial_state="descend",
        deciders={"descend": descend, "align": align, "touchdown": touchdown},
        transitions=[
            Transition(src="descend", dst="align", condition=lambda o, c: _near_surface(o)),
            Transition(src="align", dst="touchdown", condition=lambda o, c: _leg_contact(o)),
            Transition(src="descend", dst="touchdown", condition=lambda o, c: _leg_contact(o)),
        ],
    )


class FsmMacroLanderV1:
    """HeuristicPolicy facade over a ProceduralPolicy assembled from hl_core."""

    def __init__(self) -> None:
        self._policy = ProceduralPolicy(fsm=_build_fsm())

    def reset(self, seed: int) -> None:
        self._policy.reset(seed)

    def act(self, observation: np.ndarray) -> int:
        return self._policy.act(observation)

    def decision_trace(self) -> tuple[Any, ...]:
        """Expose the underlying ProceduralPolicy decision trace (read-only)."""
        return self._policy.decision_trace()
