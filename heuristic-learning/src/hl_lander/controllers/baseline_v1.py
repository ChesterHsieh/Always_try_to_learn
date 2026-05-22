"""BaselineLanderV1 — first rung of the HL iteration trajectory.

IMMUTABLE once merged to main. To improve the policy, create baseline_v2.py
(copy this as a starting point) — do NOT edit the behaviour here. See
controllers/__init__.py for the version-immutability policy.

Strategy: a hand-written "angle + descent" rule (no training, no hyperparameter
search). It derives a target lean angle from horizontal position/velocity, then
issues a discrete action that either corrects attitude or fires the main engine
to control descent.

LunarLander-v3 observation layout:
  obs[0] x position        obs[4] angle
  obs[1] y position        obs[5] angular velocity
  obs[2] x velocity        obs[6] left leg contact (bool)
  obs[3] y velocity        obs[7] right leg contact (bool)

Discrete actions: 0 = no-op, 1 = fire left engine, 2 = fire main engine, 3 = fire right engine.
"""

from __future__ import annotations

import numpy as np


class BaselineLanderV1:
    def reset(self, seed: int) -> None:
        """No-op: this controller is stateless.

        Declared for the HeuristicPolicy contract. Because there is no internal
        state or RNG, repeated reset() calls cannot leak state between episodes.
        """
        return None

    def act(self, observation: np.ndarray) -> int:
        x, y, vx, vy, angle, ang_vel, leg_left, leg_right = observation[:8]

        # Target a lean angle that steers back over the pad and damps horizontal
        # velocity; target a hover altitude proportional to horizontal offset.
        target_angle = float(np.clip(0.5 * x + 1.0 * vx, -0.4, 0.4))
        hover_target = 0.55 * abs(x)

        angle_todo = (target_angle - angle) * 0.5 - ang_vel * 1.0
        hover_todo = (hover_target - y) * 0.5 - vy * 0.5

        if leg_left or leg_right:
            # On the ground: stop steering, only damp remaining fall speed.
            angle_todo = 0.0
            hover_todo = -vy * 0.5

        if hover_todo > abs(angle_todo) and hover_todo > 0.05:
            return 2  # fire main engine
        if angle_todo < -0.05:
            return 3  # fire right orientation engine
        if angle_todo > 0.05:
            return 1  # fire left orientation engine
        return 0  # no-op
