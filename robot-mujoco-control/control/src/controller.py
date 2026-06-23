"""
Simple PD controller — computes joint position targets from a goal.
Replace this with your actual policy (RL, trajectory planning, etc.).
"""

import math
import time
from shared.protocol import SimObservation


class PDController:
    """Sinusoidal reference trajectory for Phase 1 smoke-testing."""

    def __init__(self, n_joints: int = 2, freq_hz: float = 0.5, amplitude: float = 0.8):
        self._n = n_joints
        self._freq = freq_hz
        self._amp = amplitude
        self._start = time.time()

    def compute(self, obs: SimObservation) -> list[float]:
        t = time.time() - self._start
        targets = [
            self._amp * math.sin(2 * math.pi * self._freq * t + i * math.pi / self._n)
            for i in range(self._n)
        ]
        return targets
