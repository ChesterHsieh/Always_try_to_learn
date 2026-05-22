"""RandomLander — uniform-random action lower bound. Ignores the observation."""

from __future__ import annotations

import numpy as np


class RandomLander:
    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> int:
        return int(self._rng.integers(0, 4))
