"""NoOpLander — trivial do-nothing baseline. Always action 0 (no engine)."""

from __future__ import annotations

import numpy as np


class NoOpLander:
    def reset(self, seed: int) -> None:
        return None

    def act(self, observation: np.ndarray) -> int:
        return 0
