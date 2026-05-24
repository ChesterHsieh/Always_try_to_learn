"""RandomOpponent — picks a uniformly random legal move (deterministic per seed).

Spec: hl-gobblet-env, requirement "環境介面與隨機對手".

Used to validate the env can self-play a full game. It owns a seeded numpy RNG
so the same seed replays the same choices; reset(seed) re-seeds it. It never
returns an illegal move because it samples only from legal_moves(state).
"""

from __future__ import annotations

import numpy as np

from ..moves import Move
from ..rules import legal_moves
from ..state import GobbletState


class RandomOpponent:
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def act(self, state: GobbletState) -> Move:
        moves = legal_moves(state)
        if not moves:
            raise ValueError("RandomOpponent.act called with no legal moves")
        idx = int(self._rng.integers(len(moves)))
        return moves[idx]
