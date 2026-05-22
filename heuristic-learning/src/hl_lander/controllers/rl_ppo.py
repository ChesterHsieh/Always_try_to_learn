"""RLPPOController — inference-only RL control benchmark over the HeuristicPolicy contract.

This is the bridge that lets a PPO-trained policy run through the SAME runner as
the HL rule-based controllers, so the comparison in REPORT.md is apples-to-apples
(see openspec change `hl-lander-jax-rl-baseline`, design Decision 1).

CRITICAL contract notes:
  - Training happens offline in `experiments/hl-lunar-lander/train_rl.py`. This
    controller ONLY does inference: `act()` is a network forward + argmax, with
    NO gradient update. It does not, and must not, expand the HeuristicPolicy
    interface with any `update`/`learn`/`train` method.
  - If no checkpoint exists, raise a clear error pointing the user at train_rl.py.
"""

from __future__ import annotations

from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np

from hl_lander.rl.networks import OBS_DIM, ActorCritic
from hl_lander.rl.ppo import greedy_action

# Default checkpoint location written by train_rl.py (gitignored).
# This file is src/hl_lander/controllers/rl_ppo.py, so parents[3] is the
# heuristic-learning project root where experiments/ lives.
_DEFAULT_CKPT = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "hl-lunar-lander"
    / "checkpoints"
    / "ppo_lunarlander.msgpack"
)


class RLPPOController:
    """Loads a PPO checkpoint and acts greedily. Implements HeuristicPolicy."""

    def __init__(self, checkpoint_path: Path | str | None = None) -> None:
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CKPT
        self._params = None  # loaded lazily on first reset()

    def _load(self) -> None:
        if self._params is not None:
            return
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"RL checkpoint not found at {self._checkpoint_path}. "
                "Train one first:\n"
                "    python experiments/hl-lunar-lander/train_rl.py "
                "--total-steps 500000 --seed 0"
            )
        payload = flax.serialization.msgpack_restore(self._checkpoint_path.read_bytes())
        hidden_dim = int(payload["hidden_dim"])
        # Rebuild the param-tree shape, then restore the saved values into it.
        template = ActorCritic(hidden_dim=hidden_dim).init(
            jax.random.PRNGKey(0), jnp.zeros((OBS_DIM,))
        )
        self._params = flax.serialization.from_state_dict(template, payload["params"])

    def reset(self, seed: int) -> None:
        """Load checkpoint params if not already loaded. Stateless thereafter."""
        self._load()

    def act(self, observation: np.ndarray) -> int:
        """Inference only: policy forward + argmax. NO gradient update here."""
        if self._params is None:
            self._load()
        return greedy_action(self._params, np.asarray(observation, dtype=np.float32)[:OBS_DIM])
