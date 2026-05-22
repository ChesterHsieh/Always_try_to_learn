"""Actor-critic MLP for the PPO control benchmark (flax.linen).

Input: LunarLander-v3 observation (8-dim). Outputs:
  - policy logits over the 4 discrete actions {0,1,2,3}
  - a scalar state-value estimate

Shared trunk feeds two heads. Small (two 64-unit layers) — enough for
LunarLander on CPU, intentionally not tuned for SOTA (this is a control
benchmark, not a leaderboard entry; see design Decision 1 / Risks).
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

OBS_DIM = 8
N_ACTIONS = 4


def _orthogonal(scale: float):
    """Orthogonal init — the canonical PPO weight init. Stabilizes LunarLander
    training substantially vs. flax's default Dense init (small policy-head gain
    keeps the initial policy near-uniform; tanh-trunk gain √2 preserves variance).
    """
    return nn.initializers.orthogonal(scale)


class ActorCritic(nn.Module):
    """Separate actor and critic trunks with orthogonal init (standard PPO recipe).

    Separate trunks (vs. shared) avoid the policy and value gradients fighting
    over a shared body — a common source of instability on LunarLander.
    """

    hidden_dim: int = 64

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return (logits[..., N_ACTIONS], value[...]) for a single obs or a batch."""
        bias_init = nn.initializers.constant(0.0)

        # Actor head.
        a = nn.Dense(self.hidden_dim, kernel_init=_orthogonal(np.sqrt(2)), bias_init=bias_init)(obs)
        a = nn.tanh(a)
        a = nn.Dense(self.hidden_dim, kernel_init=_orthogonal(np.sqrt(2)), bias_init=bias_init)(a)
        a = nn.tanh(a)
        logits = nn.Dense(N_ACTIONS, kernel_init=_orthogonal(0.01), bias_init=bias_init)(a)

        # Critic head.
        c = nn.Dense(self.hidden_dim, kernel_init=_orthogonal(np.sqrt(2)), bias_init=bias_init)(obs)
        c = nn.tanh(c)
        c = nn.Dense(self.hidden_dim, kernel_init=_orthogonal(np.sqrt(2)), bias_init=bias_init)(c)
        c = nn.tanh(c)
        value = nn.Dense(1, kernel_init=_orthogonal(1.0), bias_init=bias_init)(c)

        # Drop the trailing singleton so value has the batch shape (or scalar).
        return logits, jnp.squeeze(value, axis=-1)
