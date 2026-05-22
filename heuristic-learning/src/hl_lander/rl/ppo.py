"""Minimal PPO for the LunarLander-v3 control benchmark.

Design (see openspec change `hl-lander-jax-rl-baseline`, design.md):
  - Decision 2: the env loop runs in plain Python over a single gymnasium env
    (Box2D cannot be jit-ed). Only the network forward pass and the gradient
    update step are jit-ed. The bottleneck is the C-level Box2D step, not the
    tiny MLP, so leaving the env un-jit-ed is acceptable.
  - Decision 3: PPO only; DQN is a stretch task.

This is a deliberately small, single-file PPO: rollout buffer -> GAE ->
clipped surrogate + value loss + entropy bonus -> optax Adam. No vectorized
envs, no SOTA tricks. The goal is a fair, reproducible RL control number on the
same env the HL controllers run on, not a leaderboard score.

Everything that touches gradients lives here and in `train_rl.py`. The HL
mainline controllers stay gradient-free.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from .networks import N_ACTIONS, OBS_DIM, ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters. Defaults chosen to converge on LunarLander on CPU."""

    total_steps: int = 500_000
    rollout_len: int = 2048
    num_epochs: int = 10
    num_minibatches: int = 32
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    anneal_lr: bool = True  # linearly decay lr to 0 over training (standard PPO)

    @property
    def num_updates(self) -> int:
        return self.total_steps // self.rollout_len

    @property
    def updates_per_rollout(self) -> int:
        return self.num_epochs * self.num_minibatches


class Rollout(NamedTuple):
    """A fixed-length batch of transitions collected from the env loop."""

    obs: jnp.ndarray  # (T, OBS_DIM)
    actions: jnp.ndarray  # (T,)
    log_probs: jnp.ndarray  # (T,)  log pi(a|s) at collection time
    values: jnp.ndarray  # (T,)  V(s) at collection time
    rewards: jnp.ndarray  # (T,)
    dones: jnp.ndarray  # (T,)  1.0 if the step terminated/truncated the episode
    last_value: jnp.ndarray  # scalar V(s_T) for bootstrapping the final segment


# --- pure-JAX building blocks (all jit-able) -------------------------------
#
# The jit'd apply-path functions reconstruct the module from the actual width
# encoded in the params, so they work for ANY hidden_dim checkpoint without the
# module default and the trained config drifting apart.


def hidden_dim_from_params(params) -> int:
    """Read the trunk width from a param tree (the first actor Dense layer)."""
    # init() nests params under "params"; layers are "Dense_0", "Dense_1", ...
    leaves = params["params"] if "params" in params else params
    return int(leaves["Dense_0"]["kernel"].shape[1])


def _apply(params, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    model = ActorCritic(hidden_dim=hidden_dim_from_params(params))
    return model.apply(params, obs)


@jax.jit
def policy_value(params, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward pass: (logits, value) for a single obs or a batch."""
    return _apply(params, obs)


@jax.jit
def sample_action(params, obs: jnp.ndarray, key) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample an action from the policy. Returns (action, log_prob, value)."""
    logits, value = _apply(params, obs)
    action = jax.random.categorical(key, logits)
    log_prob = _categorical_log_prob(logits, action)
    return action, log_prob, value


def _categorical_log_prob(logits: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits)
    return jnp.take_along_axis(log_probs, action[..., None], axis=-1)[..., 0]


def _entropy(logits: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


@jax.jit
def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generalized Advantage Estimation over a fixed-length rollout.

    Returns (advantages, returns) where returns = advantages + values.
    """

    def step(carry, xs):
        gae, next_value = carry
        reward, value, done = xs
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        return (gae, value), gae

    (_, _), advantages = jax.lax.scan(
        step,
        (jnp.array(0.0), last_value),
        (rewards, values, dones),
        reverse=True,
    )
    returns = advantages + values
    return advantages, returns


def _loss_fn(params, batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns, cfg: PPOConfig):
    logits, values = ActorCritic(hidden_dim=cfg.hidden_dim).apply(params, batch_obs)
    log_probs = _categorical_log_prob(logits, batch_actions)

    # Normalize advantages per minibatch for stability.
    adv = batch_advantages
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    ratio = jnp.exp(log_probs - batch_old_log_probs)
    unclipped = ratio * adv
    clipped = jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

    value_loss = 0.5 * jnp.mean((values - batch_returns) ** 2)
    entropy = jnp.mean(_entropy(logits))

    total = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
    return total, (policy_loss, value_loss, entropy)


def make_update_step(cfg: PPOConfig):
    """Build a jit-ed PPO update over one rollout (multiple epochs + minibatches).

    cfg is closed over as a static, so changing hyperparameters means rebuilding
    the step — which is exactly what we want (the jit cache keys on it).
    """
    batch_size = cfg.rollout_len
    minibatch_size = batch_size // cfg.num_minibatches

    @jax.jit
    def update_step(state: TrainState, rollout: Rollout, key):
        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.values,
            rollout.dones,
            rollout.last_value,
            cfg.gamma,
            cfg.gae_lambda,
        )

        def epoch_body(carry, _):
            state, key = carry
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, batch_size)

            def minibatch_body(state, mb_idx):
                idx = jax.lax.dynamic_slice_in_dim(perm, mb_idx * minibatch_size, minibatch_size)
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (loss, aux), grads = grad_fn(
                    state.params,
                    rollout.obs[idx],
                    rollout.actions[idx],
                    rollout.log_probs[idx],
                    advantages[idx],
                    returns[idx],
                    cfg,
                )
                state = state.apply_gradients(grads=grads)
                return state, loss

            state, losses = jax.lax.scan(
                minibatch_body, state, jnp.arange(cfg.num_minibatches)
            )
            return (state, key), jnp.mean(losses)

        (state, _), epoch_losses = jax.lax.scan(
            epoch_body, (state, key), None, length=cfg.num_epochs
        )
        return state, jnp.mean(epoch_losses)

    return update_step


def init_train_state(cfg: PPOConfig, seed: int) -> TrainState:
    key = jax.random.PRNGKey(seed)
    model = ActorCritic(hidden_dim=cfg.hidden_dim)
    params = model.init(key, jnp.zeros((OBS_DIM,)))

    if cfg.anneal_lr:
        # One optimizer step per minibatch per epoch; decay linearly to 0.
        total_opt_steps = cfg.num_updates * cfg.updates_per_rollout
        lr = optax.linear_schedule(init_value=cfg.lr, end_value=0.0, transition_steps=total_opt_steps)
    else:
        lr = cfg.lr

    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def collect_rollout(env, state: TrainState, last_obs: np.ndarray, key, cfg: PPOConfig):
    """Run the env for cfg.rollout_len steps, returning (Rollout, last_obs, key).

    Plain Python loop over the gymnasium env (Box2D, un-jit-able). Only the
    per-step policy forward pass is jit-ed via `sample_action`.
    """
    obs_buf = np.zeros((cfg.rollout_len, OBS_DIM), dtype=np.float32)
    act_buf = np.zeros((cfg.rollout_len,), dtype=np.int32)
    logp_buf = np.zeros((cfg.rollout_len,), dtype=np.float32)
    val_buf = np.zeros((cfg.rollout_len,), dtype=np.float32)
    rew_buf = np.zeros((cfg.rollout_len,), dtype=np.float32)
    done_buf = np.zeros((cfg.rollout_len,), dtype=np.float32)

    obs = last_obs
    for t in range(cfg.rollout_len):
        key, act_key = jax.random.split(key)
        action, log_prob, value = sample_action(state.params, jnp.asarray(obs), act_key)
        action = int(action)

        obs_buf[t] = obs
        act_buf[t] = action
        logp_buf[t] = float(log_prob)
        val_buf[t] = float(value)

        obs, reward, terminated, truncated, _info = env.step(action)
        rew_buf[t] = float(reward)
        done = terminated or truncated
        done_buf[t] = 1.0 if done else 0.0

        if done:
            obs, _info = env.reset()

    # Bootstrap value for the segment that runs past the buffer end.
    _logits, last_value = policy_value(state.params, jnp.asarray(obs))

    rollout = Rollout(
        obs=jnp.asarray(obs_buf),
        actions=jnp.asarray(act_buf),
        log_probs=jnp.asarray(logp_buf),
        values=jnp.asarray(val_buf),
        rewards=jnp.asarray(rew_buf),
        dones=jnp.asarray(done_buf),
        last_value=jnp.asarray(float(last_value)),
    )
    return rollout, obs, key


def greedy_action(params, obs: np.ndarray) -> int:
    """Argmax over policy logits — the inference path used by RLPPOController."""
    logits, _value = policy_value(params, jnp.asarray(obs))
    return int(jnp.argmax(logits))


def evaluate_policy(env, params, num_episodes: int, key) -> float:
    """Mean greedy return over num_episodes — used for the training learning curve."""
    returns = []
    for _ in range(num_episodes):
        obs, _info = env.reset()
        done = False
        total = 0.0
        while not done:
            action = greedy_action(params, obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns))
