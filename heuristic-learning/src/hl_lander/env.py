"""Env factory. The only place that names the gymnasium env id."""

from __future__ import annotations

import gymnasium as gym

ENV_ID = "LunarLander-v3"


def make_env(seed: int, render_mode: str | None = None) -> gym.Env:
    env = gym.make(ENV_ID, render_mode=render_mode)
    env.reset(seed=seed)
    return env
