"""Episode loop + multi-seed evaluation. No metrics aggregation here (see metrics.py)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from .env import make_env
from .policy import HeuristicPolicy


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    episode_index: int
    total_return: float
    length: int
    landed: bool


def _is_landed(terminated: bool, total_return: float, last_obs) -> bool:
    """Heuristic 'successful landing' flag.

    LunarLander gives a large positive terminal bonus on a clean landing and a
    -100 penalty on crash. We treat an episode as landed when it terminated
    (not truncated), both legs were in contact (obs[6], obs[7]), and the return
    is positive.
    """
    both_legs = bool(last_obs[6]) and bool(last_obs[7])
    return terminated and both_legs and total_return > 0


def run_episode(policy: HeuristicPolicy, seed: int, episode_index: int) -> EpisodeResult:
    env = make_env(seed=seed + episode_index)
    try:
        policy.reset(seed=seed + episode_index)
        obs, _info = env.reset(seed=seed + episode_index)
        total_return = 0.0
        length = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = policy.act(obs)
            obs, reward, terminated, truncated, _info = env.step(int(action))
            total_return += float(reward)
            length += 1
        return EpisodeResult(
            seed=seed,
            episode_index=episode_index,
            total_return=total_return,
            length=length,
            landed=_is_landed(terminated, total_return, obs),
        )
    finally:
        env.close()


def evaluate(
    policy_factory: Callable[[], HeuristicPolicy],
    seeds: List[int],
    episodes_per_seed: int,
) -> List[EpisodeResult]:
    results: List[EpisodeResult] = []
    for seed in seeds:
        policy = policy_factory()
        for episode_index in range(episodes_per_seed):
            results.append(run_episode(policy, seed=seed, episode_index=episode_index))
    return results
