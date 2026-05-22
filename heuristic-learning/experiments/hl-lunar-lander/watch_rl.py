"""Watch the trained rl_ppo policy land, in a live pygame window.

Purely for eyeballing the policy — NOT an evaluation path. It reuses the same
RLPPOController (forward + argmax) and the same env factory as run.py, just with
render_mode="human" so a window opens.

Usage:
    python experiments/hl-lunar-lander/watch_rl.py
    python experiments/hl-lunar-lander/watch_rl.py --episodes 5 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `import hl_lander...` work when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hl_lander.controllers.rl_ppo import RLPPOController  # noqa: E402
from hl_lander.env import make_env  # noqa: E402


def watch(episodes: int, seed: int) -> None:
    controller = RLPPOController()
    for ep in range(episodes):
        ep_seed = seed + ep
        env = make_env(seed=ep_seed, render_mode="human")
        try:
            controller.reset(seed=ep_seed)
            obs, _info = env.reset(seed=ep_seed)
            total = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = controller.act(obs)
                obs, reward, terminated, truncated, _info = env.step(int(action))
                total += float(reward)
            landed = bool(obs[6]) and bool(obs[7]) and terminated and total > 0
            print(f"episode {ep} (seed {ep_seed}): return={total:+.1f}  landed={landed}")
        finally:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the trained rl_ppo policy (live render).")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    watch(args.episodes, args.seed)


if __name__ == "__main__":
    main()
