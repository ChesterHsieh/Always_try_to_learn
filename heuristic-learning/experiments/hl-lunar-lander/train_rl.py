"""Train the minimal PPO control benchmark on LunarLander-v3 and save a checkpoint.

This is the ONLY place (besides `src/hl_lander/rl/`) where gradient training is
allowed — the RL baseline is a control benchmark, not part of the HL mainline
(see openspec change `hl-lander-jax-rl-baseline`).

Usage:
    python experiments/hl-lunar-lander/train_rl.py \
        --total-steps 500000 --seed 0 --lr 3e-4 --out checkpoints/ppo_lunarlander.msgpack

The checkpoint is written with flax.serialization to
`experiments/hl-lunar-lander/checkpoints/` (gitignored). It bundles the params
plus the hyperparameters needed to reconstruct the network, so RLPPOController
can load it without re-deriving config.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import flax
import gymnasium
import jax
import jax.numpy as jnp

# Make `import hl_lander...` work when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hl_lander.env import ENV_ID, make_env  # noqa: E402
from hl_lander.rl.ppo import (  # noqa: E402
    PPOConfig,
    collect_rollout,
    evaluate_policy,
    init_train_state,
    make_update_step,
)

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"
DEFAULT_OUT = CKPT_DIR / "ppo_lunarlander.msgpack"


def _save_checkpoint(path: Path, params, cfg: PPOConfig, seed: int) -> None:
    """Serialize params + reconstruction metadata with flax.serialization."""
    payload = {
        "params": params,
        "hidden_dim": cfg.hidden_dim,
        "seed": seed,
        "total_steps": cfg.total_steps,
        "env_id": ENV_ID,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(flax.serialization.msgpack_serialize(payload))


def train(cfg: PPOConfig, seed: int, out: Path, eval_every: int, eval_episodes: int) -> None:
    print(f"[train_rl] env={ENV_ID} total_steps={cfg.total_steps} seed={seed} lr={cfg.lr}")
    print(f"[train_rl] flax={flax.__version__} jax={jax.__version__} gymnasium={gymnasium.__version__}")

    env = make_env(seed=seed)
    eval_env = make_env(seed=seed + 10_000)

    state = init_train_state(cfg, seed=seed)
    update_step = make_update_step(cfg)

    key = jax.random.PRNGKey(seed)
    obs, _info = env.reset(seed=seed)

    steps_done = 0
    last_eval_at = 0
    start = time.time()

    # Keep the best-eval params (model selection on the training eval_env, which
    # uses a different seed offset than run.py's seeds 0-4 — so this is not
    # fudging the final evaluation, just standard checkpoint selection).
    best_return = float("-inf")
    best_params = state.params
    best_steps = 0

    while steps_done < cfg.total_steps:
        key, collect_key, update_key = jax.random.split(key, 3)
        rollout, obs, _ = collect_rollout(env, state, obs, collect_key, cfg)
        state, loss = update_step(state, rollout, update_key)
        steps_done += cfg.rollout_len

        if steps_done - last_eval_at >= eval_every or steps_done >= cfg.total_steps:
            key, eval_key = jax.random.split(key)
            mean_return = evaluate_policy(eval_env, state.params, eval_episodes, eval_key)
            elapsed = time.time() - start
            marker = ""
            if mean_return > best_return:
                best_return = mean_return
                best_params = state.params
                best_steps = steps_done
                marker = "  <- best"
            print(
                f"[train_rl] steps={steps_done:>7d}  loss={float(loss):+.3f}  "
                f"eval_mean_return={mean_return:+.1f}  ({elapsed:.0f}s){marker}"
            )
            last_eval_at = steps_done

    env.close()
    eval_env.close()

    _save_checkpoint(out, best_params, cfg, seed)
    print(
        f"[train_rl] saved BEST checkpoint -> {out} "
        f"(eval_mean_return={best_return:+.1f} @ step {best_steps})"
    )
    print(f"[train_rl] flax version: {flax.__version__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train minimal PPO on LunarLander-v3 (control benchmark).")
    parser.add_argument("--total-steps", type=int, default=PPOConfig.total_steps)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=PPOConfig.lr)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    cfg = PPOConfig(total_steps=args.total_steps, lr=args.lr)
    out = args.out if args.out.is_absolute() else (Path(__file__).resolve().parent / args.out)
    train(cfg, seed=args.seed, out=out, eval_every=args.eval_every, eval_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
