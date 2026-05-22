"""Evaluate an HL controller on LunarLander-v3 and append a section to REPORT.md.

Usage:
    python experiments/hl-lunar-lander/run.py --controller baseline_v1 --seeds 5 --episodes 10

Controllers are dispatched by file-name key (no version-less `baseline` alias).
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import gymnasium

# Make `import hl_lander...` work when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hl_lander.env import ENV_ID  # noqa: E402
from hl_lander.metrics import summarize  # noqa: E402
from hl_lander.runner import evaluate  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parent / "REPORT.md"


def _controller_factory(name: str):
    if name == "baseline_v1":
        from hl_lander.controllers.baseline_v1 import BaselineLanderV1

        return BaselineLanderV1
    if name == "fsm_macro_v1":
        from hl_lander.controllers.fsm_macro_v1 import FsmMacroLanderV1

        return FsmMacroLanderV1
    if name == "random":
        from hl_lander.controllers.random import RandomLander

        return RandomLander
    if name == "noop":
        from hl_lander.controllers.noop import NoOpLander

        return NoOpLander
    if name == "rl_ppo":
        from hl_lander.controllers.rl_ppo import RLPPOController

        return RLPPOController
    raise SystemExit(
        f"unknown controller {name!r}; expected one of: "
        "baseline_v1, fsm_macro_v1, random, noop, rl_ppo"
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _rl_ppo_extra_lines() -> str:
    """RL-specific REPORT metadata: training steps, flax version, checkpoint id.

    Reads the checkpoint train_rl.py wrote (params + reconstruction metadata)
    and reports a sha256 prefix as the checkpoint identifier so the run is
    traceable even though the checkpoint itself is gitignored.
    """
    import hashlib

    import flax

    ckpt = Path(__file__).resolve().parent / "checkpoints" / "ppo_lunarlander.msgpack"
    if not ckpt.exists():
        return f"- flax：{flax.__version__}\n- checkpoint：(未找到 {ckpt.name})\n"

    raw = ckpt.read_bytes()
    payload = flax.serialization.msgpack_restore(raw)
    digest = hashlib.sha256(raw).hexdigest()[:12]
    total_steps = payload.get("total_steps", "unknown")
    train_seed = payload.get("seed", "unknown")
    return (
        f"- 訓練步數：{total_steps}\n"
        f"- 訓練 seed：{train_seed}\n"
        f"- flax：{flax.__version__}\n"
        f"- checkpoint：{ckpt.name} (sha256:{digest})\n"
    )


def _append_section(controller: str, command: str, seeds, episodes, summary) -> None:
    today = dt.date.today().isoformat()
    section = (
        f"\n## {controller} ({today})\n\n"
        f"- 執行指令：`{command}`\n"
        f"- gymnasium：{gymnasium.__version__}\n"
        f"- env id：{ENV_ID}\n"
        f"- seeds：{list(seeds)}\n"
        f"- episodes/seed：{episodes}（共 {summary['n']} episodes）\n"
        f"- mean return：{summary['mean_return']:.1f}\n"
        f"- std return：{summary['std_return']:.1f}\n"
        f"- landing rate：{summary['landing_rate']:.2%}\n"
        f"- git commit：{_git_commit()}\n"
    )
    if controller == "rl_ppo":
        section += _rl_ppo_extra_lines()
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(section)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an HL controller on LunarLander-v3.")
    parser.add_argument(
        "--controller",
        required=True,
        choices=["baseline_v1", "fsm_macro_v1", "random", "noop", "rl_ppo"],
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.render:
        # render path is intentionally not wired into evaluate() in this scaffold
        print("[warn] --render is a placeholder in the scaffold; running headless.")

    seeds = list(range(args.seeds))
    factory = _controller_factory(args.controller)
    results = evaluate(factory, seeds=seeds, episodes_per_seed=args.episodes)
    summary = summarize(results)

    command = (
        f"python experiments/hl-lunar-lander/run.py --controller {args.controller} "
        f"--seeds {args.seeds} --episodes {args.episodes}"
    )
    print(
        f"{args.controller}: n={summary['n']} "
        f"mean={summary['mean_return']:.1f} std={summary['std_return']:.1f} "
        f"landing_rate={summary['landing_rate']:.2%}"
    )
    _append_section(args.controller, command, seeds, args.episodes, summary)
    print(f"appended section to {REPORT_PATH}")


if __name__ == "__main__":
    main()
