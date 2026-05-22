"""以 GUI（render_mode="human"）觀看一個 HL 控制器降落 LunarLander-v3。

這支腳本只用於「看」——不寫 REPORT、不做統計，純粹開視窗呈現控制器的行為。
重用 hl_lander.env.make_env，所以 env id / 設定與正式評估一致。

用法：
    ./.venv/bin/python experiments/hl-lunar-lander/play_gui.py \\
        --controller fsm_macro_v1 --episodes 3 --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 讓 `import hl_lander...` 在以腳本執行時可用
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hl_lander.env import make_env  # noqa: E402


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
    raise SystemExit(f"unknown controller {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI playback of an HL LunarLander controller")
    parser.add_argument("--controller", default="fsm_macro_v1",
                        choices=["baseline_v1", "fsm_macro_v1", "random", "noop"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    factory = _controller_factory(args.controller)
    print(f"GUI playback: controller={args.controller} episodes={args.episodes} "
          f"(關閉視窗或 Ctrl+C 可中止)")

    for ep in range(args.episodes):
        seed = args.seed + ep
        env = make_env(seed=seed, render_mode="human")
        policy = factory()
        policy.reset(seed)
        obs, _info = env.reset(seed=seed)
        total_return, length = 0.0, 0
        terminated = truncated = False
        try:
            while not (terminated or truncated):
                action = int(policy.act(obs))
                obs, reward, terminated, truncated, _info = env.step(action)
                total_return += float(reward)
                length += 1
        finally:
            env.close()
        both_legs = bool(obs[6]) and bool(obs[7])
        landed = terminated and both_legs and total_return > 0
        flag = "✅ landed" if landed else ("⚠ truncated" if truncated else "💥 crash/fail")
        print(f"  episode {ep + 1}/{args.episodes} seed={seed}: "
              f"return={total_return:.1f} steps={length} {flag}")
        time.sleep(0.5)  # 回合之間停一下，方便觀看

    print("done.")


if __name__ == "__main__":
    main()
