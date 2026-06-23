"""把一個 HL 控制器降落 LunarLander-v3 的過程錄成 MP4。

與 play_gui.py 同源：重用 make_env 與相同的 controller 介面，差別在改用
render_mode="rgb_array" 收集每一幀，再用 imageio（自帶 ffmpeg）編成 MP4。

輸出的 MP4 走 yuv420p + 偶數尺寸，符合 LinkedIn / 多數平台播放器的相容性要求，
原生上傳貼文即可播放。

用法：
    ./.venv/bin/python experiments/hl-lunar-lander/record_video.py \\
        --controller fsm_macro_v1 --episodes 3 --seed 0 --out lander.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

# 讓 `import hl_lander...` 在以腳本執行時可用
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hl_lander.env import make_env  # noqa: E402

# LunarLander-v3 的 metadata fps；用同一個值編碼，播放速度才與真實模擬一致
DEFAULT_FPS = 50


def _controller_factory(name: str):
    if name == "baseline_v1":
        from hl_lander.controllers.baseline_v1 import BaselineLanderV1

        return BaselineLanderV1
    if name == "fsm_macro_v1":
        from hl_lander.controllers.fsm_macro_v1 import FsmMacroLanderV1

        return FsmMacroLanderV1
    if name == "rl_ppo":
        from hl_lander.controllers.rl_ppo import RLPPOController

        return RLPPOController
    if name == "random":
        from hl_lander.controllers.random import RandomLander

        return RandomLander
    if name == "noop":
        from hl_lander.controllers.noop import NoOpLander

        return NoOpLander
    raise SystemExit(f"unknown controller {name!r}")


def _pad_to_even(frame: np.ndarray) -> np.ndarray:
    """yuv420p 要求寬高皆為偶數；必要時在右/下補一列黑邊。"""
    h, w = frame.shape[:2]
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return frame


def record(controller: str, episodes: int, seed: int, out: Path, fps: int) -> None:
    factory = _controller_factory(controller)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        out,
        fps=fps,
        codec="libx264",
        macro_block_size=None,  # 我們自己處理偶數尺寸，避免 imageio 二次縮放
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    print(f"recording: controller={controller} episodes={episodes} seed={seed} → {out}")
    try:
        for ep in range(episodes):
            ep_seed = seed + ep
            env = make_env(seed=ep_seed, render_mode="rgb_array")
            policy = factory()
            policy.reset(ep_seed)
            obs, _info = env.reset(seed=ep_seed)
            total_return, length = 0.0, 0
            terminated = truncated = False
            try:
                while not (terminated or truncated):
                    writer.append_data(_pad_to_even(np.asarray(env.render())))
                    action = int(policy.act(obs))
                    obs, reward, terminated, truncated, _info = env.step(action)
                    total_return += float(reward)
                    length += 1
                # 補最後一幀，讓著陸/墜毀的定格看得到
                writer.append_data(_pad_to_even(np.asarray(env.render())))
            finally:
                env.close()
            both_legs = bool(obs[6]) and bool(obs[7])
            landed = terminated and both_legs and total_return > 0
            flag = "✅ landed" if landed else ("⚠ truncated" if truncated else "💥 crash/fail")
            print(f"  episode {ep + 1}/{episodes} seed={ep_seed}: "
                  f"return={total_return:.1f} steps={length} {flag}")
    finally:
        writer.close()

    print(f"done → {out}  ({out.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record an HL LunarLander controller to MP4")
    parser.add_argument("--controller", default="fsm_macro_v1",
                        choices=["baseline_v1", "fsm_macro_v1", "rl_ppo", "random", "noop"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--out", type=Path, default=Path("lander.mp4"))
    args = parser.parse_args()
    record(args.controller, args.episodes, args.seed, args.out, args.fps)


if __name__ == "__main__":
    main()
