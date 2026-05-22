# Heuristic Learning（HL）— LunarLander

研讀並複現 [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/#zh) 的「啟發式學習（Heuristic Learning）」典範：在不訓練神經網路、不靠梯度的前提下，用手寫規則與程式化策略控制 agent，再以環境反饋反覆迭代。

本專案把這個命題落在單一環境 **LunarLander-v3** 上做硬核對照：手寫規則 controller vs. 同 repo 自己訓練的 RL，看「不訓練」要付出多少代價。

## 結論（5 seeds × 10 episodes）

| controller    | mean return | landing rate | 訓練？           |
|---------------|-------------|--------------|------------------|
| `noop`        | -131.0      | 0%           | 無               |
| `random`      | -183.9      | 0%           | 無               |
| `baseline_v1` | **+264.4**  | **68%**      | 無（手寫規則）   |
| `fsm_macro_v1`| **+268.6**  | 58%          | 無（規則組合）   |
| `rl_ppo`      | +245.8      | 58%          | 有（梯度訓練）   |

手寫規則的 `baseline_v1`（零訓練）與梯度訓練 200 萬步的 `rl_ppo` **同級**，且兩者都遠勝弱對照——這就是 HL 命題的量化展示。完整分析與每次評估的 seed／指令／commit 見 [experiments/hl-lunar-lander/REPORT.md](experiments/hl-lunar-lander/REPORT.md)。

## 目錄結構

```text
heuristic-learning/
├── src/
│   ├── hl_core/                # 可重用的 HL 原語：rules、fsm、macros、trace、policy 介面
│   └── hl_lander/              # LunarLander 專用：env / runner / metrics
│       ├── controllers/        # 各策略：baseline_v1、fsm_macro_v1、random、noop、rl_ppo
│       └── rl/                 # RL 對照組：手寫 minimal PPO（actor-critic、GAE）
├── experiments/hl-lunar-lander/
│   ├── run.py                  # 評估 controller，append 一段到 REPORT.md
│   ├── train_rl.py             # 訓練 PPO 對照組（唯一允許梯度訓練的地方）
│   ├── play_gui.py             # 開視窗看規則 controller 降落
│   ├── watch_rl.py             # 開視窗看 rl_ppo 降落
│   └── REPORT.md               # 實驗報告（每個 controller 一段，append-only）
├── tests/                      # hl_core 原語的單元測試 + golden 測試
└── notes/                      # 概念筆記、環境備忘
```

### controller 不可變性

每個 `baseline_v{n}` / `fsm_macro_v{n}` 是 HL 迭代軌跡上的一個獨立 artefact。要改進策略就**新增 `_v{n+1}.py`**，不就地改既有版本——這樣 demo 能一頁並排比較各版本程式碼，呼應 HL「程式碼結構就是策略本身」的命題。

## 環境

自有 `pyproject.toml` + `.venv`（`uv` 管理）。dependencies 限 JAX 生態系，**禁** TensorFlow / PyTorch。

```bash
# Box2D 需要 swig；macOS 先裝（僅首次）
brew install swig
uv venv .venv --python 3.13
uv sync

# 驗證
make doctor                    # 印出 python / jax / flax 版本
make hl-lander-deps-check      # 確認沒有 TF/PyTorch
```

## 怎麼跑

```bash
make hl-lander-eval CONTROLLER=fsm_macro_v1  # 評估 controller，append 至 REPORT.md
make hl-lander-watch CONTROLLER=fsm_macro_v1 # 開 GUI；CONTROLLER=rl_ppo 走 watch_rl.py
make hl-lander-train                          # 重新訓練 RL 對照組（2M steps）
make test                                     # 執行測試
```

`CONTROLLER` 預設 `baseline_v1`；其他可覆寫參數：`SEEDS`（5）、`EPISODES`（10）、`SEED`（0）、`STEPS`（2000000）。

可用的 `CONTROLLER`：`baseline_v1`、`fsm_macro_v1`、`random`、`noop`、`rl_ppo`。

## 參考資料

- 原文：[Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/#zh)
- GitHub 版本：[learning-beyond-gradient.md](https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/learning-beyond-gradient.md)
- 詳細筆記：[learning-beyond-gradients-reference.md](notes/learning-beyond-gradients-reference.md)

## Credits

This project is inspired by and implements the Heuristic Learning paradigm from:
- **Author**: Jiayi Weng ([@Trinkle23897](https://github.com/Trinkle23897))
- **Original Work**: [Learning Beyond Gradients](https://github.com/Trinkle23897/learning-beyond-gradients)

The core thesis—that hand-crafted rules without gradient training can match RL baselines when maintained by coding agents—is demonstrated empirically in this repo through LunarLander experiments.
