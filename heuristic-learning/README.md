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

## 第二個環境：奇迹连连（Gobblet Gobblers）

為了驗證 HL 命題在「結構迥異」環境上的普適性，本 repo 另有一個離散、回合制、雙人對弈、含隱藏資訊的環境 `hl_gobblet`。它是 3×3 井字棋的進階版：每方 6 子（大／中／小各 2），大子可「吃」（蓋住）較小子；最快讓自己的子（盤面最上層）連成三線者獲勝。移動大子會重新揭露底下被蓋住的子——記憶／隱藏資訊正是這遊戲的 tricky 核心。

環境本體採不可變狀態 + 純函式推進：

```text
src/hl_gobblet/
├── state.py        # GobbletState（frozen）：每格 size→格主的堆疊、雙方手牌
├── moves.py        # Move 值物件 + 穩定的 move<->index 全動作空間編碼
├── rules.py        # legal_moves / apply_move / 連線勝負判定（含 reveal_loses 變體）
├── env.py          # GobbletEnv：gymnasium 風格、P0 視角 + 注入式對手 hook
├── render.py       # 純函式：state + 上一步 → rich 可渲染物件（無 I/O）
└── opponents/      # RandomOpponent（決定性、吃 seed）
experiments/hl-gobblet/watch_match.py  # CLI 對戰觀戰器（rich.live 逐步刷新）
```

勝負判定預設用官方規則（落子完成後才檢查連線）；另有可設定旗標 `reveal_loses` 實作進階 touch-move 變體：拿起一子的瞬間若揭露對方已連線，則拿起者判負。

### 看兩個 AI 對戰

用 `rich` 渲染的 CLI 觀戰器逐步播放整局 AI vs AI：

```bash
make hl-gobblet-watch                       # 兩個 random AI 對打（自動播放）
make hl-gobblet-watch P0=random P1=random SEED=3 DELAY=0.3
# 直接呼叫腳本可用更多選項：
./.venv/bin/python experiments/hl-gobblet/watch_match.py --step          # 按 Enter 逐步
./.venv/bin/python experiments/hl-gobblet/watch_match.py --reveal-loses  # 進階變體
```

畫面以大小寫＋顏色區分雙方（P0 大寫青色、P1 小寫洋紅），格子尾端 `*` 表示底下還蓋著別的子，並逐步顯示每一步動作與被揭露的子。目前對手只有 `random`；手寫規則 controller 與 greedy 前瞻、勝率評估管線留待後續 change。

## 參考資料

- 原文：[Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/#zh)
- GitHub 版本：[learning-beyond-gradient.md](https://github.com/Trinkle23897/learning-beyond-gradients/blob/main/learning-beyond-gradient.md)
- 詳細筆記：[learning-beyond-gradients-reference.md](notes/learning-beyond-gradients-reference.md)

## Credits

This project is inspired by and implements the Heuristic Learning paradigm from:
- **Author**: Jiayi Weng ([@Trinkle23897](https://github.com/Trinkle23897))
- **Original Work**: [Learning Beyond Gradients](https://github.com/Trinkle23897/learning-beyond-gradients)

The core thesis—that hand-crafted rules without gradient training can match RL baselines when maintained by coding agents—is demonstrated empirically in this repo through LunarLander experiments.
