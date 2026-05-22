---
project: heuristic-learning
---

# hl-lander-rl-baseline Specification

## Purpose

在同一個 `LunarLander-v3` 上以 JAX 手寫 minimal PPO 訓練的 RL 對照基準，透過 `HeuristicPolicy` 介面與 HL 規則 controller 同場比較，量化「不訓練神經網路」相對於梯度訓練 RL 的代價。

## Requirements

### Requirement: RL 對照基準範圍與紅線

本 capability SHALL 在同一個 gymnasium `LunarLander-v3`（discrete action）上，用 JAX/flax 手寫 minimal PPO 訓練一個 RL 對照基準。其梯度訓練 MUST 僅作為**對照基準**存在，MUST NOT 滲入 HL 主線（`controllers/baseline_v*.py` 保持 gradient-free）。所有程式碼 MUST 維持 repo 級 JAX-only 契約，MUST NOT 引入 `torch*` 或 `tensorflow*`。

#### Scenario: 引入禁用框架

- **WHEN** 任何 PR 在本 capability 內 import `torch`、`tensorflow` 或其衍生套件
- **THEN** 該 PR MUST 被退回；RL 對照只能用 JAX 生態系（jax / jaxlib / flax / optax）實作

#### Scenario: 梯度訓練滲入 HL 主線

- **WHEN** 任何 PR 嘗試在 `controllers/baseline_v*.py` 內加入 optax/flax 梯度更新
- **THEN** 該 PR MUST 被退回；梯度訓練僅允許出現在 `src/hl_lander/rl/` 與 `experiments/hl-lunar-lander/train_rl.py`

### Requirement: 訓練與推論分離

訓練（梯度更新）MUST 發生在離線 entrypoint `experiments/hl-lunar-lander/train_rl.py` 與 `src/hl_lander/rl/`，產出 flax checkpoint。RL 對照的線上 rollout MUST 透過實作 `HeuristicPolicy` 介面的 `controllers/rl_ppo.py`（`RLPPOController`），其 `act(observation)` 僅做 inference（network forward + argmax），MUST NOT 在 `act()` 內做任何梯度更新。

#### Scenario: rollout RL policy

- **WHEN** 開發者執行 `run.py --controller rl_ppo --seeds 5 --episodes 10`
- **THEN** `RLPPOController` MUST 載入既有 checkpoint 並透過**既有 runner**（與 HL controller 同一條路徑）評估
- **AND** 若 checkpoint 不存在，MUST 給出清楚錯誤訊息，提示先跑 `train_rl.py`

#### Scenario: RL controller 嘗試擴張介面

- **WHEN** 任何 PR 想在 `HeuristicPolicy` 介面加 `update`/`learn`/`train` 以服務 RL
- **THEN** 該 PR MUST 被退回；RL 訓練狀態留在 `rl/` 模組，不污染 inference 介面

### Requirement: Minimal PPO 達標與可重現

`src/hl_lander/rl/ppo.py` 的 minimal PPO（actor-critic MLP、GAE、clipped objective、optax Adam）SHALL 在 CPU 上訓練至 mean return ≥ 200（5 seeds × 10 episodes 評估），訓練 MUST 可由 `train_rl.py` 以記錄的指令與 seed 重跑。

#### Scenario: 完成一次 RL 訓練與評估

- **WHEN** 開發者跑完 `train_rl.py` 並以 `run.py --controller rl_ppo` 評估
- **THEN** REPORT.md MUST 新增 `## rl_ppo (<日期>)` section，記錄：執行指令、gymnasium 版本、env id、seed 列表、mean ± std、landing rate、執行日期、git commit hash、訓練步數、flax 版本、checkpoint 識別（雜湊或路徑）
- **AND** 「對照組」表格 MUST 從三方（noop / random / baseline_v1）擴成四方，加入 rl_ppo

#### Scenario: PPO 未達標

- **WHEN** RL 評估 mean return < 200
- **THEN** MUST 調整訓練（步數 / lr / GAE），MUST NOT 改評估設定（seeds / episodes）來粉飾
- **AND** 若多次調整仍不收斂，MUST 在 REPORT.md 誠實記錄實際分數與已知限制，不得宣稱達標
