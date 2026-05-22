## Why

`hl-lunar-lander-scaffold` 已落地純規則 HL controller（`baseline_v1`，mean +264 / 68% 登陸），目前 REPORT.md 的 RL 對照只能引用文獻分數（PPO ~280、DQN ~250）。火力展示要有說服力，需要一條「同 env、同 JAX 生態系、本 repo 自己跑出來」的 RL baseline，與 HL 規則 controller 並排比較。本 change 用 JAX 手寫 minimal PPO（必要時加 DQN）在同一個 gymnasium `LunarLander-v3` 上訓練，產出 checkpoint，再透過既有 `HeuristicPolicy` 介面 rollout，把 RL 線寫進同一份 REPORT.md。

**與 HL 典範的關係（meta spec 範圍檢查）**：RL **不是** HL 方法本身——它是**對照基準**。HL 的命題是「不訓練神經網路也能達到 RL 等級」，因此 demo *必須* 有一個梯度訓練的 RL 對照才能量化「不訓練付出多少代價」。本 change 的梯度訓練僅用於產生對照基準，HL 主線（`controllers/baseline_v*.py`）仍保持 gradient-free。此例外明確列在 `## Out of Scope` 與 design.md。

## What Changes

- 新增 `heuristic-learning/src/hl_lander/rl/` 子模組：JAX/flax 手寫 minimal **PPO**（actor-critic MLP），gymnasium env loop + jit 過的 update step。CPU 即可訓練。
- 訓練 entrypoint `experiments/hl-lunar-lander/train_rl.py`：跑訓練、存 flax checkpoint 到 `experiments/hl-lunar-lander/checkpoints/`（已在 `.gitignore`）。
- 新增 `controllers/rl_ppo.py`：`RLPPOController` 載入 checkpoint，實作 `HeuristicPolicy.act()`（greedy / argmax over policy logits），讓 RL policy 走**既有 runner**，落進同一份 REPORT.md。
- `run.py` 的 `--controller` 增加 `rl_ppo` 選項。
- REPORT.md 補一個 `## rl_ppo` section，並把「對照組」表格從三方擴成四方（noop / random / baseline_v1 / rl_ppo）。
- **不**引入 `torch` / `tensorflow`；沿用 repo 級 JAX-only 契約（`hl-roadmap-drop-shared-venv` 提升的那條）。
- **可選（stretch）**：若 PPO 不穩，加 minimal DQN（`rl/dqn.py` + `controllers/rl_dqn.py`）；標為 stretch task，非 apply 必要。

## Capabilities

### New Capabilities
- `hl-lander-rl-baseline`: 在同一個 `LunarLander-v3` 上以 JAX 手寫 minimal PPO 訓練的 RL 對照基準，透過 `HeuristicPolicy` 介面與 HL 規則 controller 同場比較。

### Modified Capabilities
<!-- 不修改 hl-lunar-lander 的既有 requirements；RL controller 沿用其 HeuristicPolicy 介面與 runner -->

## Impact

- **新檔案**：`src/hl_lander/rl/{__init__.py, ppo.py, networks.py}`、`src/hl_lander/controllers/rl_ppo.py`、`experiments/hl-lunar-lander/train_rl.py`。
- **依賴**：只用既有 JAX 生態系（jax、jaxlib、flax、optax）；optax 此處**確實用於梯度訓練**（PPO 的 Adam update），這是與 HL 主線唯一的差異點，已在 design 說明。
- **介面契約**：RL controller MUST 實作 `HeuristicPolicy`，不得擴張該介面（訓練在離線 `train_rl.py`，不在 `act()` 內）。
- **產物**：checkpoint 落在 `experiments/hl-lunar-lander/checkpoints/`（gitignored），REPORT.md 記錄 checkpoint 雜湊與訓練步數。
- **meta spec**：本 change 觸碰 `hl-research-roadmap` 的「研究範圍宣告」requirement 之邊界（梯度訓練）；以「對照基準」身分通過範圍檢查，不修改該 requirement，僅在本 change 文件聲明例外。
- **不影響**：HL 主線 controller、`learn-jax/`、其他子專案。
