## 1. RL 模組骨架

- [x] 1.1 建立 `heuristic-learning/src/hl_lander/rl/__init__.py`（空 package marker；無副作用 import）。
- [x] 1.2 建立 `heuristic-learning/src/hl_lander/rl/networks.py`：flax actor-critic MLP（共享 trunk 或分離皆可），輸入 8 維 obs，輸出 4-way policy logits + 標量 value。
- [x] 1.3 建立 `heuristic-learning/src/hl_lander/rl/ppo.py`：minimal PPO（rollout buffer、GAE、clipped surrogate、value loss、entropy bonus、optax Adam），update step 用 `jax.jit` 包；env step 走 gymnasium 單環境 Python 迴圈（Box2D 不可 jit）。

## 2. 訓練 entrypoint

- [x] 2.1 建立 `heuristic-learning/experiments/hl-lunar-lander/train_rl.py`：CLI（`--total-steps`、`--seed`、`--lr`、`--out`），跑 PPO 訓練，用 `flax.serialization` 存 checkpoint 到 `experiments/hl-lunar-lander/checkpoints/`（已 gitignored）。
- [x] 2.2 訓練過程印出/記錄簡易學習曲線摘要（每 N steps 的 eval mean return），收尾印出 checkpoint 路徑與 flax 版本。

## 3. RL controller（inference-only）

- [x] 3.1 建立 `heuristic-learning/src/hl_lander/controllers/rl_ppo.py`：`RLPPOController` 實作 `HeuristicPolicy`。`reset(seed)` 載入 checkpoint params（若未載入過），`act(obs)` 做 network forward + argmax。checkpoint 不存在時丟清楚錯誤，提示先跑 `train_rl.py`。
- [x] 3.2 確認 `RLPPOController` 不在 `act()` 內做任何梯度更新，且不需要擴張 `HeuristicPolicy` 介面。

## 4. 串接既有實驗管線

- [x] 4.1 在 `experiments/hl-lunar-lander/run.py` 的 controller dispatch 加 `rl_ppo` → `controllers.rl_ppo.RLPPOController`，並把 `rl_ppo` 加進 `--controller` 的 choices。
- [x] 4.2 跑 `run.py --controller rl_ppo --seeds 5 --episodes 10`，把 `## rl_ppo` section append 進 REPORT.md（含訓練步數、flax 版本、checkpoint 識別）。

## 5. REPORT 與對照表

- [x] 5.1 把 REPORT.md「對照組」表格從三方擴成四方（noop / random / baseline_v1 / rl_ppo），用實際數字。
- [x] 5.2 在 REPORT.md「RL 對照組（未來工作）」段落改寫成「RL 對照組（已落地）」，替換掉文獻口頭對照，標明這是本 repo 自己訓練的結果。

## 6. 達標驗收

- [x] 6.1 確認 RL 評估 mean return ≥ 200；未達標則調整訓練（步數/lr/GAE），**不**改評估設定粉飾；若仍不收斂，誠實記錄實際分數與限制。
- [x] 6.2 `make hl-lander-deps-check` 確認沒有偷渡 torch/tensorflow。
- [x] 6.3 `openspec validate hl-lander-jax-rl-baseline --strict` 通過。

## 7. Stretch（非 apply 必要）

- [x] 7.1 若 PPO 不穩或想多一條對照：建立 `rl/dqn.py`（replay buffer + target network）與 `controllers/rl_dqn.py`，比照 PPO 串進 run.py 與 REPORT。標為 stretch，不阻擋本 change 完成。
  - **決議：略過（skipped）**。PPO 已穩定收斂至官方評估 mean +245.8（≥ 200 目標、且與 `baseline_v1` +264 同級），觸發 DQN 的前提（「PPO 不穩」）不成立；本 stretch task 明文非 apply 必要，為維持 change 範圍而不實作。日後若要多一條 RL 對照可獨立補上。
