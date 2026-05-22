## Context

`hl-lunar-lander-scaffold` 提供了 `HeuristicPolicy` 介面（窄：`reset(seed)` + `act(obs) -> int`）、`runner.evaluate`、`metrics.summarize`、多 section REPORT.md，以及三個 controller（baseline_v1 / random / noop）。HL 主線刻意 gradient-free。本 change 要在**不破壞**這個骨架的前提下，補一條梯度訓練的 RL 對照基準。

關鍵限制：purejaxrl（`learn-jax/purejaxrl/`）只支援 gymnax env，餵不進 gymnasium 的 Box2D LunarLander（前一輪已確認）。env 已鎖定 Box2D / CPU，所以 RL 也得自己寫一份吃 gymnasium 的 train loop。

## Goals / Non-Goals

**Goals:**

- 用 JAX/flax 手寫 minimal PPO（actor-critic MLP），在同一個 gymnasium `LunarLander-v3` 上訓練到「能與 HL 規則 controller 同級」（目標 mean ≥ 200）。
- 訓練產物（flax params）透過既有 `HeuristicPolicy` 介面 rollout，落進同一份 REPORT.md，與 HL 並排比較。
- 全程 CPU、JAX-only，不引入 torch/tensorflow。

**Non-Goals:**

- 不追 RL SOTA、不做大規模 hyperparameter sweep（minimal 即可）。
- 不把 RL 變成 HL 主線的一部分——它是對照基準，HL controller 仍 gradient-free。
- 不向量化大規模平行環境（gymnasium Box2D 不可 jit，single-env loop 即可；CPU 上 LunarLander PPO 數十萬步約數分鐘）。
- 不擴張 `HeuristicPolicy` 介面（訓練在離線 entrypoint，`act()` 只做 inference）。

## Decisions

### Decision 1: 訓練（離線）與 rollout（線上）分離；RL controller 只做 inference

`train_rl.py` 跑梯度訓練、存 checkpoint。`controllers/rl_ppo.py` 的 `RLPPOController` 在 `__init__`/`reset` 載入 checkpoint，`act(obs)` 對 policy network 做一次 forward + argmax。這樣 RL policy 走**既有 runner**，與 HL controller 完全同條評估路徑，比較才公平；同時 `HeuristicPolicy` 介面不需要長出 `update()`。

**Alternatives 考量：**

- 在 `act()` 內 online 訓練：會逼介面長出 `update()`，破壞 HL「介面只做 inference」的設計。淘汰。
- 訓練與 rollout 用不同 env 設定：比較失去公平性。淘汰。

### Decision 2: env loop 走 gymnasium（不 jit env），只 jit policy/update

Box2D 不可 jit。所以 rollout 用 Python 迴圈跑 gymnasium，把收集到的 transitions 堆成 batch 後，餵進 **jit 過的** PPO update step（loss + optax Adam）。瓶頸是 env step（C 層 Box2D），不是 forward pass，所以不 jit env 可接受。

**Alternatives 考量：**

- 自己寫 JAX-native LunarLander 讓全程 jit：前一輪評估過，工作量等於另一個 capability，且 sim-to-sim gap 風險高。淘汰（env 已鎖 Box2D）。
- 用 purejaxrl：只吃 gymnax，餵不進 Box2D。淘汰。

### Decision 3: 先 PPO，DQN 列為 stretch（非 apply 必要）

PPO 對連續/離散都穩、實作量適中（actor-critic + GAE + clip）。DQN 需要 replay buffer + target network，多一層狀態管理。先把 PPO 跑出對照數字；DQN 留 stretch task，PPO 若不穩再補。

**Alternatives 考量：**

- 先 DQN：LunarLander 的經典 DQN baseline 知名，但實作 replay/target 較囉嗦，且 PPO 已足以建立對照。延後。

### Decision 4: 梯度訓練的範圍例外，明文聲明而非修改 meta spec

`hl-research-roadmap` 的「研究範圍宣告」MUST 排除「以梯度下降訓練神經網路為主軸」的方法，但其 scenario 允許「在 proposal/design 明確說明關聯或列入 Out of Scope」。RL baseline 的梯度訓練**不是主軸**——主軸是 HL，RL 只是量化對照。因此**不修改** meta spec，僅在本 change 聲明此為對照基準例外。

**Alternatives 考量：**

- 修改 meta spec 開「RL 對照」白名單：過度一般化，且會弱化「本 repo 是 HL 研究」的定位。淘汰——用 per-change 聲明即可。

## Risks / Trade-offs

- **[風險] CPU 上 PPO 訓練太慢或不收斂**　→　**緩解**：minimal PPO + 合理 default（~5e5 steps、GAE λ=0.95、clip 0.2）；REPORT 記錄實際 wall-clock 與訓練曲線摘要；不收斂則縮 env step 或調 lr，仍 CPU 可跑。
- **[風險] checkpoint 格式 / flax 版本漂移**　→　**緩解**：用 flax 內建 serialization（`flax.serialization`），checkpoint 連同 flax 版本記進 REPORT；checkpoint gitignored，靠 `train_rl.py` 可重跑。
- **[風險] 觀眾質疑「你不是說 gradient-free 嗎，怎麼又訓練了」**　→　**緩解**：demo 與 REPORT 明確標示 rl_ppo 是「對照基準」，HL 主線 controller 全程無梯度；這正是 demo 想凸顯的對比。
- **[Trade-off] minimal PPO 可能略低於文獻 PPO**：可接受——對照目的是「同 repo 同 env 的 RL 上限約略多少」，不是刷榜。
