# HL Lunar Lander — 實驗報告

本檔案由 `run.py` 以 append 方式維護：每次評估新增一個 `## <controller> (<日期>)` section，**不覆寫**其他 controller 的歷史紀錄。

## 版本不可變性說明

每個 `baseline_v{n}` 是 HL 迭代軌跡上的一個獨立 first-class artefact。一旦合入 `main` 即視為不可變——要改進策略就新增 `baseline_v{n+1}.py`，不就地修改既有版本。這讓 demo 能一頁並排比較各版本程式碼，呼應 HL「程式碼結構就是策略本身」的命題。

## 對照組

四方對照（弱對照與 `baseline_v1` 為 2026-05-20，`rl_ppo` 為 2026-05-21；皆 5 seeds × 10 episodes）：

| controller   | mean return | std   | landing rate | 訓練？        |
|--------------|-------------|-------|--------------|---------------|
| `noop`       | -131.0      | 41.7  | 0%           | 無            |
| `random`     | -183.9      | 63.7  | 0%           | 無            |
| `baseline_v1`| **+264.4**  | 37.0  | **68%**      | 無（手寫規則）|
| `rl_ppo`     | +245.8      | 81.9  | 58%          | 有（梯度訓練）|

`baseline_v1`（手寫規則、零訓練）與 `rl_ppo`（JAX 手寫 PPO、200 萬步梯度訓練）**同級**：mean +264 vs +245，兩者都遠勝兩個弱對照（-131 / -183）。這正是 HL 命題的量化展示——**不訓練神經網路，也能達到 RL 等級**。

值得注意的反差：`rl_ppo` 的 mean return 雖與 `baseline_v1` 相當，但 std 較大（81.9 vs 37.0）、landing rate 反而較低（58% vs 68%）。RL 策略傾向「高效收集 reward」（精準懸停/定位），但在「雙腳觸地 + 正報酬 + terminated」這個嚴格登陸判定上觸發較少；規則 controller 則更穩定地完成乾淨著陸。亦即：**RL 拿到的分數略高但更不穩、且不必然對應教科書式的軟著陸**。

`baseline_v1` 大幅勝過兩個弱對照（+264 vs -131 / -183），且 68% episodes 成功登陸，遠超 spec smoke 門檻（mean ≥ 0）。

**關於 spec 中「noop < random」的預期排序**：實測 `noop`（-131）反而**優於** `random`（-183）。這不是 bug——LunarLander 上「什麼都不做、垂直墜落」的懲罰，比「亂噴側向引擎把船甩飛出場」要小。有意義的論述是「規則 controller ≫ 兩個弱對照」，此點強烈成立；spec scenario 的 `noop < random` 過於武斷，列為 follow-up（待修 spec 或接受兩個弱對照無嚴格排序）。

## RL 對照組（已落地）

RL baseline 已由 change `hl-lander-jax-rl-baseline` 落地：**本 repo 自己訓練的** JAX/flax 手寫 minimal PPO（actor-critic MLP、GAE、clipped surrogate、optax Adam；全程 CPU、JAX-only，**不**引入 torch/tensorflow），在同一個 `LunarLander-v3` 上訓練 200 萬步，官方評估（seeds 0–4 × 10 episodes、greedy/argmax）得 **mean +245.8**。這不再是文獻口頭對照，而是同 repo、同 env、同一條 runner 路徑跑出來的實測數字。

- **訓練／推論分離**：梯度訓練只在離線 `experiments/hl-lunar-lander/train_rl.py` 與 `src/hl_lander/rl/`；線上 rollout 走 `controllers/rl_ppo.py` 的 `RLPPOController`，其 `act()` 僅做 network forward + argmax，**不**做任何梯度更新，也未擴張 `HeuristicPolicy` 介面。HL 主線 controller（`baseline_v*.py`）維持 gradient-free。
- **範圍定位**：`rl_ppo` 是**對照基準**，不是 HL 方法本身。它存在的意義是量化「不訓練付出多少代價」——而結論是：代價極小，手寫規則的 `baseline_v1`（+264）與梯度訓練的 `rl_ppo`（+245）打平。
- **可重現**：`python experiments/hl-lunar-lander/train_rl.py --total-steps 2000000 --seed 0`（checkpoint gitignored，靠此指令重跑；checkpoint 雜湊記於下方 `## rl_ppo` section）。訓練採 best-eval checkpoint 選取（以 train_rl 內部 eval_env，seed offset 與官方 seeds 0–4 不同，故非粉飾官方評估）。

---

<!-- 以下為 run.py 於 2026-05-20 append 的原始 section -->

## noop (2026-05-20)

- 執行指令：`python experiments/hl-lunar-lander/run.py --controller noop --seeds 5 --episodes 10`
- gymnasium：1.3.0
- env id：LunarLander-v3
- seeds：[0, 1, 2, 3, 4]
- episodes/seed：10（共 50 episodes）
- mean return：-131.0
- std return：41.7
- landing rate：0.00%
- git commit：9582f8b4b737fe1981c414668f5a32a2c89bd27f

## random (2026-05-20)

- 執行指令：`python experiments/hl-lunar-lander/run.py --controller random --seeds 5 --episodes 10`
- gymnasium：1.3.0
- env id：LunarLander-v3
- seeds：[0, 1, 2, 3, 4]
- episodes/seed：10（共 50 episodes）
- mean return：-183.9
- std return：63.7
- landing rate：0.00%
- git commit：9582f8b4b737fe1981c414668f5a32a2c89bd27f

## baseline_v1 (2026-05-20)

- 執行指令：`python experiments/hl-lunar-lander/run.py --controller baseline_v1 --seeds 5 --episodes 10`
- gymnasium：1.3.0
- env id：LunarLander-v3
- seeds：[0, 1, 2, 3, 4]
- episodes/seed：10（共 50 episodes）
- mean return：264.4
- std return：37.0
- landing rate：68.00%
- git commit：9582f8b4b737fe1981c414668f5a32a2c89bd27f

## rl_ppo (2026-05-21)

- 執行指令：`python experiments/hl-lunar-lander/run.py --controller rl_ppo --seeds 5 --episodes 10`
- gymnasium：1.3.0
- env id：LunarLander-v3
- seeds：[0, 1, 2, 3, 4]
- episodes/seed：10（共 50 episodes）
- mean return：245.8
- std return：81.9
- landing rate：58.00%
- git commit：9582f8b4b737fe1981c414668f5a32a2c89bd27f
- 訓練步數：2000000
- 訓練 seed：0
- flax：0.10.4
- checkpoint：ppo_lunarlander.msgpack (sha256:d8783c097a16)

## fsm_macro_v1 (2026-05-21)

- 執行指令：`python experiments/hl-lunar-lander/run.py --controller fsm_macro_v1 --seeds 5 --episodes 10`
- gymnasium：1.3.0
- env id：LunarLander-v3
- seeds：[0, 1, 2, 3, 4]
- episodes/seed：10（共 50 episodes）
- mean return：268.6
- std return：52.4
- landing rate：58.00%
- git commit：9582f8b4b737fe1981c414668f5a32a2c89bd27f
- 組裝：以 `hl_core` 的 RuleTable + FiniteStateMachine + MacroAction 組成（descend → align → touchdown），非單檔 if-else
- 相對 baseline_v1（mean 264.4 / std 37.0 / landing 68%）的進步／退步：mean return 略升（+4.2，264.4 → 268.6），達成 spec「mean ≥ baseline_v1」的硬性門檻。關鍵在新增的 `align` 階段「先擺正姿態、僅在快速下墜時點主引擎」：迭代過程中（見 design），未加 align 的版本 mean 僅 238、std 高達 109；加入 align 後 std 收斂到 52.4，把原本拖累平均的高變異墜毀 episode 拉回，才讓 mean 反超 baseline。代價是 landing rate 由 68% → 58%（以「更穩、更高的平均回報」換「略低的著陸率」）；後續 v2 可在保住 mean 的前提下重拾著陸率（design Open Questions 已標註 align 煞車閾值為調參點，本次取 0.15）。
- 四方對照（同 5 seeds × 10 episodes）：noop mean=-131.0、random mean=-183.9、baseline_v1 mean=264.4、fsm_macro_v1 mean=268.6 —— 程式碼組合策略 ≫ 弱對照且 ≥ baseline_v1，滿足 spec scenario。
