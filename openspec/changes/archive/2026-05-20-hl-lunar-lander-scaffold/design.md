## Context

「Learning Beyond Gradients」（Trinkle 23897, 2024–2025）提出的 HL 典範強調**用程式碼結構而非神經網路權重承載策略**：agent 觀察環境→直接修改 controller 程式碼→重新評分→迭代。本 repo 的 `heuristic-learning/` 子專案已經宣告了這條研究路線（見 `hl-research-roadmap` spec），但目前 `src/` 與 `experiments/` 都是空的，下一個 capability 開出來會無處可掛。

LunarLander 雖然沒出現在原文（原文用 Ant、HalfCheetah、Atari、Montezuma、VizDoom），但它是 Box2D 經典控制環境，state（位置、速度、角度、角速度、兩腳接觸）8 維、action 4 維離散 or 2 維連續，遠比 Ant（27 維 state、8 維 action）小。把骨架先在 LunarLander 上跑通，後續加 CV heuristics、MPC、CPG 都不會被環境複雜度卡住。

主要利害關係人：HL 子專案的後續所有 capability 作者（會 fork 這套骨架），以及維護 `learn-jax/.venv` 的人（負責處理 Box2D 安裝）。

## Goals / Non-Goals

**Goals:**

- 提供一份 **stable, minimal** 的 HL controller 介面（`HeuristicPolicy`），讓後續主題不需要動 runner 就能插入新策略。
- 跑通一條 end-to-end pipeline：建立 env → instantiate policy → run N seeds → 寫 metrics → 產 REPORT.md。
- baseline procedural controller 只要能「比隨機策略好」即可（不追分數），目的是驗證骨架，而非演算法。
- 嚴格遵守 `hl-research-roadmap` 的實驗紀錄契約（結果寫進 `experiments/<capability>/REPORT.md`）。Python 環境契約則於本 capability 改成「自有 venv + JAX-only」（見 Decision 5）。
- **展示 HL 迭代軌跡**：以 `baseline_v1.py`、`baseline_v2.py`… 的方式保留每一版規則 controller 的完整程式碼，demo 時可一頁並排比較，呼應 HL「程式碼結構就是策略本身」的命題。
- **提供弱對照組**：`random` / `noop` controller，證明規則控制器不是運氣，並建立 REPORT 表格的下界。

**Non-Goals:**

- **不**做梯度訓練、**不**呼叫 `optax`/`flax` 進入訓練流程（即便 venv 內裝了 JAX 生態系，本 change 不用它跑 RL）。
- **不**追 LunarLander 的 SOTA 分數（>200 留給後續迭代 change 處理）。
- **不**做 LunarLanderContinuous——本 change 鎖定 discrete action 版本（`LunarLander-v3`，4 個離散動作），減少介面分支。
- **不**做 CV heuristics、MPC、CPG——這些是各自獨立 capability。
- **不**寫 hyperparameter sweep 工具——HL 的迭代主軸是改程式碼，不是調參。
- **不**做 RL 對照組（JAX 手寫 minimal PPO/DQN）——另開 `hl-lander-jax-rl-baseline` change 處理。本 change 的對照組僅止於 random / noop。
- **不**做自動 controller search 或 LLM-driven code edit——HL 迭代由人類在迴圈中編輯，自動化留待之後評估。

## Decisions

### Decision 1: 用 `gymnasium`（不是 `gym`）並指定 `LunarLander-v3`

`gymnasium` 是 OpenAI Gym 的維護分支，`gym` 已經 archive。`LunarLander-v3` 是目前最新版本（v2 在 gymnasium 0.29 之後 deprecated）。

**Alternatives 考量：**

- `gymnax`（已在 `learn-jax/.venv` 內）：不含 LunarLander，只有 classic control 的 JAX 化版本。淘汰。
- 自己寫一份簡化版 lander：成本高、不可信。淘汰。

### Decision 2: `HeuristicPolicy` 介面只暴露 `reset(seed)` + `act(observation) -> action`

刻意不要 `learn()` / `update()` 方法。**HL 的「學習」發生在原始碼層級**——agent 修改 `controllers/baseline.py` 的程式碼後重新執行，而不是在 process 內呼叫 `policy.update()`。介面越窄，後續 controller（CPG 振盪器、MPC planner、search agent）越容易塞進來。

**Alternatives 考量：**

- 加 `policy.update(transition)`：會引導使用者寫成 online RL，與 HL 典範衝突。淘汰。
- 完全函式式（`act(state, obs) -> (state, action)`）：對 JAX 純函式友善，但跟 gymnasium 的 imperative 風格不搭，且 CPG 這類有內部相位狀態的 controller 會被迫把 state 暴露在外。延後到真的需要時再切。

### Decision 3: Baseline controller 用「高度－速度－角度」三段式規則

從 LunarLander 的 obs 8 維中取角度與角速度做姿態控制，再用垂直速度決定主引擎，水平速度決定側向引擎。這是社群多年共識的最小規則 controller（不用任何超參搜尋就能拿到 ~100–150 分），足以證明骨架活著。

**Alternatives 考量：**

- 純隨機策略 baseline：跑得通但學不到「規則策略 > 隨機」這件事，不利於後續比較。淘汰。
- PID controller：比規則 baseline 強，但引入連續參數→誘導調參→偏離 HL 精神。延後到單獨 capability。

### Decision 4: 評分用 5 個 seeds 跑 10 episodes，取 mean ± std

LunarLander 隨機性主要來自起始狀態，5 個 seeds × 10 episodes = 50 次評分對 baseline 規則 controller 已足夠收斂。再多會拖慢 smoke test。

**Alternatives 考量：**

- 3 seeds × 5 episodes：太少，std 不穩定。
- 10 seeds × 100 episodes：smoke 從秒級變分鐘級，不適合做 CI gate。

### Decision 5: `heuristic-learning/` 自建 venv，但限 JAX 生態系（禁 TF / PyTorch）

放棄原本「共用 `learn-jax/.venv`」的契約。原因：火力展示需要本 repo 能單獨 clone / 單獨跑，依賴 sibling 目錄會讓 demo 開場就解釋一堆。改成 `heuristic-learning/` 自己 `pyproject.toml` + `uv` 管理 `heuristic-learning/.venv`。但**鎖一條紅線**：dependencies 只能在 JAX 生態系（jax / jaxlib / flax / optax / gymnax / gymnasium…），不准引入 `tensorflow*`（含 `tensorflow-probability`）或 `torch*`。實作時發現 `distrax` 會 transitive 拉進 `tensorflow-probability`，故已將其排除（純規則 controller 不需要機率分布）。這保留了「HL 是 gradient-free 範式」的精神紀律——TF/PyTorch 本身雖然不等於梯度訓練，但會誘導 contributor 隨手寫 `nn.Module` + `optim.Adam`，破壞 demo 的論述。

**Alternatives 考量：**

- 維持共用 `learn-jax/.venv`：使用者已明確放棄該契約，因 demo / 移植性需求。淘汰。
- 不限制生態系，全交給 reviewer 把關：火力展示風險太高，會被偷渡一個 PyTorch 出來。淘汰。
- 用 conda：本 repo 已用 `uv`，雙工具會混亂。淘汰。

**`hl-research-roadmap` 的處理：** 該 spec 內「共用 Python 環境契約」requirement 與此決策衝突，需在本 change archive 前另開一個 change 修改它（或在本 change 內加 delta spec）。本次先在本 capability 內以**新 requirement** 覆蓋（spec 層級的 precedence：capability spec 比 meta spec 更具體），後續再回頭收拾 meta spec。

### Decision 6: 目錄分層 `src/hl_lander/{policy,env,runner,metrics,controllers/}`

模組職責切清楚：

- `policy.py`：抽象介面（Protocol 或 ABC）。
- `env.py`：thin wrapper，處理 seed 與 action space 規格。
- `runner.py`：單一 episode loop、episode return。
- `metrics.py`：聚合 mean/std/landing rate。
- `controllers/baseline.py`：第一個具體 controller。

後續主題（CV、MPC、CPG）只在 `controllers/` 內加新檔案，其他不動。

### Decision 7: HL 迭代軌跡以「複製檔案」（`_v1`/`_v2`/...）承載，而非 git history

每一版 controller 開新檔案：`controllers/baseline_v1.py`、`baseline_v2.py`、`baseline_v3.py`…，一旦合入 main 就**永不就地修改**。

理由（demo 導向）：

- 本子專案的火力展示重點之一是 HL 迭代過程本身。Demo 簡報需要一頁並排顯示所有版本的 controller 程式碼讓觀眾看 diff——這在 git history 下要切視窗切指令，現場效果差。
- HL 在原文裡的核心命題是「程式碼結構就是策略本身」。讓多個版本作為**同時存在的 first-class artefacts**，比把舊版藏在 git history 更貼合這個命題。
- 磁碟成本：每版規則 controller < 5 KB，10 版以下完全可忽略。

**Alternatives 考量：**

- 單一檔案 + git tag：repo 看起來乾淨，但 demo 與 review 都要靠 `git diff <tag>..<tag>` 切換，現場與簡報都不直觀。淘汰。
- 單一檔案 + 同檔多 class（`BaselineV1`、`BaselineV2`…）：import 路徑一致，但檔案越長越難讀，且每次新增版本都會碰到舊版的程式碼區塊（合入 main 後其實也算「修改」舊版檔案）。淘汰。

**配套：**

- `run.py --controller baseline_v1` / `random` / `noop` 等以**檔名為 key** 做 dispatch。
- REPORT.md 以「## baseline_v1 (YYYY-MM-DD)」做 section 標頭，每段紀錄當時 git commit hash、mean ± std、landing rate、與前一版的 Δreturn 文字描述。

### Decision 8: RL 對照組獨立成 `hl-lander-jax-rl-baseline` change，本 change 只裝 random / noop

本 change 的對照組僅止於 `random`（均勻取 action）與 `noop`（永遠取 action 0）。RL 對照組以 JAX 手寫 minimal PPO/DQN（300–500 行）、跑在同一個 gymnasium env 上，另開獨立 change。

理由：

- 守住「先把框架打好」的原句——scaffold 範圍清楚、能單獨 ship。
- 任一子項出問題（最可能：JAX RL train loop 的 numerical stability、checkpoint I/O）不會牽動 scaffold 與 baseline 軌跡的進度。
- 對應 SDD 精神：每個 change 解一個問題；正好同時示範「change 是會迭代的」這個賣點。
- 對照觀眾體感：random / noop 已足以證明 `baseline_v1` 不是隨機抖出來的成績；RL baseline 是「同 env 同生態系 vs HL」的硬核對照，價值在後續 change 累積出來再做。

**Alternatives 考量：**

- 一次塞 JAX PPO 進本 change：proposal/design/tasks 顯著膨脹，scaffold ship 時間被 RL 進度拖長。淘汰。
- 用文獻 PPO ~280、DQN ~250 做口頭對照即可、永不寫 RL code：對 demo 是夠的，但放棄了「我能同 repo 同生態系跑 RL」這個可信度。**部分採用**：在本 change 的 REPORT.md 初版可暫時引用文獻分數作為 placeholder，等下一個 change 把 JAX RL 落地後再覆蓋成自己跑出來的數字。
- 預留 `controllers/rl_jax.py` interface stub：誘導讀者去填空，違反 YAGNI。淘汰。

## Risks / Trade-offs

- **[風險] Box2D 在 macOS Apple Silicon 安裝可能要 swig**　→　**緩解**：README 寫明 `brew install swig` 前置；CI/smoke 失敗時打印安裝提示。
- **[風險] 規則 baseline 太弱（<0 分）讓人誤以為框架壞了**　→　**緩解**：smoke test 設低門檻 mean return ≥ 0（不是 ≥ 200），失敗時 fixture 會印 episode trace。
- **[風險] `HeuristicPolicy` 介面太窄，CPG/MPC 需要內部狀態**　→　**緩解**：用 `reset(seed)` 重建內部狀態；如真不夠用，未來開新 change 擴展介面（不是現在猜）。
- **[風險] 某個 JAX 生態系套件的 transitive deps 偷渡 TF/PyTorch**（例：`tensorboard` 會拉 `tensorflow`）→　**緩解**：1.1 task 內驗證 `uv pip list` 不含禁用清單；CI 可加 `uv pip list | grep -E 'tensorflow|torch'` 反向 assertion。
- **[風險] 與 `hl-research-roadmap` 的共用 venv 契約衝突**　→　**緩解**：本 capability spec 以更具體層級覆蓋，並在本 change archive 前另開 change 修改 meta spec 的對應 requirement（或在本 change 補一份 modified delta）。
- **[風險] 版本檔案氾濫**：HL 迭代到 10 版以上時 `controllers/` 會有一長串 `baseline_v*.py`　→　**緩解**：超過 5 版時開新 change 評估歸檔策略（如 `controllers/archived/`），但本 change 不預先設計，避免 YAGNI。
- **[風險] 觀眾誤以為 `_v2` 就地修改了 `_v1`**　→　**緩解**：spec 明文「合入 main 後不可變」、`controllers/__init__.py` 加註解、REPORT.md 第一個 section 寫一段「版本不可變性」的說明。
- **[Trade-off] RL 對照組延後**：本 change ship 時 demo 只有 random / noop / HL 三方對照，沒有 RL 線。接受這個短期缺口，換取 scaffold 能單獨 ship；REPORT.md 暫以文獻 PPO ~280 做口頭對照 placeholder。
- **[Trade-off] 鎖 discrete action 版**：少一條連續動作分支，但失去示範「連續啟發式控制」的機會。為了骨架簡潔接受這個代價，連續版留給後續 capability。
- **[Trade-off] 不寫 JAX 版本**：失去 jit 速度，但 HL 的瓶頸從來不是 forward pass。等到真的要做 vectorized rollouts 再考慮。
