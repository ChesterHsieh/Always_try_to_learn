## Why

「Learning Beyond Gradients」一文示範了 HL（啟發式學習）在 Ant、HalfCheetah 等連續控制環境下，純靠手寫規則＋程式碼迭代就能逼近梯度型 RL 的成績。但目前本 repo 的 HL 研究只剩 `hl-research-roadmap` 這個 meta-spec，**還沒有任何一個 capability 真正落地**，使用者也想在 LunarLander 這個入門級連續控制環境上練手。LunarLander 比 Ant 維度小（state 8 維、action 4 維離散或 2 維連續），適合做為 HL 流程的「最小可跑」骨架，把目錄、介面、實驗紀錄、評分流程全部跑通，後續主題（CV heuristics、MPC、CPG）才能直接 fork 這套骨架。

## What Changes

- 新增 `heuristic-learning/src/hl_lander/` 目錄與 Python package 骨架（policy 介面、env adapter、runner、metrics）。
- 定義 `HeuristicPolicy` 抽象介面（`reset(seed)` / `act(observation) -> action`），讓後續所有 HL 子主題能套同一份 runner。
- 提供一個最小可跑的 baseline procedural policy（簡單的「高度－速度－角度」規則控制器），不訓練、不調參。
- **以「複製檔案」承載 HL 迭代軌跡**：controller 命名為 `baseline_v1.py`、`baseline_v2.py`…，一旦合入 main 即視為不可變的歷史版本，後續迭代必須開新檔案，不得就地修改。目的是讓 demo 能一頁並排所有版本看 diff，呼應 HL「程式碼結構就是策略本身」的精神。
- **新增弱對照組 controllers**：`controllers/random.py`（random action lower bound）、`controllers/noop.py`（trivial do-nothing baseline），用來證明規則 controller 不是隨機抖出來的。RL 對照組（JAX 手寫 minimal PPO/DQN）**不在本 change 範圍內**，另開 `hl-lander-jax-rl-baseline` change 處理。
- 新增 `experiments/hl-lunar-lander/run.py`、`REPORT.md` 範本（多 section 結構，每個 controller 版本各一段，含執行指令、git commit hash、mean ± std、landing rate、與前一版的 delta）。
- 為 `heuristic-learning/` 建立**獨立** `pyproject.toml` 與 `.venv`（用 `uv`），把 `gymnasium[box2d]` 與必要 JAX 生態系套件裝進去；明令**禁止** `tensorflow*` / `torch*` 進入這個 venv（防止 demo 期間有人偷渡 nn.Module + optimizer，與 HL 範式衝突）。
- 提供 `make hl-lander-smoke` 一鍵跑通 baseline，作為 CI／回歸的依據。

## Capabilities

### New Capabilities
- `hl-lunar-lander`: 在 `LunarLander-v3` 上以 HL 典範執行的最小框架，包含 policy 介面、env adapter、runner、baseline procedural controller 與實驗紀錄結構。

### Modified Capabilities
<!-- 沒有修改既有 capability 的 requirements；`hl-research-roadmap` 的契約被新 capability 沿用而非變更 -->

## Impact

- **新檔案**：`heuristic-learning/pyproject.toml`、`heuristic-learning/src/hl_lander/{__init__.py, policy.py, controllers/__init__.py, controllers/baseline_v1.py, controllers/random.py, controllers/noop.py, env.py, runner.py, metrics.py}`、`heuristic-learning/experiments/hl-lunar-lander/{run.py, REPORT.md}`、`heuristic-learning/Makefile`（新增 target）。
- **依賴**：`heuristic-learning/pyproject.toml` 列出 `jax`、`jaxlib`、`flax`、`optax`、`gymnax`、`gymnasium[box2d]`（含 `box2d-py`、`pygame` for render）等 JAX 生態系套件；**禁止** `tensorflow*`（含 `tensorflow-probability`）/ `torch*`（含 transitive）；**不列入 `distrax`**（會 transitive 拉進 tensorflow-probability，本 change 純規則 controller 用不到）；不引入 mujoco（後續 capability 再談）。
- **架構契約**：`HeuristicPolicy` 介面成為後續所有 `hl-*` capability 共用的 contract；改動需開新 change。
- **與既有 spec 衝突**：本 change 與 `hl-research-roadmap` 的「共用 Python 環境契約」requirement 衝突；本 capability spec 以更具體層級覆蓋，meta spec 的對齊另開 change 處理。
- **不影響**：`ai-monitor-system/`、`learn-jax/` 主程式、其他子專案；`learn-jax/.venv` 不再被本子專案使用。
