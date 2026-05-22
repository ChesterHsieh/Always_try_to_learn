## ADDED Requirements

### Requirement: HL Lunar Lander 範圍宣告

此 capability SHALL 以 `gymnasium` 的 `LunarLander-v3`（discrete action 版本）為唯一目標環境，並 MUST 透過修改 controller 原始碼而非梯度訓練來迭代策略，與 `hl-research-roadmap` 的 HL 範式保持一致。

#### Scenario: 嘗試加入梯度訓練程式碼

- **WHEN** 任何 PR 在 `heuristic-learning/src/hl_lander/` 內 import `optax`、`torch.optim`、`flax.training`、`jax.grad` 或等效梯度工具
- **THEN** 該 PR 必須在 `proposal.md` 或 `design.md` 內明確說明這仍屬於 HL 範式（例如僅用於 MPC 內的 model 學習），否則 reviewer SHALL 拒絕

#### Scenario: 切換到非 LunarLander 環境

- **WHEN** 一個 task 想把 `hl_lander.env` 指向 `LunarLanderContinuous-v3` 或其他環境
- **THEN** 必須先開新的 OpenSpec change 修改本 capability 的 requirements，**不得**直接改程式碼

### Requirement: HeuristicPolicy 抽象介面

`heuristic-learning/src/hl_lander/policy.py` SHALL 定義一個名為 `HeuristicPolicy` 的抽象介面（可以是 `typing.Protocol` 或 `abc.ABC`），且該介面 MUST 僅包含 `reset(seed: int) -> None` 與 `act(observation: np.ndarray) -> int` 兩個方法。

#### Scenario: 新增一個 controller

- **WHEN** 開發者在 `controllers/` 內新增任何 `*.py`，其中包含 controller 類別
- **THEN** 該類別 MUST 實作 `HeuristicPolicy` 介面
- **AND** 不得**為了便利**加入 `update`、`learn`、`train` 等方法到 `HeuristicPolicy` 本身上

#### Scenario: runner 取得 policy

- **WHEN** `runner.py` 在 episode 開始前呼叫 `policy.reset(seed)`
- **THEN** policy 內部所有有狀態欄位（如 CPG 相位、MPC 規劃緩衝、CV 狀態機）MUST 被重新初始化，使同一 seed 必能重現同一 trajectory

### Requirement: Baseline Procedural Controller v1

`heuristic-learning/src/hl_lander/controllers/baseline_v1.py` SHALL 提供一個名為 `BaselineLanderV1` 的 controller，使用「高度－速度－角度」三段式規則（不訓練、不需要任何超參搜尋），並 MUST 在 5 個 seeds × 10 episodes 的評估下達到 mean return ≥ 0。本 controller 為 HL 迭代軌跡的起點；後續迭代版本（v2、v3…）由獨立 change 新增為 `baseline_v2.py`、`baseline_v3.py` 等新檔案，本 change 不負責 v2 之後。

#### Scenario: 跑 baseline_v1 smoke

- **WHEN** 開發者執行 `make hl-lander-smoke`（或 `heuristic-learning/.venv/bin/python experiments/hl-lunar-lander/run.py --controller baseline_v1 --seeds 5 --episodes 10`）
- **THEN** 程式 MUST 在合理時間內（< 60 秒於本地 CPU）完成
- **AND** 印出 mean return ≥ 0、std、landing rate 三項數字
- **AND** 不得**丟出未捕捉例外**或留下 zombie env

### Requirement: 實驗紀錄結構（多 section）

每次完整跑完任一 controller 的評估 MUST 在 `heuristic-learning/experiments/hl-lunar-lander/REPORT.md` 留下可重現的紀錄。REPORT.md SHALL 採用「每個 controller 版本各佔一個 section」的多 section 結構，section 標頭格式為 `## <controller-name> (<YYYY-MM-DD>)`，每段內容 MUST 至少包含：執行指令、`gymnasium.__version__`、env id、seed 列表、episode 數、mean return、std、landing rate、執行日期、執行當下的 git commit hash。對 `baseline_v{n}` 系列（n ≥ 2），該 section MUST 額外包含「與 `baseline_v{n-1}` 的 Δreturn 文字描述」。

#### Scenario: 完成一次評估

- **WHEN** runner 跑完所有 seeds 的所有 episodes
- **THEN** 必須在 `REPORT.md` 內新增（或更新）對應 controller 的 section，且不得覆寫其他 controller 的歷史 section
- **AND** 新 section MUST 同時記錄該次執行的 git commit hash（透過 `git rev-parse HEAD` 取得）

#### Scenario: 跑 baseline_v2 之後的 controller

- **WHEN** 未來開新 change 加入 `baseline_v2.py` 並執行評估
- **THEN** REPORT.md 必須出現 `## baseline_v2 (<日期>)` section
- **AND** 該 section MUST 包含一句以上對「相對 baseline_v1 的進步／退步」之文字描述（不要求量化置信區間，但必須是可被讀者理解的因果說明）

### Requirement: Controller 版本不可變性

任何已合入 `main` 的 `controllers/baseline_v{n}.py`（或其他 `_v{n}` 系列檔案）MUST 被視為不可變的歷史版本，後續 HL 迭代 MUST 開新檔案 `_v{n+1}.py`，MUST NOT 就地修改 `_v{n}.py` 的策略行為。允許的就地修改僅限於：補 docstring、修正不影響行為的 typo、補 type hint。

#### Scenario: 開發者想直接改 baseline_v1.py 的規則邏輯

- **WHEN** 一個 PR 嘗試修改 `baseline_v1.py` 內任何會改變 `act()` 輸出分佈的程式碼
- **THEN** 該 PR MUST 被 reviewer 退回
- **AND** 開發者應改為新增 `baseline_v2.py`（複製 v1 為起點再修改）

#### Scenario: baseline_v1.py 內僅修正 typo

- **WHEN** 一個 PR 僅修改 `baseline_v1.py` 內的 docstring、註解、type hint 或無行為差異的格式
- **THEN** 該 PR 可以被接受
- **AND** 但仍 MUST 在 PR 描述明確標註「無行為變動」且 reviewer 應 spot-check `git diff` 確認

### Requirement: 弱對照組 Controllers

`heuristic-learning/src/hl_lander/controllers/` SHALL 提供兩個強制存在的弱對照 controllers：`random.py` 內的 `RandomLander`（每步從 4 個離散 action 均勻抽樣）與 `noop.py` 內的 `NoOpLander`（永遠輸出 action 0）。兩者 MUST 實作 `HeuristicPolicy` 介面，且 MUST 能被 `run.py --controller {random|noop}` dispatch。

#### Scenario: 跑三方對照

- **WHEN** 開發者依序執行 `--controller noop`、`--controller random`、`--controller baseline_v1`，各 5 seeds × 10 episodes
- **THEN** REPORT.md MUST 分別出現三個獨立 section
- **AND** `baseline_v1` 的 mean return MUST 明顯高於 `noop` 與 `random` 兩者（規則 controller ≫ 弱對照）；若違反，視為 baseline_v1 或 runner 有 bug，必須調查
- **AND** `noop` 與 `random` 之間**不**要求嚴格排序——在 LunarLander 上垂直墜落（noop）可能優於亂噴引擎（random），兩者皆為弱下界即可

#### Scenario: random / noop 試圖讀取 observation 以外的環境內部狀態

- **WHEN** PR 在 `random.py` 或 `noop.py` 內 import `gymnasium.envs.box2d` 內部模組或讀 env 的私有屬性
- **THEN** 該 PR MUST 被退回——對照組存在的意義就是「不看 obs（noop）」或「均勻抽（random）」，引入額外資訊會破壞對照

### Requirement: 獨立 venv 與 JAX-only 生態系限制

`heuristic-learning/` SHALL 擁有自己的 `pyproject.toml` 與獨立 venv（建議路徑 `heuristic-learning/.venv`），用 `uv` 管理。本 capability 的所有 Python 程式 MUST 透過該 venv 執行，且該 venv 的 dependencies MUST 僅依賴 JAX 生態系（jax、jaxlib、flax、optax、gymnax、gymnasium 等），MUST NOT 直接或間接依賴 `tensorflow`、`tensorflow-*`（含 `tensorflow-probability`）、`torch`、`torchvision`、`torchaudio`、`pytorch-lightning` 等 TF / PyTorch 套件。注意：`distrax` 雖屬 JAX 生態系，但會 transitive 拉進 `tensorflow-probability`，故**不得**列入本 capability 的 dependencies（純規則 controller 不需要機率分布）。

#### Scenario: 加入新相依

- **WHEN** 任一 task 需要新增 Python 套件（例如 `gymnasium[box2d]`）
- **THEN** 該套件必須加到 `heuristic-learning/pyproject.toml`，並執行 `uv sync` 安裝到 `heuristic-learning/.venv`
- **AND** 該 task 必須驗證 `uv pip list` 輸出不包含 `tensorflow`、`torch` 或它們的衍生套件
- **AND** 若該套件的 transitive dependencies 會拉進 TF / PyTorch，必須改用 extras-free 變體或拒絕加入

#### Scenario: 執行任何 Python 程式

- **WHEN** 開發者或 CI 在本 capability 範圍內執行任何 `*.py`
- **THEN** 直譯器 MUST 是 `heuristic-learning/.venv/bin/python`，不得使用系統 Python、`learn-jax/.venv` 或 conda 環境
