## Why

「Learning Beyond Gradients」的啟發式學習（HL）典範主張：策略應由**可讀、可編輯、可測試的程式碼**構成，而非神經網路權重。目前 repo 只有 `hl-lunar-lander` 內單檔式的三段規則 `BaselineLanderV1`，缺乏可被各 `hl-*` 環境共用的程序化策略組合元件（規則表、狀態機、巨集動作）。這對應原文五大主題中的 **procedural-policy（rule-based / state machine controllers）**，且**不涉及任何梯度訓練**，完全落在 HL 範式內。現在做的理由：在進入 MPC / CPG 之前，先把「以程式碼組合策略」的共用骨架與不變式定義清楚，後續主題才能 fork 重用而非各自重造。

## What Changes

- 新增一個與環境無關的**程序化策略元件庫** `heuristic-learning/src/hl_core/`，提供三類可組合積木：
  - **規則表（rule table）**：以 priority 排序的 `(guard, action)` 條件列表，第一個滿足的 guard 決定 action；guard 與 action 皆為純函式、無副作用。
  - **狀態機（finite state machine）**：明確的 `State` 列舉 + transition 表，每個 state 綁定一個子策略（可為規則表或巨集動作）；狀態與轉移皆為可讀變數，符合原文「explicit readable variables」。
  - **巨集動作（macro-action）**：把一段定長或帶終止條件的 action 序列封裝成單一決策單元，支援被狀態機觸發、可被中斷（interrupt guard）。
- 定義 `ProceduralPolicy` 組合介面：實作既有 `HeuristicPolicy`（`reset`/`act`），但內部由上述積木組裝；新增**唯讀的決策軌跡導出**（trace export）方法，讓每一步的「觸發了哪條規則 / 處於哪個 state / 是否在執行 macro」可被記錄，服務原文強調的 explainability 與 regression test。
- 在 `hl-lunar-lander` 上提供一個**示範組裝** `controllers/fsm_macro_v1.py`（新檔，不修改不可變的 `baseline_v1.py`），用狀態機（descend → align → touchdown）＋巨集動作（穩定噴射序列）重現並超越 baseline_v1 的 mean return，作為「程式碼組合策略可被迭代」的最小證據。
- 把上述積木的行為冻結為 **regression 測試 + golden trace**，對齊原文「舊能力轉成回歸測試以防退化」。

## Capabilities

### New Capabilities
- `hl-procedural-policy`: 與環境無關的程序化策略組合層——規則表、有限狀態機、巨集動作三類積木，及其組合介面、決策軌跡導出、回歸測試骨架。對應原文 procedural-policy 主題，明確排除 MPC（另開 `hl-mpc` change）、CPG、CV heuristics。

### Modified Capabilities
- `hl-lunar-lander`: 新增一個建構於 `hl-procedural-policy` 之上的示範 controller（`fsm_macro_v1`）並納入三方對照與 REPORT.md 紀錄。此為**新增 controller 檔案**，不改動既有 `HeuristicPolicy` 介面語意，也不修改不可變的 `baseline_v1.py`；屬於在既有 requirements 允許範圍內的延伸（controller 以新檔加入），故僅以 delta 方式補充「示範組裝」相關的 scenario，不變更既有不變式。

## Impact

- **新增程式碼**：`heuristic-learning/src/hl_core/`（`rules.py`、`fsm.py`、`macros.py`、`policy.py`、`trace.py`），與既有 `hl_lander` 解耦、可被任何 `hl-*` import。
- **新增測試**：`heuristic-learning/tests/hl_core/`（規則表優先序、FSM 轉移、macro 中斷、trace 不變式、golden trace 回歸）。
- **新增實驗**：`experiments/hl-lunar-lander/` 下 `fsm_macro_v1` 的執行入口與 REPORT.md section。
- **依賴**：沿用 `hl-research-roadmap` 的 JAX-only / 獨立 venv 契約，**不新增** TF / PyTorch 套件；積木為純 Python + numpy，不需要新相依。
- **不影響**：`baseline_v1.py`、`random.py`、`noop.py`、既有 `HeuristicPolicy` 介面與既有 REPORT.md 歷史 section。
