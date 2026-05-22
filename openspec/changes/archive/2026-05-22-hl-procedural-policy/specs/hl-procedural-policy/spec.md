## ADDED Requirements

### Requirement: 程序化策略範圍宣告

`hl-procedural-policy` capability SHALL 僅以**純程式碼構成的程序化策略積木**為範圍：規則表（rule table）、有限狀態機（finite state machine）、巨集動作（macro-action），對齊 `hl-research-roadmap` 的 procedural-policy 主題與 HL 範式。本 capability MUST NOT 包含任何梯度訓練、神經網路權重更新，且 MUST NOT 實作 MPC（model predictive control）、CPG（rhythmic gait）或電腦視覺啟發式——上述各自另開 capability。

#### Scenario: 嘗試加入梯度或神經網路

- **WHEN** 任何 PR 在 `heuristic-learning/src/hl_core/` 內 import `optax`、`jax.grad`、`flax.training`、`torch` 或等效梯度／權重更新工具
- **THEN** 該 PR MUST 被 reviewer 退回，因為 procedural-policy 的策略必須完全由可讀程式碼分支構成

#### Scenario: 嘗試把 MPC / CPG 塞進本 capability

- **WHEN** 一個 task 想在 `hl_core/` 內加入滾動視窗最佳化（MPC）或相位振盪器（CPG）
- **THEN** 開發者 MUST 改為新開 `hl-mpc` 或 `hl-cpg` change，**不得**擴張本 capability 範圍

### Requirement: 與環境無關且與既有 capability 解耦

程序化策略積木 SHALL 放在 `heuristic-learning/src/hl_core/`，MUST NOT import 任何特定環境模組（例如 `hl_lander`、`gymnasium.envs.box2d` 內部模組）。積木 MUST 只依賴標準函式庫與 `numpy`，使任何 `hl-*` 環境皆可直接 import 重用。

#### Scenario: hl_core 反向依賴特定環境

- **WHEN** PR 在 `hl_core/` 任一檔案 import `hl_lander` 或某個具體環境的型別
- **THEN** 該 PR MUST 被退回——依賴方向只能是「環境 import hl_core」，不可反向

#### Scenario: 另一個環境重用積木

- **WHEN** 未來新增 `hl_ant` 或 `hl_breakout` 環境並 import `hl_core` 的規則表 / FSM / macro
- **THEN** 該 import MUST 成功且不需修改 `hl_core` 任何原始碼

### Requirement: 規則表（Rule Table）

`heuristic-learning/src/hl_core/rules.py` SHALL 提供一個 `RuleTable`，由一組依 priority 排序的規則組成，每條規則為 `(guard, action_fn)`：`guard(observation, ctx) -> bool` 與 `action_fn(observation, ctx) -> action` 皆 MUST 為**無副作用的純函式**。`RuleTable.decide(observation, ctx)` MUST 回傳「第一條 guard 為真之規則」的 action，並 MUST 同時回報被觸發規則的識別名稱以利 trace。

#### Scenario: 多條規則命中時取最高優先序

- **WHEN** observation 同時滿足 priority=10 與 priority=20 兩條規則的 guard（數字越小越優先）
- **THEN** `decide` MUST 回傳 priority=10 那條的 action
- **AND** trace 紀錄的觸發規則名稱 MUST 是 priority=10 那條

#### Scenario: 沒有任何 guard 命中

- **WHEN** 所有規則的 guard 對某 observation 皆回傳 False
- **THEN** `decide` MUST 回傳該 `RuleTable` 宣告的 `default_action`，而非丟出例外
- **AND** trace 紀錄的觸發規則名稱 MUST 標示為 `"default"`

#### Scenario: guard 或 action 試圖改動傳入物件

- **WHEN** 單元測試對同一 observation 連續呼叫 `decide` 兩次
- **THEN** 兩次回傳的 action MUST 完全相同，且傳入的 observation 物件 MUST NOT 被就地修改（驗證純函式不變式）

### Requirement: 有限狀態機（Finite State Machine）

`heuristic-learning/src/hl_core/fsm.py` SHALL 提供一個 `FiniteStateMachine`，包含：一組明確命名的 `State`、一個 `initial_state`、一張 transition 表（`(from_state, condition) -> to_state`），以及每個 state 綁定的子策略（可為 `RuleTable` 或 macro）。`FiniteStateMachine.step(observation, ctx)` MUST 先依 transition 表更新當前 state，再委派當前 state 的子策略產生 action；當前 state MUST 是可讀的具名變數。

#### Scenario: 條件成立時發生狀態轉移

- **WHEN** FSM 處於 state A，且某條 `(A, condition)` 的 condition 對當前 observation 成立
- **THEN** `step` 之後當前 state MUST 變為該 transition 指定的目標 state
- **AND** 該步回傳的 action MUST 由**轉移後**的 state 子策略產生

#### Scenario: 無可用轉移時停留原狀態

- **WHEN** FSM 處於 state A 且沒有任何由 A 出發的 transition condition 成立
- **THEN** 當前 state MUST 維持為 A
- **AND** action 由 A 的子策略產生

#### Scenario: reset 後狀態回到初始

- **WHEN** 對 FSM 呼叫 `reset(seed)`
- **THEN** 當前 state MUST 回到 `initial_state`，所有 state 內部緩衝（含其綁定的 macro 進度）MUST 清空，使同 seed 可重現同一 trajectory

### Requirement: 巨集動作（Macro-Action）

`heuristic-learning/src/hl_core/macros.py` SHALL 提供一個 `MacroAction`，把「一段定長或帶終止條件的 action 序列」封裝成單一決策單元。`MacroAction` MUST 暴露 `is_active()`、`next_action(observation, ctx)`、以及一個可選的中斷條件 `interrupt(observation, ctx) -> bool`。當 macro 正在執行時被觸發 `interrupt` 為真，MUST 立即終止並交還控制權，而非耗盡整段序列。

#### Scenario: macro 正常執行到序列結束

- **WHEN** 一個長度為 N 的定長 macro 被觸發且 interrupt 從未成立
- **THEN** 連續 N 次 `next_action` MUST 依序回傳序列中的 N 個 action
- **AND** 第 N 次之後 `is_active()` MUST 回傳 False

#### Scenario: 執行中被中斷條件終止

- **WHEN** macro 執行到第 k 步（k < N）時 `interrupt` 對當前 observation 回傳 True
- **THEN** macro MUST 立即標記為非 active，第 k 步之後不再產生序列中的後續 action
- **AND** 控制權 MUST 交還給呼叫它的 FSM 或規則表

#### Scenario: macro 被 FSM 觸發

- **WHEN** FSM 進入一個綁定 macro 的 state 並呼叫 `step`
- **THEN** 只要 macro `is_active()`，後續每步 action MUST 來自 macro 的 `next_action`
- **AND** macro 結束或被中斷後，FSM MUST 能依 transition 表離開該 state

### Requirement: ProceduralPolicy 組合介面與決策軌跡導出

`heuristic-learning/src/hl_core/policy.py` SHALL 提供 `ProceduralPolicy`，其 MUST 實作既有 `hl_lander` 的 `HeuristicPolicy` 介面（`reset(seed)`、`act(observation)`），且其決策內部 MUST 由規則表 / FSM / macro 積木組裝。`ProceduralPolicy` MUST 額外提供一個**唯讀**的決策軌跡導出 `decision_trace()`，回傳每一步的結構化紀錄（至少含：步序、當前 state 名稱、觸發規則名稱、是否處於 macro、輸出 action）。trace 導出 MUST NOT 改變策略行為。

#### Scenario: 不污染既有介面

- **WHEN** runner 以既有 `HeuristicPolicy` 型別持有一個 `ProceduralPolicy` 並呼叫 `reset` / `act`
- **THEN** 行為 MUST 與其他 controller 一致，runner MUST NOT 需要知道 `ProceduralPolicy` 的存在即可運作
- **AND** `HeuristicPolicy` 介面本身 MUST NOT 因為 trace 功能而新增 `update`/`learn`/`train` 等方法

#### Scenario: 決策軌跡可被導出且與行為一致

- **WHEN** 對同一 seed 跑完一整個 episode 後呼叫 `decision_trace()`
- **THEN** trace 長度 MUST 等於該 episode 的步數
- **AND** 對同一 seed 跑兩次，兩次的 `decision_trace()` MUST 完全相同（確定性可重現）

#### Scenario: trace 導出不得有副作用

- **WHEN** 在 episode 中途呼叫 `decision_trace()`
- **THEN** 後續 `act` 的輸出 MUST 與「未曾呼叫過 trace」的情況完全相同

### Requirement: 回歸測試與 Golden Trace

`hl-procedural-policy` 的每一類積木（規則表、FSM、macro）與 `ProceduralPolicy` MUST 在 `heuristic-learning/tests/hl_core/` 下有對應的單元測試；任何示範組裝（如 `hl-lunar-lander` 的 `fsm_macro_v1`）一旦穩定，其在固定 seed 下的 `decision_trace()` MUST 被存成 golden trace 並納入回歸測試，對齊原文「舊能力轉成回歸測試以防退化」。

#### Scenario: 跑 hl_core 單元測試

- **WHEN** 開發者執行 `heuristic-learning/.venv/bin/python -m pytest tests/hl_core/`
- **THEN** 測試 MUST 至少涵蓋：規則表優先序與 default、FSM 轉移與 reset、macro 中斷、`ProceduralPolicy` 確定性與 trace 不變式
- **AND** 全部 MUST 通過

#### Scenario: 既有能力的 golden trace 退化

- **WHEN** 一個 PR 改動 `hl_core` 導致某個已凍結組裝在固定 seed 下的 `decision_trace()` 與 golden trace 不符
- **THEN** 回歸測試 MUST 失敗
- **AND** 開發者 MUST 在 PR 內明確說明這是刻意的行為變更並更新 golden trace，否則 reviewer SHALL 退回
