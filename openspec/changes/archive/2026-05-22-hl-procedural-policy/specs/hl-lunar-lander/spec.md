## ADDED Requirements

### Requirement: 程序化策略示範 Controller（fsm_macro_v1）

`heuristic-learning/src/hl_lander/controllers/fsm_macro_v1.py` SHALL 提供一個名為 `FsmMacroLanderV1` 的 controller，**建構於 `hl_core` 的 `ProceduralPolicy` 之上**（規則表 + 有限狀態機 + 巨集動作），用以示範「以程式碼組合的策略可被迭代並超越單檔規則」。此 controller MUST 實作既有 `HeuristicPolicy` 介面，MUST 能被 `run.py --controller fsm_macro_v1` dispatch，且 MUST NOT 修改不可變的 `baseline_v1.py`。狀態機 SHALL 至少包含 descend / align / touchdown 三個具名 state，touchdown 階段 SHALL 由一個穩定噴射 macro 處理。

#### Scenario: 跑 fsm_macro_v1 smoke

- **WHEN** 開發者執行 `heuristic-learning/.venv/bin/python experiments/hl-lunar-lander/run.py --controller fsm_macro_v1 --seeds 5 --episodes 10`
- **THEN** 程式 MUST 在 < 60 秒於本地 CPU 完成
- **AND** 印出 mean return、std、landing rate 三項數字
- **AND** 不得丟出未捕捉例外或留下 zombie env

#### Scenario: 示範策略勝過弱對照且不劣於 baseline_v1

- **WHEN** 依序跑 `--controller noop`、`--controller random`、`--controller baseline_v1`、`--controller fsm_macro_v1`，各 5 seeds × 10 episodes
- **THEN** `fsm_macro_v1` 的 mean return MUST 明顯高於 `noop` 與 `random`
- **AND** `fsm_macro_v1` 的 mean return SHALL ≥ `baseline_v1`（作為「程式碼組合策略可迭代進步」的證據；若低於，視為組裝或 runner 有 bug 必須調查）

#### Scenario: 透過 ProceduralPolicy 組裝而非另寫一坨 if-else

- **WHEN** reviewer 檢視 `fsm_macro_v1.py`
- **THEN** 其決策邏輯 MUST 由 `hl_core` 的 `RuleTable` / `FiniteStateMachine` / `MacroAction` 組裝而成，MUST NOT 在 controller 內重新手刻一套與 `hl_core` 平行的狀態管理
- **AND** 該 controller MUST 能透過 `ProceduralPolicy.decision_trace()` 導出每步的 state / 規則 / macro 紀錄

### Requirement: 示範 Controller 的實驗紀錄與 Golden Trace

跑完 `fsm_macro_v1` 評估 MUST 在 `heuristic-learning/experiments/hl-lunar-lander/REPORT.md` 依既有多 section 規範新增 `## fsm_macro_v1 (<YYYY-MM-DD>)` section，且 MUST 包含一句以上「相對 baseline_v1 的進步／退步」之文字描述。穩定後，`fsm_macro_v1` 在固定 seed 下的 `decision_trace()` MUST 被存為 golden trace 並納入 `hl-procedural-policy` 的回歸測試。

#### Scenario: 完成一次 fsm_macro_v1 評估

- **WHEN** runner 跑完 `fsm_macro_v1` 所有 seeds 的所有 episodes
- **THEN** REPORT.md MUST 出現 `## fsm_macro_v1 (<日期>)` section，含執行指令、`gymnasium.__version__`、env id、seed 列表、episode 數、mean return、std、landing rate、執行日期、git commit hash
- **AND** MUST NOT 覆寫其他 controller 的歷史 section
- **AND** 該 section MUST 含一句以上對「相對 baseline_v1」的因果說明

#### Scenario: 凍結後的 golden trace 守門

- **WHEN** `fsm_macro_v1` 被視為穩定並產生 golden trace 後，某 PR 改動使其在固定 seed 下的 `decision_trace()` 改變
- **THEN** `tests/hl_core/` 的回歸測試 MUST 失敗，迫使開發者明確聲明行為變更並更新 golden trace
