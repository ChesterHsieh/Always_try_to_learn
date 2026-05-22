## 1. 準備與骨架

- [x] 1.1 確認在 `heuristic-learning/.venv` 下執行（`./.venv/bin/python -c "import numpy"`），確認 `hl_core` 不需新增任何相依（純標準庫 + numpy）；若需 `uv sync` 則執行並用 `make hl-lander-deps-check`（或 `uv pip list`）反向確認無 TF / PyTorch
- [x] 1.2 建立 `heuristic-learning/src/hl_core/` 套件骨架（`__init__.py` 先留匯出佔位）與 `heuristic-learning/tests/hl_core/` 測試目錄（含 `golden/`）

## 2. trace 基礎型別（先於積木，因積木回傳它）

- [x] 2.1 在 `hl_core/trace.py` 定義 `TraceRecord`（`@dataclass(frozen=True)`：`step`、`state`、`rule`、`macro_active`、`action`）與窄 Protocol `Decider`（`decide(obs, ctx) -> tuple[int, TraceRecord]`）
- [x] 2.2 在 `tests/hl_core/test_policy_trace.py` 先寫 trace 不變式的失敗測試骨架（frozen 不可變、欄位齊全）

## 3. 規則表 RuleTable（TDD）

- [x] 3.1 在 `tests/hl_core/test_rules.py` 寫失敗測試：多規則命中取最高優先序、無命中回 `default_action` 且 trace rule="default"、純函式不變式（連兩次同 obs 同 action 且不就地修改 obs）—— 對應 spec 三個 scenario（RED）
- [x] 3.2 在 `hl_core/rules.py` 實作 `Rule(guard, action_fn, priority, name)`（frozen）與 `RuleTable.decide(obs, ctx) -> (action, TraceRecord)`，依 priority 升冪取第一個 guard 為真者，無命中回 default（GREEN）
- [x] 3.3 跑 `pytest tests/hl_core/test_rules.py` 全綠並重構（IMPROVE）

## 4. 巨集動作 MacroAction（TDD）

- [x] 4.1 在 `tests/hl_core/test_macros.py` 寫失敗測試：定長 macro 連續 N 步依序輸出且第 N 步後 `is_active()` 為 False、執行中 `interrupt` 為真時於第 k 步立即終止並交還控制權、reset 後進度歸零 —— 對應 spec scenario（RED）
- [x] 4.2 在 `hl_core/macros.py` 實作 `MacroAction`（frozen 設定：action 序列 + 可選 `interrupt`；進度由外部 runtime 狀態持有），暴露 `is_active()` / `next_action(obs, ctx)`（GREEN）
- [x] 4.3 跑 `pytest tests/hl_core/test_macros.py` 全綠並重構

## 5. 有限狀態機 FiniteStateMachine（TDD）

- [x] 5.1 在 `tests/hl_core/test_fsm.py` 寫失敗測試：condition 成立時轉移且 action 由轉移後 state 產生、無可用轉移則停留原 state、`reset(seed)` 後回 `initial_state` 且清空各 state（含 macro）進度 —— 對應 spec scenario（RED）
- [x] 5.2 在 `hl_core/fsm.py` 實作 `State`、`Transition`、`FiniteStateMachine`：每個 state 綁一個 `Decider`（RuleTable 或包了 macro 的 decider），`step(obs, ctx)` 先更新 state 再委派子策略並回傳 `(action, TraceRecord)`（GREEN）
- [x] 5.3 跑 `pytest tests/hl_core/test_fsm.py` 全綠並重構

## 6. ProceduralPolicy 組合層與 trace 導出（TDD）

- [x] 6.1 補完 `tests/hl_core/test_policy_trace.py` 失敗測試：`ProceduralPolicy` 為 `HeuristicPolicy` 子型別（`isinstance` runtime_checkable）、同 seed 兩次 `decision_trace()` 完全相同、`decision_trace()` 長度等於步數、中途呼叫 `decision_trace()` 不改變後續 `act` 輸出、回傳為不可變快照（RED）
- [x] 6.2 在 `hl_core/policy.py` 實作 `ProceduralPolicy`：實作 `reset(seed)`/`act(obs)`，內部持有一個可重置 runtime 狀態物件（FSM 當前 state、各 macro 進度），`act` 收集 `TraceRecord` 進唯讀 buffer 並只回 action；`decision_trace()` 回傳 tuple 快照（GREEN）
- [x] 6.3 確認 `HeuristicPolicy` 介面**未**因 trace 而新增 `update`/`learn`/`train`；`hl_core` 任一檔**未** import `hl_lander` 或具體環境（spec 解耦守門）
- [x] 6.4 更新 `hl_core/__init__.py` 匯出 `RuleTable`/`FiniteStateMachine`/`MacroAction`/`ProceduralPolicy`/`TraceRecord`，跑 `pytest tests/hl_core/` 全綠

## 7. LunarLander 示範組裝 fsm_macro_v1

- [x] 7.1 在 `hl_lander/controllers/fsm_macro_v1.py` 以 `hl_core` 積木組裝 `FsmMacroLanderV1`：descend / align / touchdown 三具名 state，descend+align 用 `RuleTable`（以 baseline_v1 的高度－速度－角度三段規則為等價起點），touchdown 綁定穩定噴射 `MacroAction`；實作 `HeuristicPolicy`
- [x] 7.2 在 `run.py` 的 `--controller` dispatch 表新增 `fsm_macro_v1 -> FsmMacroLanderV1`（不改動其他 controller 行為，runner 仍只用 `HeuristicPolicy` 介面）
- [x] 7.3 確認 `fsm_macro_v1.py` 的決策邏輯**全部**來自 `hl_core` 積木（無平行手刻狀態管理），且未修改不可變的 `baseline_v1.py`

## 8. 評估、對照與實驗紀錄

- [x] 8.1 跑 smoke：`./.venv/bin/python experiments/hl-lunar-lander/run.py --controller fsm_macro_v1 --seeds 5 --episodes 10`，確認 < 60 秒、印出 mean return/std/landing rate、無未捕捉例外或 zombie env
- [x] 8.2 跑四方對照（noop / random / baseline_v1 / fsm_macro_v1，各 5 seeds × 10 episodes），確認 `fsm_macro_v1` mean return ≫ noop、random 且 ≥ baseline_v1；若不達標依 spec 視為 bug 並調查（noop=-131、random=-184、baseline_v1=264.4、fsm_macro_v1=268.6 ✓）
- [x] 8.3 在 `experiments/hl-lunar-lander/REPORT.md` 新增 `## fsm_macro_v1 (<YYYY-MM-DD>)` section：執行指令、`gymnasium.__version__`、env id、seed 列表、episode 數、mean return、std、landing rate、日期、`git rev-parse HEAD`，並含一句以上「相對 baseline_v1 的進步／退步」因果說明；不覆寫其他歷史 section

## 9. Golden Trace 回歸守門

- [x] 9.1 凍結 `fsm_macro_v1` 行為後，跑固定 seed=0 單 episode 導出 `decision_trace()`，序列化（只存語意欄位：step/state/rule/macro_active/action）為 `tests/hl_core/golden/fsm_macro_v1.seed0.json`（165 records，涵蓋 descend/align/touchdown）
- [x] 9.2 在 `tests/hl_core/`（例如 `test_golden_fsm_macro_v1.py`）新增回歸測試：load golden 後逐 record 比對 `FsmMacroLanderV1` 在 seed=0 的實際 trace；故意微調規則驗證測試會 RED（確認守門有效）再還原（perturb deadband → 165→204 records，RED；還原 GREEN）
- [x] 9.3 跑 `pytest tests/hl_core/ --cov=src/hl_core --cov-report=term-missing` 確認全綠且 hl_core 覆蓋率 ≥ 80%（17 passed，coverage 97%）

## 10. 收尾驗證

- [x] 10.1 `openspec validate hl-procedural-policy --strict` 通過
- [x] 10.2 black / isort / ruff 格式化 `hl_core/` 與 `fsm_macro_v1.py`；確認無 `print()`（用 logging）、所有 function 有 type annotation（採 ruff format + check，全綠；source 無 print）
- [x] 10.3 自我檢查 HL 紅線：`hl_core/` 與 `fsm_macro_v1.py` 無 import `optax`/`jax.grad`/`flax.training`/`torch`（grep 全無）；`heuristic-learning/.venv` 經本專案直譯器與 `make hl-lander-deps-check` 雙重確認無 TF / PyTorch（早先 `uv pip list` 命中的 tensorflow-probability 來自 shell 誤啟用的 `learn-jax/.venv`，與本專案無關）
