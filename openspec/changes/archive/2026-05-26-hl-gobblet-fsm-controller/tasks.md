## 1. 套件骨架與威脅評估純函式

- [x] 1.1 新增 `heuristic-learning/src/hl_gobblet/controllers/__init__.py`，匯出 `FsmGobbletV1`（對齊 `hl_lander/controllers/` 慣例）
- [x] 1.2 新增決策輔助模組（如 `controllers/_assess.py`）：純函式 `i_can_win(state, reveal_loses) -> bool`、`opponent_winning_lines(state) -> tuple[...]`、`opp_can_win(state) -> bool`，皆不就地修改 `state`
- [x] 1.3 先寫 `tests/hl_gobblet/test_fsm_assess.py`（RED）：覆蓋「我方一手致勝」「對手一手致勝」「無威脅」三類局面，斷言評估結果；確認測試先失敗
- [x] 1.4 實作 1.2 直到 1.3 通過（GREEN），確認評估在 `reveal_loses` 開／關下皆依當前變體規則判定

## 2. 兩模式 RuleTable 子策略

- [x] 2.1 新增候選步評分純函式（如 `controllers/_score.py`）：線潛力／雙重威脅評分、保守發展評分，平手以 `move_to_index` 升冪打破
- [x] 2.2 組裝 `aggressive` 的 `hl_core.RuleTable`：依序 `win_now` → 阻斷對手致勝 → `make_threat` → `develop`，`default_action` 為「ctx 合法步取決定性第一步」的保底回呼
- [x] 2.3 組裝 `defensive` 的 `hl_core.RuleTable`：依序 `win_now` → `block` → `safe_develop`，同樣設保底
- [x] 2.4 先寫 `tests/hl_gobblet/test_fsm_rules.py`（RED）：斷言「能贏就贏優先」「攻擊模式仍擋致命威脅」「無規則命中走 default 且不丟例外」「平手分數取最小索引」
- [x] 2.5 實作 2.1–2.3 直到 2.4 通過（GREEN）

## 3. FSM 組裝與 FsmGobbletV1 控制器

- [x] 3.1 以 `hl_core.FiniteStateMachine` 組裝：初始 state `aggressive`、兩 state 各綁對應 RuleTable、轉移依 `i_can_win`/`opp_can_win`（傳入的 observation 為 `GobbletState`，ctx 帶合法步、威脅評估、`reveal_loses`）
- [x] 3.2 實作 `controllers/fsm_gobblet_v1.py` 的 `FsmGobbletV1`：`reset(seed)`（重設 state 為 `aggressive`、清空 trace）、`act(state) -> Move`、唯讀 `decision_trace()`（步序、state 名稱、觸發規則名稱、Move 與索引）
- [x] 3.3 確認 `act` 永不就地修改傳入 `state`、永遠回傳 `legal_moves(state)` 中的 Move
- [x] 3.4 先寫 `tests/hl_gobblet/test_fsm_controller.py`（RED）：斷言「opponent 介面相容」「兩模式切換三情境」「合法性保證」「reset 後可重現」「trace 確定性與無副作用」
- [x] 3.5 實作 3.1–3.3 直到 3.4 通過（GREEN）

## 4. 觀戰整合與棋力／golden 回歸

- [x] 4.1 修改 `experiments/hl-gobblet/watch_match.py` 的 `_opponent_factory`：新增 `fsm` 分支回傳 `FsmGobbletV1`；不改動「只觀看」既有行為
- [x] 4.2 手動驗證 `--p0 fsm --p1 random --seed 0`（及另一 seed）可跑完一整局並顯示結果，相同 seed 重現相同序列
- [x] 4.3 新增 `tests/hl_gobblet/test_fsm_vs_random.py`：一組固定 seed 對打多局，斷言 `FsmGobbletV1` 對 `RandomOpponent` 勝率顯著過半（依實測校準門檻，例如 ≥ 70%）
- [x] 4.4 對固定 seed 的 `fsm`-vs-`random` 對局產生並凍結 golden `decision_trace()`，加入回歸測試（任何改動使其變化需 PR 說明並更新 golden）

## 5. 收尾與驗證

- [x] 5.1 跑 `heuristic-learning/.venv/bin/python -m pytest tests/hl_gobblet/`，全部通過
- [x] 5.2 確認本變更未修改 `hl_core/` 與 `hl_gobblet/` 既有檔案（僅新增 controllers/ 與測試、調整觀戰 factory），且 `hl_core` 仍可獨立 import
- [x] 5.3 確認 HL 紅線：無 `optax`/`jax.grad`/`flax`/`torch` import、無樹搜尋、所有參數為具名常數
- [x] 5.4 跑 lint 與全測：本專案工具鏈僅含 `ruff`（無 black/isort/mypy），`ruff check` 全綠；`ruff format` 非本專案強制（既有檔案亦不符其風格），新程式碼維持與周邊一致的寬行風格
