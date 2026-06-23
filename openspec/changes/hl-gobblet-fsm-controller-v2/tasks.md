## 1. v2 威脅評估純函式（fork 偵測）

- [x] 1.1 新增 `heuristic-learning/src/hl_gobblet/controllers/_assess_v2.py`：純函式 `i_can_fork(state, reveal_loses) -> bool`、`opp_can_fork(state) -> bool`，以 `apply_move`（immutable）展開一手後盤面、用「兩子且第三格可合法 claim 的線數 ≥ 2」判定；重用既有 `_assess.opponent_winning_lines` 風格的 claim 判定，皆不就地修改 `state`、不展開對手回手樹
- [x] 1.2 先寫 `tests/hl_gobblet/test_fsm_v2_assess.py`（RED）：覆蓋「我方可造雙殺」「對手即將造雙殺」「無雙重威脅回報為假」三類局面，斷言 `i_can_fork`／`opp_can_fork`；確認測試先失敗
- [x] 1.3 實作 1.1 直到 1.2 通過（GREEN），確認 `reveal_loses` 開／關下皆依當前變體規則判定

## 2. v2 選步評分純函式（fork 獎勵 + gobble 得失）

- [x] 2.1 新增 `controllers/_score_v2.py`：在 v1 線潛力／中心／保留／暴露項之上新增具名常數 `_FORK_BONUS`（一手後我方可 claim 威脅線數 ≥ 2 加成）、`gobble_value`（覆蓋對手 top 解除其威脅線加分）、`reveal_penalty`（`MOVE` 搬起自家子後揭露對手可立即成線重罰），平手以 `move_to_index` 升冪打破
- [x] 2.2 先寫 `tests/hl_gobblet/test_fsm_v2_rules.py`（RED）：斷言「覆蓋對手大子解除威脅的步評分較高」「搬移後揭露對手線的步被懲罰」「造 fork 的步因 `_FORK_BONUS` 勝出」「平手取最小索引」；確認先失敗
- [x] 2.3 實作 2.1 直到 2.2 通過（GREEN）

## 3. v2 模式與規則表（aggressive / defensive / setup_fork）

- [x] 3.1 新增 `controllers/_modes_v2.py`：以 `hl_core.RuleTable` 組裝三模式——`aggressive`(`win_now`→`block_then_gobble`→`make_fork`→`make_threat`→`develop`)、`defensive`(`win_now`→`block`/`block_fork`→`safe_develop`)、`setup_fork`(`win_now`→`block`→`commit_fork`→`develop`)；各表 `default_action` 為「ctx 合法步取決定性第一步」保底；新增 v2 專屬 `GobbletCtxV2`，`refresh` 一併快取 `i_can_fork`／`opp_can_fork`
- [x] 3.2 先寫 `tests/hl_gobblet/test_fsm_v2_modes.py`（RED）：斷言「能贏就贏優先（規則名 `win_now`）」「防守模式擋掉對手雙重威脅」「無規則命中走 `default` 不丟例外」「平手分數取最小索引」；確認先失敗
- [x] 3.3 實作 3.1 直到 3.2 通過（GREEN）

## 4. FSM 組裝與 FsmGobbletV2 控制器

- [x] 4.1 新增 `controllers/fsm_gobblet_v2.py`：以 `hl_core.FiniteStateMachine` 組裝初始 `aggressive`、三 state 各綁對應 RuleTable，轉移依 `i_can_win`/`opp_can_win`/`i_can_fork`（任一 state 在 `i_can_win` 走致勝；`opp_can_win and not i_can_win`→`defensive`；`not opp_can_win and not i_can_win and i_can_fork`→`setup_fork`；威脅消解回 `aggressive`/`defensive`）
- [x] 4.2 實作 `FsmGobbletV2`：`reset(seed)`（重設 state 為 `aggressive`、清空 trace）、`act(state) -> Move`（永不就地修改 `state`、永遠回傳 `legal_moves(state)` 中的 Move）、唯讀 `decision_trace()`（步序、state 名稱含 `setup_fork`、觸發規則名稱、Move 與索引）
- [x] 4.3 於 `controllers/__init__.py` 匯出 `FsmGobbletV2`（與 `FsmGobbletV1` 並列，不改動 v1 匯出）
- [x] 4.4 先寫 `tests/hl_gobblet/test_fsm_v2_controller.py`（RED）：斷言「opponent 介面相容」「三模式切換情境（含進入 `setup_fork`）」「合法性保證」「reset 後可重現」「trace 確定性與無副作用」；確認先失敗
- [x] 4.5 實作 4.1–4.2 直到 4.4 通過（GREEN）

## 5. 棋力門檻：v2 對 v1 的誠實門檻（一手前瞻下）

- [x] 5.1 新增 `tests/hl_gobblet/_matchup.py`：`play_match(p0, p1, seed, opening_plies) -> Status | None`（交手前走 seeded 隨機開局製造分佈）、`winrate(a, b, seeds) -> float`（輪流先後手，決定性）
- [x] 5.2 新增 `tests/hl_gobblet/test_fsm_v2_vs_v1.py`：固定 seed × seeded 隨機開局 × 雙先後手，斷言 (a) v2 對 v1 ≥ 平手 floor 0.40、(b) v2 對隨機 ≥ v1 對隨機 − 0.05、(c) v2 顯式 fork 行為（進入 setup_fork/commit_fork、一手內把對手站立可 claim 線壓到 < 2）
- [x] 5.3 **實作量測修正**：在 HL 紅線（嚴格一手前瞻）內加 fork 偵測/防 fork、gobble、揭露懲罰、站立危險、位置壓力評分並做權重 grid search，量測證實 v2 對 v1 頭對頭穩定 ~41–51%（上限 ~46.9%）——該局先手優勢 ~73.5% 且 v1 已近一手最優，無法可靠 ≥70%；經使用者確認改為「不劣於 v1 + 顯式 fork 行為」的誠實門檻（見 design D6 與 spec）

## 6. 3×3 交互對打結果矩陣

- [x] 6.1 新增 `experiments/hl-gobblet/matchup_matrix.py`：對 `{random, v1, v2}` 每個有序配對（各輪先後手、seeded 隨機開局、固定 seed 集）統計勝率，印出 3×3 矩陣；決定性可重現、不寫檔
- [x] 6.2 新增 `tests/hl_gobblet/test_matchup_matrix.py`：斷言 v1／v2 對 random 皆顯著過半、v2-vs-random ≥ v1-vs-random − epsilon、v2-vs-v1 ≥ 平手 floor；相同輸入重現相同矩陣
- [x] 6.3 手動執行 `matchup_matrix.py`（seeds=100, opening=4）確認：random↔v1/v2 = 2.0%/2.5%、v1/v2 對 random = 98%/97.5%、v1↔v2 ≈ 50%/46.5%

## 7. 觀戰整合與 golden 回歸

- [x] 7.1 修改 `experiments/hl-gobblet/watch_match.py` 的 `_opponent_factory`：新增 `fsm-v2` 分支回傳 `FsmGobbletV2`（傳入 `reveal_loses`）；不改動「只觀看」既有行為
- [x] 7.2 手動驗證 `--p0 fsm-v2 --p1 fsm --seed 0`（及另一 seed）可跑完一整局並顯示結果，相同 seed 重現相同序列
- [x] 7.3 達到 5.2 門檻後，對固定 seed 的 v2 對局產生並凍結 golden `decision_trace()`（新增 `tests/hl_gobblet/_gen_golden_v2.py` 與 `golden/fsm_gobblet_v2.seed0.json`），加入 `test_golden_fsm_gobblet_v2.py` 回歸（任何改動使其變化需 PR 說明並更新 golden）

## 8. 收尾與驗證

- [x] 8.1 跑 `heuristic-learning/.venv/bin/python -m pytest tests/hl_gobblet/`，全部通過（含 v1 既有 golden 與 v2 新測試）
- [x] 8.2 確認本變更未修改 `hl_core/`、`hl_gobblet/` 既有檔案與 `fsm_gobblet_v1.py` 及 v1 用到的既有輔助模組；v1 golden trace 測試仍綠；`hl_core` 仍可獨立 import
- [x] 8.3 確認 HL 紅線：v2 程式碼無 `optax`/`jax.grad`/`flax`/`torch` import、無 MCTS/多層 minimax、fork 偵測僅看一手後盤面、所有「參數」為具名常數
- [x] 8.4 跑 `ruff check`（本專案僅含 ruff，無 black/isort/mypy）全綠，新程式碼維持與周邊一致風格
