## Context

`hl-gobblet-env` 提供不可變局面、純函式 `legal_moves(state)` / `apply_move(state, move)` / `status_of(state)` / `line_winner(...)`，以及 opponent 介面 `reset(seed)` + `act(state) -> Move`。`hl-procedural-policy` 的 `hl_core` 提供 `RuleTable`、`FiniteStateMachine`、`MacroAction`、`ProceduralPolicy` 與唯讀 `decision_trace()`。

一個關鍵落差：`hl_core` 的 `ProceduralPolicy.act(observation)` 與 `FiniteStateMachine.step(observation, ctx)` 是為**向量觀測 → 整數動作**（lander）設計的；Gobblet 的決策是**離散結構化的**——輸入是 `GobbletState`、輸出是從 `legal_moves(state)` 中選出的某個 `Move`。本變更必須在不修改 `hl_core` 原始碼（其需求不變）的前提下，把同一組積木套到棋類上。

約束：HL 紅線——MUST NOT 出現任何梯度／神經網路／權重更新；所有「參數」皆為原始碼中的具名常數；不新增第三方依賴。依賴方向只能是「環境 import hl_core」，本控制器 import `hl_core` 與 `hl_gobblet`，`hl_core` 不得反向依賴。

## Goals / Non-Goals

**Goals:**
- 交付一個免訓練的 Gobblet 對手 `FsmGobbletV1`，棋力明顯強於 `RandomOpponent`（在多個 seed 的對打中勝率顯著過半），且決策可解釋。
- 以 FSM 切換兩個可讀具名模式 `aggressive` / `defensive`，切換條件為純函式的盤面威脅評估。
- 重用 `hl_core` 積木（RuleTable／FSM／Macro／ProceduralPolicy 的精神），驗證積木跨環境可用；不修改 `hl_core` 與 `hl_gobblet` 既有原始碼。
- 提供唯讀 `decision_trace()`，並對固定 seed 凍結 golden trace 納入回歸測試。
- 接進 `watch_match.py` 觀戰，可與 `random` 對打觀看。

**Non-Goals:**
- 不做任何學習、梯度、神經網路、MCTS、minimax 深層搜尋（最多一層「一手致勝／一手致敗」的純規則前瞻，不做樹搜尋）。
- 不改寫 `hl-gobblet-env` 的環境介面或勝負判定；不擴張 `hl-procedural-policy` 範圍。
- 不追求最佳解／不證明必勝；只要可解釋且穩定勝過隨機。
- 觀戰腳本不做統計或寫報告（維持既有「只觀看」約束）。

## Decisions

### 決策 1：把 hl_core 的決策積木「適配」到棋類，而非修改 hl_core

`hl_core` 的 `FiniteStateMachine` / `RuleTable` 簽章是泛型的 `(observation, ctx)`——`observation` 可為任意物件。因此把 `observation` 直接傳 `GobbletState`、把 `ctx` 帶上「該局面的合法步清單與威脅評估快取」即可重用 FSM 的「轉移表 + 每個 state 綁定子策略 + 具名當前 state」骨架，無需改動 `hl_core`。

- **問題**：`hl_core` 的 `action_fn` 與 macro 序列在 lander 是回傳整數動作；Gobblet 要回傳 `Move`。
- **取捨**：`action_fn(observation, ctx) -> action` 的 `action` 同樣是泛型；回傳 `Move` 完全合法。Macro（定長序列）對棋類意義不大（每步合法步集合會變），故本控制器**只用 RuleTable + FSM 兩種積木**，不綁 MacroAction。這仍在 `hl-procedural-policy` 宣告的積木範圍內（規則表／FSM／macro 三選用其二）。
- **替代方案**：(a) 改 `hl_core` 讓它「棋類友善」——否決，會擴張並污染 `hl-procedural-policy`；(b) 完全手刻一個平行 FSM——否決，違背「重用積木、驗證跨環境」的目的。

### 決策 2：兩模式以「一手致勝／一手致敗」威脅評估切換

每個 ply 開始時，對 `legal_moves(state)` 做一層純規則前瞻，算出兩個布林：
- `i_can_win`：存在某合法步，套用後 `status_of` 判我方勝。
- `opp_can_win`：對「對手視角的當前威脅」評估——盤面上是否存在某條線，對手只差一手即可連成（即對手最上層在某線已有兩格、第三格可被對手合法佔用）。

FSM 轉移規則（具名、可讀）：
- 進入 `defensive`：當 `opp_can_win` 為真且 `i_can_win` 為假（擋不過就先擋）。
- 進入 `aggressive`：當 `i_can_win` 為真（有殺先殺），或 `opp_can_win` 為假（無立即威脅就進攻）。
- 初始 state = `aggressive`。

- **取捨**：`i_can_win` 用「實際套用每個合法步再 `status_of`」精準判定（線數小、3×3，成本可忽略）；`opp_can_win` 用「線威脅計數」啟發式，避免昂貴的對手回合模擬，也避免把 `reveal_loses` 規則的複雜度引入切換層。
- **替代方案**：模擬對手最佳回應（minimax 一層半）——否決，成本與複雜度都拉高，且本目標只需穩定勝過隨機。

### 決策 3：各模式的子策略為「依 priority 排序的純函式 RuleTable」

兩個 state 各綁一個 `RuleTable`（沿用 `hl_core.RuleTable`，priority 數字越小越優先，命中第一條為真的 guard）：

- `aggressive` 規則（高到低優先）：
  1. `win_now` — 若某合法步可立即連線，下它。
  2. `block_then_gobble` — 若對手有一手致勝的格，且我方能用「更大子吃在那格」或「佔住第三格」阻斷，下阻斷步（攻擊模式仍需保命）。
  3. `make_threat` — 走能讓自己同時逼近兩條線（製造雙重威脅）的步；以線潛力評分挑最高。
  4. `develop` — 否則下能佔據中心或保留大子手牌的「最穩健發展步」。
- `defensive` 規則（高到低優先）：
  1. `win_now` — 即便防守，能贏就贏。
  2. `block` — 擋掉／吃掉對手的致勝威脅（多個威脅時擋評分最高者）。
  3. `safe_develop` — 無威脅時下不易被反吃的保守步（避免把大子過早曝險）。

- `default_action`：每個 RuleTable 都宣告一個保底動作，但棋類無法用固定常數當保底（合法步隨局面變）。因此 `default_action` 設為一個**回呼**「在當前 `ctx.legal_moves` 中取決定性的第一個合法步」，確保任何局面都回傳合法 `Move`、永不丟例外。
- **替代方案**：default 直接 raise——否決，違反「沒有 guard 命中時回傳 default、不丟例外」的 `hl-procedural-policy` 精神。

### 決策 4：評分與選步皆為決定性純函式

所有「挑最高分的步」皆以 `(score 高到低, move_to_index 升冪)` 為穩定排序鍵，確保相同局面、相同 seed → 相同選擇。控制器持有 seed 僅為對齊 opponent 介面與未來可選的隨機平手打破（預設不啟用隨機，平手一律取最小 `move_to_index`）。`reset(seed)` 清空當前 state 回 `aggressive` 與 trace 緩衝。

### 決策 5：對齊 lander 的 controllers/ 與 trace/golden 慣例

新增 `src/hl_gobblet/controllers/`（含 `__init__.py`、`fsm_gobblet_v1.py`、決策輔助純函式模組）。`watch_match.py` 的 `_opponent_factory` 增加 `fsm` 分支回傳 `FsmGobbletV1`。`decision_trace()` 每步記錄：步序、當前 state 名稱、觸發規則名稱、輸出 Move（與其索引）。對固定 seed 的 `fsm`-vs-`random` 對局凍結 golden trace，放在 `tests/hl_gobblet/`。

## Risks / Trade-offs

- [`opp_can_win` 啟發式漏判某些威脅（例如需要先移動己方子才暴露的威脅）] → 緩解：威脅評估只求「不比隨機差且能擋常見一手殺」；對打回歸測試以勝率門檻（顯著過半）驗收，而非要求零失分。日後若要更強，另開 change 引入一層對手模擬，不在本範圍。
- [`reveal_loses` 變體下，「拿起即判負」可能讓某些 move 反而害己] → 緩解：`i_can_win` 與 `block` 的合法步評估一律經由 `apply_move(..., reveal_loses=ctx.reveal_loses)` 與 `status_of`，確保切換層與規則層都在當前變體規則下判定；controller 從 `ctx` 取得 `reveal_loses` 設定，預設關閉。
- [把 `GobbletState` 當 `observation` 傳進 `hl_core`，可能讓人誤以為 `hl_core` 依賴 `hl_gobblet`] → 緩解：型別耦合只發生在本 controller 模組（import 方向正確：controller import 兩者）；`hl_core` 原始碼零改動，回歸測試保證 `hl_core` 仍可獨立 import。
- [Macro 未被使用，可能被視為「沒完整示範積木」] → 取捨：棋類的可變合法步集合使定長 macro 語意薄弱；明確記為設計選擇，`hl-procedural-policy` 只要求三積木「對齊主題」，未要求每個示範都用滿三種。
- [golden trace 過脆，規則微調就壞] → 緩解：golden 只凍結「穩定後」的組裝（與 `hl-procedural-policy` 規定一致）；測試說明要求刻意行為變更需在 PR 內說明並更新 golden。

## Open Questions

- 勝率驗收門檻的具體數值（例如 100 局 fsm-vs-random，fsm 勝率 ≥ 70%?）——於 tasks/實作時依實測校準，先以「顯著過半」為規格門檻。
- `aggressive` 的 `make_threat` 評分函式細節（雙重威脅的權重）留待實作時以小規模自我對打微調，不影響本設計的介面與切換骨架。
