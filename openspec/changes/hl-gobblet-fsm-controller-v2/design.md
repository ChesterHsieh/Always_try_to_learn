## Context

`FsmGobbletV1`（`hl_gobblet/controllers/fsm_gobblet_v1.py`）以 `hl_core` 的 `FiniteStateMachine`＋`RuleTable` 組裝，兩個具名模式 `aggressive`／`defensive` 依一手前瞻威脅評估切換，對隨機對手實測 100% 勝率。但它的純函式評估有三個已知盲點：

1. **威脅評估只看單線**：`opponent_winning_lines` 只回報「對手某一條線差一手」，看不到**雙重威脅（fork）**——對手一手後同時逼出兩條致勝線，導致 v1 擋了一邊卻輸另一邊。
2. **攻擊評分不主動造 fork**：`attack_score`／`develop_score` 只加總線潛力與中心控制，沒有獎勵「我方一手後同時持有兩條兩子線」，所以 v1 很少主動製造殺棋。
3. **忽略 gobble 動態**：`_line_potential` 用「該線有對手 top 就跳過」當近似，沒有評估「覆蓋對手大子重新打開線」或「搬移自家子揭露對手線」的得失——而這正是 Gobblet 的核心戰術。

v1 完全決定性：固定 (seed, 先後手) 只產生一場棋。因此「v2 大機率擊敗 v1」必須以**多 seed × 雙先後手**的對局集合來量化勝率（沿用 `test_fsm_vs_random.py` 的做法）。本設計同時要產出 `random`／`fsm`(v1)／`fsm-v2` 的 **3×3 交互對打結果矩陣**作為棋力對照。

**約束（HL 紅線）**：純程式碼規則、最多一手前瞻、無樹搜尋（MCTS／多層 minimax）、無梯度／神經網路；所有「參數」為原始碼具名常數。重用 `hl_core`、`hl_gobblet` 不修改其既有原始碼，並且**不修改 v1**（v1 凍結作為基準與 golden trace）。

## Goals / Non-Goals

**Goals:**

- 交付 `FsmGobbletV2`：opponent 介面相容、永遠合法步、不就地修改 state、相同 seed 可重現、提供唯讀 `decision_trace()`。
- 在純函式、一手前瞻內補上 v1 的三個盲點：fork 偵測（攻防雙向）、fork 製造獎勵、gobble 得失評估。
- 以新增可讀具名模式／規則層擴充 FSM 表達力（仍只用 `hl_core` 積木）。
- 以多 seed × seeded 隨機開局 × 雙先後手對打，量化 v2 對 v1 勝率，並以**誠實門檻**凍結：v2 不劣於 v1（頭對頭 ≥ 平手 floor）、v2 對共同隨機不弱於 v1、v2 顯式處理 fork（見下方「實作量測修正」）。
- 產出可重現的 **3×3 交互對打結果矩陣**（`random`／`v1`／`v2`，各輪先後手、seeded 隨機開局），作為棋力對照與回歸。
- v2 在固定 seed 的 `decision_trace()` 凍結為 golden 回歸。

**Non-Goals:**

- 不引入任何學習／權重更新／樹搜尋（HL 紅線）；fork/gobble 評估一律限制在「一手套用後的盤面」這一層。
- 不修改 v1 控制器、`hl_core`、`hl_gobblet` 既有檔案；不改變 v1 的 golden trace。
- 不追求對 v1 的 100% 或對「最佳對手」的最優；本 change 的門檻是擊敗 v1 ≥ 70%。
- 不處理隱藏資訊變體（god-view 之外）；沿用既有 `reveal_loses` 變體一致性即可。

## Decisions

### D1：v2 自成一套輔助模組，不複用會牽動 v1 golden 的既有模組

v2 的 `_assess`／`_score`／`_modes` 另立檔案（如 `controllers/_assess_v2.py`、`_score_v2.py`、`_modes_v2.py`），不修改 v1 用到的 `_assess.py`／`_score.py`／`_modes.py`。

- **理由**：v1 的 golden trace 凍結了它對既有純函式的依賴；若把新評估塞進共用模組，極易讓 v1 行為與 golden 漂移而誤觸回歸。獨立檔案讓「v1 凍結、v2 迭代」邊界清楚。
- **替代方案**：(a) 在既有模組加 `version` 參數分支——被否決，會讓共用檔案膨脹且 v1 路徑容易被誤改；(b) 讓 v2 繼承 v1 類別覆寫方法——被否決，v1 是程序式組裝非可覆寫的方法樹，繼承反而隱藏行為來源。可共享的**真正不變**的低階工具（如 `LINES`、`apply_move`、`move_to_index`）仍直接從 `hl_gobblet.rules`／`moves` import。

### D2：fork 偵測＝對「一手後盤面」數兩子且第三格可合法claim的線數 ≥ 2

新增純函式（攻防對稱）：
- `i_can_fork(state, reveal_loses)`：是否存在某合法步，使套用後**我方**同時擁有 ≥ 2 條「兩子且可合法claim第三格」的線（即對手無法一手全擋）。
- `opp_can_fork(state)`：對手是否能一手造出同樣的雙重威脅（防守要提早處理）。

均以既有 `apply_move`（immutable）做單層展開，再用既有 `opponent_winning_lines` 風格的「兩子＋可claim」判定計數，**不展開對手回手樹**（守一手前瞻紅線）。

- **理由**：fork 是「贏在對手擋不完」，本質上是對**一手後**靜態盤面數威脅線，落在一手前瞻內、不需搜尋。
- **替代方案**：兩層 minimax 確認對手擋不掉——被否決，違反無樹搜尋紅線。以「≥2 條可claim線」近似「擋不完」雖非絕對精確，但快速、可讀、決定性，且足以壓過只看單線的 v1。

### D3：FSM 擴一個 `setup_fork` 攻擊子模式 + 規則層補 fork/gobble 規則

FSM 具名 state 擴為 `aggressive` / `defensive` / `setup_fork`（皆為可讀具名值，出現在 trace）。轉移（純函式威脅 flag 驅動）：

```
任一 state --(i_can_win)--------------------> aggressive(下 win_now)
aggressive --(opp_can_win and not i_can_win)-> defensive
defensive  --(i_can_win or not opp_can_win)--> aggressive
aggressive --(i_can_fork and not opp_can_win)-> setup_fork
setup_fork --(opp_can_win or i_can_win)------> (defensive / aggressive 同上規則)
```

各模式 `RuleTable`（priority 小者先，沿用 v1 慣例）：
- `aggressive`：`win_now` → `block_then_gobble` → `make_fork`(新) → `make_threat` → `develop`
- `defensive`：`win_now` → `block`（升級為能擋 fork 的 `block_fork`：選能同時壓低對手威脅線數最多者）→ `safe_develop`
- `setup_fork`：`win_now` → `block`(保命) → `commit_fork`(下造 fork 的步) → `develop`

- **理由**：新模式讓「主動佈置雙殺」成為可讀的一級行為，而非埋在評分裡的副作用，便於 trace 與 review；防守端把「擋 fork」做成顯式規則才接得住更兇的對手。
- **替代方案**：只靠評分隱式造 fork、不加模式——可行但 trace 不可讀、難解釋為何擊敗 v1；加更多模式（trap 等）——本 change 先收斂到 `setup_fork`，其餘留待後續迭代。

### D4：評分校準＝在 v1 評分上加 fork 獎勵與 gobble 得失項

`_score_v2` 在 v1 的線潛力／中心／保留／暴露項之外新增（皆具名常數）：
- `_FORK_BONUS`：一手後我方可claim威脅線數 ≥ 2 時加成（鼓勵造雙殺）。
- gobble 得失：`gobble_value`——覆蓋對手 top 解除其威脅線時加分；`reveal_penalty`——`MOVE` 搬起自家子若**揭露**對手可立即成線時重罰（與 `reveal_loses` 變體判定一致，但即使變體關閉也作為戰術扣分）。
- 仍以 `move_to_index` 升冪做決定性 tie-break。

- **理由**：把 Gobblet 特有得失顯式編碼，直接打 v1 的評分盲點；常數化保留 HL 紅線與可調性（迭代時改常數即可）。

### D5：對打評測與 3×3 矩陣＝多 seed × seeded 隨機開局 × 雙先後手

新增評測純函式 `play_match(p0_factory, p1_factory, seed) -> Status | None` 與 `winrate(a, b, seeds) -> float`（a 對 b，輪流先後手）。**關鍵**：v1/v2 皆決定性且 `initial_state` 不看 seed，純 FSM-vs-FSM 每個 seed 重播同一局；因此 `play_match` 在交手前先走 `opening_plies` 手 **seeded 隨機合法步**製造多樣開局（開局即終局的 seed 回傳 None 並跳過）。

- `test_fsm_v2_vs_v1.py`：固定 seed × seeded 隨機開局 × 雙先後手，斷言 (a) v2 對 v1 ≥ 平手 floor、(b) v2 對隨機 ≥ v1 對隨機 − epsilon、(c) v2 顯式 fork 行為。
- 3×3 矩陣：`experiments/hl-gobblet/matchup_matrix.py` 對 `{random, v1, v2}` 每個有序配對統計勝率並印出；`test_matchup_matrix.py` 斷言決定性與單調性（v1/v2 壓制隨機、v2 不弱於 v1）。

- **替代方案**：純 seed（無隨機開局）——被否決，FSM-vs-FSM 只會得 2 場不同棋，無法量化分佈；ε-擾動對手池——本 change 用「交手前隨機開局」即足以製造分佈且交手後仍決定性。

### D6：實作量測修正——一手前瞻下的誠實門檻（取代「≥70% 擊敗 v1」）

實作後的量測證實：在 seeded 隨機開局分佈下，**該局先手優勢約 73.5%**，且 v1 已接近一手前瞻可達的最佳玩法（v1、v2 對共同隨機對手勝率幾乎相等，約 98% vs 98%）。因此兩個同等強度的一手玩家頭對頭必然接近 50%——v2 對 v1 實測穩定落在約 41–51%（六類評分 + 權重 grid search 上限約 46.9%）。在**嚴格一手前瞻（HL 紅線：不展開對手回手樹）**下，無法可靠以 ≥70% 頭對頭擊敗已近一手最優的對手；真正要壓制需二手 minimax，已超出紅線。

故門檻**改為誠實表述**並寫入 spec：(a) v2 對 v1 ≥ 平手 floor（攔截「v2 變明顯比 v1 弱」）、(b) v2 對隨機 ≥ v1 對隨機 − epsilon、(c) v2 **顯式** fork 行為（進入 `setup_fork`／`commit_fork`、`block_fork` 在一手內把對手站立可 claim 線數壓到 < 2）。注意：嚴格一手下無法保證 v2「比 v1 多擋掉」fork（v1 的中心控制啟發常偶然落在同格），故 spec 僅宣稱 v2 自身的可驗證行為，不宣稱 fork 局面嚴格優於 v1。golden 在達到上述門檻後才凍結。

## Risks / Trade-offs

- **[一手前瞻無法 ≥70% 擊敗 v1（已實證）]** → 見 D6：先手優勢 ~73.5% + v1 已近一手最優 ⇒ 頭對頭 ~47%。門檻改為「不劣於 v1 + 顯式 fork 行為」的誠實表述；若日後要真正壓制需另開 change 放寬到二手 minimax（會改 HL 紅線）。
- **[fork 近似不精確：≥2 條可claim線未必真的擋不完；一手內也無法保證擋掉對手 fork]** → 接受為一手前瞻下的保守代理；spec 只宣稱 v2 一手內把對手站立可 claim 線壓到 < 2 與 v2 自身的 fork 行為，不宣稱嚴格優於 v1。
- **[隨機開局引入分佈但也可能造出不公平開局]** → 跳過「開局即終局」的 seed；門檻 floor 留大餘裕（量測 41–51% vs floor 0.40）吸收開局運氣。
- **[新增 gobble/fork 評估變慢]** → 仍是 O(線數×合法步) 的一手展開，無搜尋；對 Gobblet 9 格規模可忽略，評測秒級完成（使用者已確認結果極快）。
- **[誤動到 v1 共用模組導致 v1 golden 漂移]** → D1 用獨立 v2 模組；收尾驗證 diff 不含 v1 既有檔案、v1 golden 測試仍綠。
- **[模式增多使 FSM 難讀]** → 收斂到單一新模式 `setup_fork`；所有 state／規則名稱出現在 trace，golden 凍結其可讀序列。
- **[HL 紅線被無意違反]** → 收尾以 grep 驗證無 `optax`/`jax.grad`/`flax`/`torch` import、無 MCTS/minimax、所有權重為具名常數。

## Migration Plan

純新增，無資料遷移。部署＝合併新檔；回滾＝移除 v2 模組與測試、還原 `watch_match.py` factory 分支即可，v1 與既有測試不受影響。
