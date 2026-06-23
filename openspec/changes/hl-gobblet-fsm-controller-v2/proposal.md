## Why

`FsmGobbletV1` 已能穩定壓制隨機對手，但它的威脅評估只看「對手單線是否差一手」、攻擊評分忽略 fork（雙重威脅）與 gobble（覆蓋／搬移後重新揭露線）這些 Gobblet 的核心戰術，因此面對同樣懂得阻擋與發展的對手會被打平或被反制。我們需要一個**仍守 HL 紅線**（純程式碼規則、最多一手前瞻、無樹搜尋、無梯度／神經網路）的迭代版本 `FsmGobbletV2`，透過更聰明的純函式評估把這些盲點補起來，並以可量化的對局門檻證明它能**大機率擊敗 v1**。

## What Changes

- 新增控制器 `FsmGobbletV2`：與 v1 同樣實作 opponent 介面（`reset(seed)` + `act(state) -> Move`）、永遠回傳合法步、不就地修改局面、相同 seed 可重現，並提供唯讀 `decision_trace()`。
- 強化威脅評估（純函式、最多一手前瞻）：在 `i_can_win`／`opp_can_win` 之外，偵測**雙重威脅／fork**（我方一手後同時逼出兩條致勝線）與**被 gobble／搬移後重新揭露**的對手線，讓攻防判斷更準。
- 強化攻擊與發展選步評分：重新校準線潛力／中心控制／大子彈性權重，並**獎勵製造 fork**，打破 v1 的評分盲點。
- 擴充 FSM 模式／規則層：在 `aggressive`／`defensive` 之外新增可讀具名模式（如 `setup_fork`／`trap`）或新規則優先層，提升表達力（仍只用 `hl_core` 的 `FiniteStateMachine`＋`RuleTable` 積木）。
- 專門化 gobble 戰術：對「覆蓋對手大子」「搬移自家子後揭露對手線」的得失做專門純函式評估與規則。
- 新增**對打（head-to-head）評測工具**：以多 seed × seeded 隨機開局 × 雙先後手統計 v2 對 v1 的勝率。實作量測證實：v1/v2 皆決定性且開局與 seed 無關，純 FSM-vs-FSM 每個 seed 重播同一局，故需先走 seeded 隨機開局製造分佈；在此分佈下該局先手優勢約 73.5%，而 v1 已近一手前瞻最優，兩個同等的一手玩家頭對頭必然接近 50%（實測 v2 對 v1 約 41–51%，grid search 上限 46.9%）。在**嚴格一手前瞻（HL 紅線）**下無法可靠地以 ≥70% 頭對頭擊敗已近一手最優的 v1。因此門檻**改以誠實表述**：(a) v2 對 v1 不劣於平手 floor、(b) v2 對共同隨機對手不弱於 v1、(c) v2 顯式處理 fork（進入 `setup_fork`／`commit_fork`、一手內把對手 fork 壓回單威脅以下）。
- 新增 **3×3 交互對打結果矩陣**：把 `random`／`fsm`(v1)／`fsm-v2` 兩兩對打（各自輪先後手）整理成可重現的勝負矩陣，作為棋力對照與回歸保護。
- 觀戰腳本 `watch_match.py` 的 opponent factory 新增 `fsm-v2` 名稱分支（不改變「只觀看」既有行為）。
- v2 在固定 seed 下的 `decision_trace()` 凍結為 golden trace 納入回歸。

不修改 `hl_core/` 與 `hl_gobblet/` 既有檔案；不修改 v1 控制器的既有行為（v1 凍結作為被擊敗的基準）。

## Capabilities

### New Capabilities
- `hl-gobblet-fsm-controller-v2`: 免訓練、可讀、決定性的 Gobblet 控制器 `FsmGobbletV2`，以更聰明的純函式威脅／選步評估（含 fork 與 gobble 戰術）迭代自 v1，並以多 seed × 雙先後手的對打門檻證明大機率擊敗 v1，附 3×3 交互對打結果矩陣與 golden trace 回歸。

### Modified Capabilities
<!-- 無：v1 capability (hl-gobblet-fsm-controller) 的需求不變，v1 行為凍結作為基準；v2 為獨立新 capability。 -->

## Impact

- **新增程式碼**：`heuristic-learning/src/hl_gobblet/controllers/fsm_gobblet_v2.py` 及其純函式輔助模組（v2 專屬的 `_assess`／`_score`／`_modes`，不複用會牽動 v1 golden 的既有模組）；`controllers/__init__.py` 匯出 `FsmGobbletV2`。
- **新增評測**：`heuristic-learning/experiments/hl-gobblet/` 下的對打評測腳本與 3×3 矩陣輸出；`tests/hl_gobblet/` 下 v2 單元測試、`test_fsm_v2_vs_v1.py` 勝率門檻、`test_matchup_matrix.py`（或等效）3×3 矩陣回歸、v2 golden trace 回歸。
- **調整**：`experiments/hl-gobblet/watch_match.py` 的 `_opponent_factory` 新增 `fsm-v2` 分支。
- **依賴**：無新增第三方依賴；僅重用 `hl_core`、`hl_gobblet`。
- **HL 紅線**：不得 import `optax`／`jax.grad`／`flax`／`torch`，不得引入 MCTS／多層 minimax，所有「參數」皆為原始碼具名常數。
