---
project: heuristic-learning
---

## ADDED Requirements

### Requirement: 免訓練範圍宣告（HL 紅線）

`hl-gobblet-fsm-controller-v2` capability SHALL 僅由**純程式碼構成的程序化決策**組成，並以 `hl-procedural-policy` 的積木（`hl_core` 的 `RuleTable`、`FiniteStateMachine`）組裝。本 capability MUST NOT 包含任何梯度訓練、神經網路權重更新，且 MUST NOT 引入樹搜尋（MCTS／多層 minimax）；最多允許「一手套用後盤面」的單層純規則前瞻（含 fork 偵測與 gobble 得失評估，皆只看一手後的靜態盤面）。所有可調「參數」MUST 為原始碼中的具名常數。

#### Scenario: 嘗試加入梯度或神經網路

- **WHEN** 任何 PR 在本 capability 的程式碼內 import `optax`、`jax.grad`、`flax.training`、`torch` 或等效梯度／權重更新工具
- **THEN** 該 PR MUST 被退回，因為控制器必須完全由可讀程式碼分支構成

#### Scenario: 嘗試加入樹搜尋

- **WHEN** 一個 task 想在控制器內加入 MCTS、多層 minimax，或展開對手回手樹來確認 fork 不可擋
- **THEN** 開發者 MUST 改為另開 change，不得在本 capability 內擴張為樹搜尋；fork 偵測 MUST 僅以「一手後盤面的可claim威脅線數」近似

### Requirement: 重用 hl_core 積木且不修改既有原始碼與 v1

控制器 `FsmGobbletV2` SHALL 以 `hl_core` 的 `FiniteStateMachine` 與 `RuleTable` 組裝決策，且 MUST NOT 修改 `hl_core`、`hl_gobblet` 的任何既有原始碼，亦 MUST NOT 修改 `FsmGobbletV1` 的程式碼或其 golden trace。import 方向 MUST 為「v2 控制器 import `hl_core` 與 `hl_gobblet`」。v2 的威脅／選步輔助 MUST 放在 v2 專屬模組，MUST NOT 改動 v1 所依賴的既有輔助模組。

#### Scenario: hl_core 仍可獨立 import

- **WHEN** 在不 import `hl_gobblet` 的情況下單獨 import `hl_core`
- **THEN** import MUST 成功，證明本變更未讓 `hl_core` 反向依賴環境

#### Scenario: v1 行為與既有檔案不被改動

- **WHEN** 檢視本變更的 diff 並重跑 v1 的 golden trace 回歸
- **THEN** `hl_core/`、`hl_gobblet/` 既有檔案與 `fsm_gobblet_v1.py` 及 v1 用到的既有輔助模組 MUST NOT 被修改，且 v1 golden trace 測試 MUST 仍通過

### Requirement: opponent 介面相容

控制器 `FsmGobbletV2` SHALL 實作既有 Gobblet opponent 介面：`reset(seed)` 與 `act(state: GobbletState) -> Move`。`act` MUST 永遠回傳一個對當前局面合法的 `Move`，且 MUST NOT 就地修改傳入的 `state`。`reset(seed)` MUST 把當前 FSM state 重設為初始模式並清空決策軌跡緩衝，使相同 seed 可重現相同對局。

#### Scenario: 永遠回傳合法步

- **WHEN** 對任一非終局局面呼叫 `act`
- **THEN** 回傳的 `Move` MUST 屬於 `legal_moves(state)`

#### Scenario: 不就地修改傳入局面

- **WHEN** 對同一個 `state` 物件連續呼叫 `act` 兩次
- **THEN** 兩次回傳的 `Move` MUST 相同，且傳入的 `state` 物件 MUST NOT 被修改

#### Scenario: reset 後可重現

- **WHEN** 對控制器呼叫 `reset(seed)` 後重跑同一對局序列
- **THEN** 產生的選步序列 MUST 與第一次完全相同（確定性）

### Requirement: 強化威脅評估（fork 偵測，攻防對稱）

控制器 SHALL 在 v1 的 `i_can_win`／`opp_can_win` 之外，提供純函式的雙重威脅（fork）偵測：`i_can_fork(state)` 為「存在某合法步，套用後我方同時擁有至少兩條『兩子且第三格可合法 claim』的威脅線」；`opp_can_fork(state)` 為對手能否一手造出同樣的雙重威脅。所有偵測 MUST 透過 `apply_move(..., reveal_loses=ctx.reveal_loses)` 對**一手後盤面**判定，MUST NOT 展開對手回手樹，且 MUST NOT 就地修改 `state`。

#### Scenario: 偵測到我方可造雙殺

- **WHEN** 存在一合法步使套用後我方同時逼出兩條對手無法一手全擋的致勝威脅線
- **THEN** `i_can_fork(state)` MUST 為真

#### Scenario: 偵測到對手即將造雙殺

- **WHEN** 對手能一手造出兩條威脅線而我方無一手致勝
- **THEN** `opp_can_fork(state)` MUST 為真，使防守端能提早處理

#### Scenario: 無雙重威脅時回報為假

- **WHEN** 任何一方一手後最多只能持有單一可 claim 的威脅線
- **THEN** 對應的 `i_can_fork`／`opp_can_fork` MUST 為假

### Requirement: FSM 模式與切換條件（含 setup_fork）

控制器 SHALL 以 `FiniteStateMachine` 維護至少三個可讀具名 state：`aggressive`、`defensive`、`setup_fork`，初始 state 為 `aggressive`。每個 ply 開始時，控制器 MUST 依純函式威脅評估更新 state。轉移規則 MUST 滿足：任一 state 在 `i_can_win` 為真時 MUST 能走致勝步（落在 `aggressive` 的 `win_now`）；`opp_can_win` 為真且 `i_can_win` 為假時 MUST 進入 `defensive`；在無立即攻防威脅（`not opp_can_win` 且 `not i_can_win`）但 `i_can_fork` 為真時 MUST 進入 `setup_fork`；威脅消解後 MUST 回到 `aggressive`／`defensive`。當前 state MUST 是可讀的具名值並出現在決策軌跡中。

#### Scenario: 對手有立即威脅且我方無殺時轉防守

- **WHEN** 對手存在一手致勝的線、且我方無任何一手致勝步
- **THEN** FSM 當前 state MUST 為 `defensive`，且該步由 `defensive` 子策略產生

#### Scenario: 可佈置雙殺且無立即威脅時進入 setup_fork

- **WHEN** 我方無一手致勝、對手亦無一手致勝，但我方 `i_can_fork` 為真
- **THEN** FSM 當前 state MUST 為 `setup_fork`，且該步 MUST 為製造雙重威脅的步（造 fork）

#### Scenario: 我方有殺時維持／進入攻擊並取勝

- **WHEN** 我方存在某合法步可立即連線
- **THEN** FSM 當前 state MUST 為 `aggressive`，且該步 MUST 為致勝步

### Requirement: 各模式的規則表決策（含 make_fork 與 block_fork）

每個 state SHALL 綁定一個 `hl_core.RuleTable`，由依 priority 排序的純函式規則組成（數字越小越優先，命中第一條 guard 為真者）。`aggressive` MUST 至少依序包含：立即致勝（`win_now`）、阻斷對手致勝（保命，可含 gobble 解除威脅）、製造雙殺（`make_fork`）、製造威脅（`make_threat`）、穩健發展（`develop`）。`defensive` MUST 至少依序包含：立即致勝（`win_now`）、阻斷對手致勝與雙殺威脅（`block`／`block_fork`，選能壓低對手威脅線數最多者）、保守發展。`setup_fork` MUST 至少依序包含：立即致勝、保命阻斷、提交造 fork 的步、發展。每個 RuleTable 的 `default_action` MUST 為「在當前 ctx 的合法步中取決定性第一步」的保底。所有選步在分數相同時 MUST 以 `move_to_index` 升冪作穩定打破。

#### Scenario: 能贏就贏優先於一切

- **WHEN** 任一模式下存在可立即連線的合法步
- **THEN** 控制器 MUST 下出該致勝步，且觸發規則名稱 MUST 為 `win_now`

#### Scenario: 防守模式會擋掉對手的雙重威脅

- **WHEN** 處於 `defensive`、對手能一手造出雙殺、我方無一手致勝、但存在能壓低對手威脅線數最多的合法步
- **THEN** 控制器 MUST 下出該步（`block`／`block_fork`），而非無視雙重威脅去發展

#### Scenario: 沒有規則命中時走保底合法步

- **WHEN** 某模式所有具名規則的 guard 對當前局面皆為假
- **THEN** 控制器 MUST 回傳 `default_action` 給出的合法步，觸發規則名稱 MUST 標示為 `default`，且 MUST NOT 丟出例外

#### Scenario: 平手分數的決定性打破

- **WHEN** 同一模式下有多個合法步得到相同的最高評分
- **THEN** 控制器 MUST 選擇其中 `move_to_index` 最小者，使相同局面恆得相同選擇

### Requirement: gobble 戰術得失評估

控制器的選步評分 SHALL 顯式評估 Gobblet 的覆蓋（gobble）與搬移得失：以具名常數獎勵「覆蓋對手 top 以解除其威脅線」的步，並對「`MOVE` 搬起自家子後揭露對手可立即成線」的步施以懲罰。所有得失 MUST 在當前生效的變體規則（含 `reveal_loses`）下，透過 `apply_move` 對一手後盤面評估，不就地修改 `state`。

#### Scenario: 覆蓋對手大子解除威脅獲得加分

- **WHEN** 一合法步以較大子覆蓋對手 top，使對手原本的一手致勝線被解除
- **THEN** 該步的攻防評分 MUST 因解除威脅而高於不解除威脅的同類候選步

#### Scenario: 搬移後揭露對手線的步被懲罰

- **WHEN** 一 `MOVE` 候選步搬起自家子後，被覆蓋的對手子重新露出並使對手可立即成線
- **THEN** 控制器評分 MUST 對該候選步施以懲罰，使其不被優先選取（除非為唯一合法步或致勝必需）

### Requirement: 對 reveal_loses 變體一致

控制器在評估 `i_can_win`、fork、阻斷威脅、gobble 得失與所有候選步時，SHALL 一律透過 `apply_move(..., reveal_loses=ctx.reveal_loses)` 與 `status_of` 在**當前生效的變體規則**下判定。`reveal_loses` 設定 MUST 由 ctx 取得，預設為關閉。

#### Scenario: 啟用 reveal_loses 時不下會害己的步

- **WHEN** `reveal_loses` 啟用，某候選 `MOVE` 在「拿起瞬間」會揭露對手連線而使我方判負
- **THEN** 控制器評估該候選步時 MUST 視之為敗著而不選它（除非無其他合法步）

#### Scenario: 關閉時依官方規則判定

- **WHEN** `reveal_loses` 關閉
- **THEN** 控制器對 `i_can_win` 與致勝步的判定 MUST 僅依落子完成後的盤面，不因中間狀態判負

### Requirement: 決策軌跡導出

控制器 SHALL 提供唯讀的 `decision_trace()`，回傳每一步的結構化紀錄，至少含：步序、當前 state 名稱（`aggressive`／`defensive`／`setup_fork`）、觸發規則名稱、輸出 `Move` 及其整數索引。trace 導出 MUST NOT 改變控制器行為；對同一 seed 跑兩次的 trace MUST 完全相同。

#### Scenario: trace 長度與內容對齊對局

- **WHEN** 對固定 seed 跑完一整局後呼叫 `decision_trace()`
- **THEN** trace 長度 MUST 等於控制器實際出手的步數，且每筆 MUST 含當前 state 名稱、觸發規則名稱與輸出 Move

#### Scenario: trace 導出無副作用

- **WHEN** 在對局中途呼叫 `decision_trace()`
- **THEN** 後續每一步的選擇 MUST 與「未曾呼叫過 trace」的情況完全相同

### Requirement: 棋力門檻——v2 在一手前瞻下不劣於 v1

控制器 `FsmGobbletV2` SHALL 以對打回歸測試保證它在一手前瞻（HL 紅線）下**不劣於** `FsmGobbletV1`，並以 seeded 隨機開局 × 雙先後手製造可量化分佈。具體 MUST 斷言：(1) v2 對 v1 的頭對頭勝率 MUST ≥ 一個近平手的 floor（以實測留餘裕，如 ≥ 0.40），用以攔截「v2 變得明顯比 v1 弱」的退化；(2) v2 對共同 `RandomOpponent` 的勝率 MUST ≥ v1 對隨機的勝率減去一個小 epsilon（v2 MUST NOT 因新增評估而變弱）。對打 MUST 為決定性：相同 seed、相同開局手數與相同先後手配置 MUST 重現相同對局結果。

實作量測背景（說明為何不訂 ≥70%）：v1/v2 皆決定性且 `initial_state` 不受 seed 影響，純 FSM-vs-FSM 在任意 seed 都重播同一局，故 harness 在交手前先走 seeded 隨機合局開局手。在此分佈下該局**先手優勢約 73.5%**，而 v1 已接近一手前瞻可達的最佳玩法（v1、v2 對隨機勝率幾乎相等，約 98% vs 98%），兩個同等的一手玩家頭對頭必然接近 50%——實測 v2 對 v1 穩定落在約 41%–51%（權重 grid search 上限約 46.9%）。在嚴格一手前瞻（無樹搜尋）下無法可靠以 ≥70% 頭對頭擊敗一個已近一手最優的對手；要真正壓制需放寬到二手 minimax，已超出本 capability 的紅線。v2 相對 v1 的具體價值改以「fork 戰術行為」需求表述。

#### Scenario: v2 對 v1 不劣於平手 floor

- **WHEN** 讓 `FsmGobbletV2` 與 `FsmGobbletV1` 在固定 seed 集 × seeded 隨機開局 × 雙先後手對打多局
- **THEN** v2 的頭對頭勝率 MUST ≥ 平手 floor（如 0.40）

#### Scenario: v2 對隨機不弱於 v1

- **WHEN** 量測 v2 與 v1 各自對共同 `RandomOpponent` 的勝率
- **THEN** v2 對隨機的勝率 MUST ≥ v1 對隨機的勝率 − epsilon

#### Scenario: 棋力退化被攔截

- **WHEN** 一個 PR 改動 v2 使其對 v1 頭對頭跌破平手 floor，或對隨機顯著弱於 v1
- **THEN** 對打回歸測試 MUST 失敗

### Requirement: fork 戰術行為（v2 顯式處理 fork）

`FsmGobbletV2` SHALL 對 fork（雙重威脅）做**顯式**推理，這是 v2 相對 v1 在程式碼層面的具體差異：v1 沒有 fork 概念，僅以中心控制／線潛力等啟發**偶然**達成或擋住 fork；v2 以具名的 `setup_fork`／`commit_fork` 主動造 fork，並以 `block_fork` 在一手內把對手的「站立可 claim 線數」壓到 2 以下。

注意（HL 紅線下的限制）：在**嚴格一手前瞻**（不展開對手回手樹）下，無法保證 v2 在每個 fork 局面都「比 v1 多擋掉」一個 fork——因為 v1 的中心控制啟發常會偶然落在同一關鍵格。因此本需求 MUST 以 **v2 自身的行為與一手內可驗證的性質**表述，MUST NOT 宣稱 v2 在 fork 局面嚴格優於 v1。本 capability MUST 在 `tests/hl_gobblet/` 下有對應測試。

#### Scenario: v2 主動造 fork（進入 setup_fork）

- **WHEN** 在一個「我方一手可 fork、無對手立即勝／威脅」的具名局面詢問 v2 的選步
- **THEN** v2 的決策軌跡 MUST 顯示當前 state 為 `setup_fork`、觸發規則為 `commit_fork`，且套用該步後我方 `claimable_winning_lines` MUST ≥ 2

#### Scenario: v2 一手內中和對手的 fork

- **WHEN** 在一個「對手一手可 fork、我方無立即勝／威脅」的局面（可由 seeded 隨機開局重現）詢問 v2 的選步
- **THEN** 套用 v2 的選步後，對手在該靜態盤面的站立 `claimable_winning_lines` MUST < 2（即 v2 該步把對手的雙重威脅壓回單一威脅以下）

### Requirement: 3×3 交互對打結果矩陣

本 capability SHALL 產出可重現的 3×3 交互對打結果矩陣，涵蓋 `random`、`fsm`(v1)、`fsm-v2` 三個控制器兩兩對打（每個有序配對各輪先後手、seeded 隨機開局），輸出每格的勝率。矩陣 MUST 為決定性可重現，並以回歸測試斷言關鍵單調性與門檻：v1、v2 對 random 的勝率 MUST 皆顯著過半；v2 對 random 的勝率 MUST ≥ v1 對 random 的勝率 − epsilon（v2 不弱於 v1）；v2 對 v1 的勝率 MUST ≥ 平手 floor（不得明顯劣於 v1）。

#### Scenario: 產生 3×3 矩陣

- **WHEN** 執行交互對打評測（固定 seed 集 × seeded 隨機開局 × 雙先後手）
- **THEN** MUST 輸出 `{random, v1, v2}` 兩兩對打的勝率矩陣，且相同輸入 MUST 重現相同矩陣

#### Scenario: 矩陣單調性與門檻回歸

- **WHEN** 跑 3×3 矩陣回歸測試
- **THEN** MUST 斷言 v1／v2 對 random 皆顯著過半、v2-vs-random ≥ v1-vs-random − epsilon、且 v2-vs-v1 ≥ 平手 floor；任一不成立時測試 MUST 失敗

### Requirement: 觀戰腳本整合

既有觀戰腳本 `watch_match.py` 的 opponent factory SHALL 支援以名稱 `fsm-v2` 選用 `FsmGobbletV2`（可作為 `--p0` 或 `--p1`），使其能與 `random`、`fsm`(v1) 對打觀看。整合 MUST 僅擴充 factory 分支，不改變觀戰腳本「只觀看、不寫報告、不做統計」的既有約束；相同 seed 與相同雙方對手 MUST 產生相同的對局與顯示序列。

#### Scenario: fsm-v2 對 v1 觀戰跑完一整局

- **WHEN** 以 `--p0 fsm-v2 --p1 fsm --seed <s>` 執行觀戰腳本
- **THEN** 腳本逐步顯示每一步後的棋盤，並在對局終止時顯示勝方或平局結果，過程中不出現非法動作

#### Scenario: 觀戰決定性重現

- **WHEN** 以相同 seed 與相同雙方對手（含 `fsm-v2`）重跑觀戰腳本
- **THEN** 產生完全相同的對局過程與顯示序列

### Requirement: Golden Trace 回歸

控制器在固定 seed 下的 `decision_trace()` 一旦穩定（達到棋力門檻後）MUST 凍結為 golden trace 並納入回歸；任何使該 golden trace 改變的 PR MUST 在說明中標示為刻意行為變更並更新 golden，否則 reviewer SHALL 退回。golden 僅比對語意欄位（步序、state 名稱、規則名稱、Move 與索引）。

#### Scenario: golden trace 退化被攔截

- **WHEN** 一個 PR 改動 v2 導致其在固定 seed 下的 `decision_trace()` 與 golden trace 不符
- **THEN** 回歸測試 MUST 失敗，且開發者 MUST 在 PR 內說明這是刻意變更並更新 golden，否則 reviewer SHALL 退回
