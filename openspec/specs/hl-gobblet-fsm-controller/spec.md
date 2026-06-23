---
project: heuristic-learning
---

# hl-gobblet-fsm-controller Specification

## Purpose

在 heuristic-learning 專案下，以 `hl-procedural-policy` 的純程式碼積木（有限狀態機 `FiniteStateMachine` 與規則表 `RuleTable`）組裝一個免訓練、可讀、決定性的奇迹连连（Gobblet Gobblers）控制器 `FsmGobbletV1`。控制器重用既有 `hl_core` 與 `hl_gobblet`（不修改其原始碼），以 `aggressive`／`defensive` 兩模式依純函式威脅評估切換，並在每個模式下以依優先序排列的純函式規則選步；一律在當前生效的變體規則（含 `reveal_loses`）下評估，保證永遠回傳合法步、不就地修改局面、相同 seed 可重現。控制器另提供唯讀的決策軌跡導出，並以單元測試、勝過隨機對手的棋力門檻與 golden trace 回歸保護其行為，且整合進既有的觀戰腳本以供觀看。

## Requirements

### Requirement: 免訓練範圍宣告（HL 紅線）

`hl-gobblet-fsm-controller` capability SHALL 僅由**純程式碼構成的程序化決策**組成，並以 `hl-procedural-policy` 的積木（規則表、有限狀態機）組裝。本 capability MUST NOT 包含任何梯度訓練、神經網路權重更新，且 MUST NOT 引入樹搜尋（MCTS／深層 minimax）；最多允許「一手致勝／一手致敗」的單層純規則前瞻。所有可調「參數」MUST 為原始碼中的具名常數。

#### Scenario: 嘗試加入梯度或神經網路

- **WHEN** 任何 PR 在本 capability 的程式碼內 import `optax`、`jax.grad`、`flax.training`、`torch` 或等效梯度／權重更新工具
- **THEN** 該 PR MUST 被退回，因為控制器必須完全由可讀程式碼分支構成

#### Scenario: 嘗試加入樹搜尋

- **WHEN** 一個 task 想在控制器內加入 MCTS 或多層 minimax 搜尋
- **THEN** 開發者 MUST 改為另開 change，不得在本 capability 內擴張為樹搜尋

### Requirement: 重用 hl_core 積木且不修改既有原始碼

控制器 SHALL 以 `hl_core` 的 `FiniteStateMachine` 與 `RuleTable` 組裝決策，且 MUST NOT 修改 `hl_core` 或 `hl_gobblet` 的任何既有原始碼。import 方向 MUST 為「控制器 import `hl_core` 與 `hl_gobblet`」，`hl_core` MUST NOT 反向依賴任何具體環境。控制器把 `GobbletState` 當作傳給積木的 observation、把當前局面的合法步與威脅評估放入 ctx。

#### Scenario: hl_core 仍可獨立 import

- **WHEN** 在不 import `hl_gobblet` 的情況下單獨 import `hl_core`
- **THEN** import MUST 成功，證明本變更未讓 `hl_core` 反向依賴環境

#### Scenario: 未改動 hl_core 與 hl_gobblet 既有檔案

- **WHEN** 檢視本變更的 diff
- **THEN** `hl_core/` 與 `hl_gobblet/` 既有檔案 MUST NOT 被修改（僅新增 `hl_gobblet/controllers/` 與測試，並調整觀戰腳本的 opponent factory）

### Requirement: opponent 介面相容

控制器 `FsmGobbletV1` SHALL 實作既有 Gobblet opponent 介面：`reset(seed)` 與 `act(state: GobbletState) -> Move`。`act` MUST 永遠回傳一個對當前局面合法的 `Move`，且 MUST NOT 就地修改傳入的 `state`。`reset(seed)` MUST 把當前 FSM state 重設為初始模式並清空決策軌跡緩衝，使相同 seed 可重現相同對局。

#### Scenario: 永遠回傳合法步

- **WHEN** 對任一非終局局面呼叫 `act`
- **THEN** 回傳的 `Move` MUST 屬於 `legal_moves(state)`

#### Scenario: 不就地修改傳入局面

- **WHEN** 對同一個 `state` 物件連續呼叫 `act` 兩次
- **THEN** 兩次回傳的 `Move` MUST 相同，且傳入的 `state` 物件 MUST NOT 被修改

#### Scenario: reset 後可重現

- **WHEN** 對控制器呼叫 `reset(seed)` 後重跑同一對局序列
- **THEN** 產生的選步序列 MUST 與第一次完全相同（確定性）

### Requirement: FSM 兩模式與切換條件

控制器 SHALL 以 `FiniteStateMachine` 維護兩個可讀具名 state：`aggressive`（攻擊）與 `defensive`（防守），初始 state 為 `aggressive`。每個 ply 開始時，控制器 MUST 依純函式威脅評估更新 state：定義 `i_can_win`（存在某合法步，套用後 `status_of` 判我方勝）與 `opp_can_win`（盤面存在某線，對手只差一手即可在合法情況下連成）。轉移規則 MUST 為：`opp_can_win` 為真且 `i_can_win` 為假時進入 `defensive`；`i_can_win` 為真或 `opp_can_win` 為假時進入 `aggressive`。當前 state MUST 是可讀的具名值，並出現在決策軌跡中。

#### Scenario: 對手有立即威脅且我方無殺時轉防守

- **WHEN** 對手存在一手致勝的線、且我方無任何一手致勝步
- **THEN** FSM 當前 state MUST 變為 `defensive`，且該步動作由 `defensive` 子策略產生

#### Scenario: 我方有殺時維持／進入攻擊

- **WHEN** 我方存在某合法步可立即連線（不論對手是否也有威脅）
- **THEN** FSM 當前 state MUST 為 `aggressive`，且該步 MUST 下出致勝步

#### Scenario: 無立即威脅時維持攻擊

- **WHEN** 對手沒有任何一手致勝的線
- **THEN** FSM 當前 state MUST 為 `aggressive`，由 `aggressive` 子策略產生發展步

### Requirement: 各模式的規則表決策

每個 state SHALL 綁定一個 `hl_core.RuleTable`，由依 priority 排序的純函式規則組成（數字越小越優先，命中第一條 guard 為真的規則）。`aggressive` MUST 至少依序包含：立即致勝（`win_now`）、阻斷對手致勝（保命）、製造威脅、穩健發展。`defensive` MUST 至少依序包含：立即致勝（`win_now`）、阻斷對手致勝威脅、保守發展。每個 RuleTable 的 `default_action` MUST 為「在當前 ctx 的合法步中取決定性第一步」的保底，使任何局面皆有合法輸出而不丟例外。所有選步在分數相同時 MUST 以 `move_to_index` 升冪作穩定打破，確保決定性。

#### Scenario: 能贏就贏優先於一切

- **WHEN** 任一模式下存在可立即連線的合法步
- **THEN** 控制器 MUST 下出該致勝步，且觸發規則名稱 MUST 為 `win_now`

#### Scenario: 攻擊模式仍會擋致命威脅

- **WHEN** 處於 `aggressive`、我方無一手致勝、但對手有一手致勝且我方可合法阻斷
- **THEN** 控制器 MUST 下出阻斷步，而非無視威脅去發展

#### Scenario: 沒有規則命中時走保底合法步

- **WHEN** 某模式所有具名規則的 guard 對當前局面皆為假
- **THEN** 控制器 MUST 回傳 `default_action` 給出的合法步，觸發規則名稱 MUST 標示為 `default`，且 MUST NOT 丟出例外

#### Scenario: 平手分數的決定性打破

- **WHEN** 同一模式下有多個合法步得到相同的最高評分
- **THEN** 控制器 MUST 選擇其中 `move_to_index` 最小者，使相同局面恆得相同選擇

### Requirement: 對 reveal_loses 變體一致

控制器在評估 `i_can_win`、阻斷威脅與所有候選步時，SHALL 一律透過 `apply_move(..., reveal_loses=ctx.reveal_loses)` 與 `status_of` 在**當前生效的變體規則**下判定，使切換層與規則層都遵循同一規則。`reveal_loses` 設定 MUST 由 ctx 取得，預設為關閉。

#### Scenario: 啟用 reveal_loses 時不下會害己的步

- **WHEN** `reveal_loses` 啟用，某候選 `MOVE` 在「拿起瞬間」會揭露對手連線而使我方判負
- **THEN** 控制器評估該候選步時 MUST 視之為敗著而不選它（除非無其他合法步）

#### Scenario: 關閉時依官方規則判定

- **WHEN** `reveal_loses` 關閉
- **THEN** 控制器對 `i_can_win` 與致勝步的判定 MUST 僅依落子完成後的盤面，不因中間狀態判負

### Requirement: 決策軌跡導出

控制器 SHALL 提供唯讀的 `decision_trace()`，回傳每一步的結構化紀錄，至少含：步序、當前 state 名稱（`aggressive`／`defensive`）、觸發規則名稱、輸出 `Move` 及其整數索引。trace 導出 MUST NOT 改變控制器行為；對同一 seed 跑兩次的 trace MUST 完全相同。

#### Scenario: trace 長度與內容對齊對局

- **WHEN** 對固定 seed 跑完一整局後呼叫 `decision_trace()`
- **THEN** trace 長度 MUST 等於控制器實際出手的步數，且每筆 MUST 含當前 state 名稱、觸發規則名稱與輸出 Move

#### Scenario: trace 導出無副作用

- **WHEN** 在對局中途呼叫 `decision_trace()`
- **THEN** 後續每一步的選擇 MUST 與「未曾呼叫過 trace」的情況完全相同

### Requirement: 觀戰腳本整合

既有觀戰腳本 `watch_match.py` 的 opponent factory SHALL 支援以名稱 `fsm` 選用 `FsmGobbletV1`（可作為 `--p0` 或 `--p1`），使其能與 `random` 對手對打觀看。整合 MUST 僅擴充 factory 分支，不改變觀戰腳本「只觀看、不寫報告、不做統計」的既有約束；相同 seed 與相同雙方對手 MUST 產生相同的對局與顯示序列。

#### Scenario: fsm 對 random 觀戰跑完一整局

- **WHEN** 以 `--p0 fsm --p1 random --seed <s>` 執行觀戰腳本
- **THEN** 腳本逐步顯示每一步後的棋盤，並在對局終止時顯示勝方或平局結果，過程中不出現非法動作

#### Scenario: 觀戰決定性重現

- **WHEN** 以相同 seed 與相同雙方對手（含 `fsm`）重跑觀戰腳本
- **THEN** 產生完全相同的對局過程與顯示序列

### Requirement: 棋力門檻與 Golden Trace 回歸

控制器 MUST 在 `tests/hl_gobblet/` 下有單元測試，至少涵蓋：兩模式切換條件、各模式規則優先序與 default 保底、`act` 永不回傳非法步、`decision_trace()` 確定性與無副作用。對打測試中，`FsmGobbletV1` 對 `RandomOpponent` 在多個固定 seed 的對局勝率 MUST 顯著過半（明顯優於隨機）。控制器在固定 seed 下的 `decision_trace()` 一旦穩定 MUST 凍結為 golden trace 並納入回歸；任何使該 golden trace 改變的 PR MUST 在說明中標示為刻意行為變更並更新 golden，否則 reviewer SHALL 退回。

#### Scenario: 跑控制器單元測試

- **WHEN** 開發者執行 `heuristic-learning/.venv/bin/python -m pytest tests/hl_gobblet/`
- **THEN** 測試 MUST 涵蓋兩模式切換、規則優先序與 default、合法性保證、trace 確定性，且全部 MUST 通過

#### Scenario: 勝過隨機對手

- **WHEN** 讓 `FsmGobbletV1` 與 `RandomOpponent` 在一組固定 seed 對打多局
- **THEN** `FsmGobbletV1` 的勝率 MUST 顯著過半

#### Scenario: golden trace 退化被攔截

- **WHEN** 一個 PR 改動控制器導致其在固定 seed 下的 `decision_trace()` 與 golden trace 不符
- **THEN** 回歸測試 MUST 失敗，且開發者 MUST 在 PR 內說明這是刻意變更並更新 golden，否則 reviewer SHALL 退回
