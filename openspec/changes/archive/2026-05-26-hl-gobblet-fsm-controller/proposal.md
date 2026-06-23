## Why

`hl-gobblet-env` 目前只有 `RandomOpponent` 可下棋，無法展示 HL（啟發式學習）典範「不靠訓練、純程式碼即策略」在零和棋類上也成立。`hl-procedural-policy` 的積木（RuleTable／FSM／Macro）至今只在 `hl-lunar-lander` 上被示範過；把同一套積木套到一個離散、回合制、有隱藏資訊（大子吃小子）的棋類，能驗證這些積木與環境無關、可跨環境重用，並為日後對打／自我對打提供一個比隨機強、且決策可解釋的對手。

## What Changes

- 新增一個**完全不訓練、不含梯度／神經網路**的程序化 Gobblet 對手 `FsmGobbletV1`，以 `hl_core` 的 RuleTable／FiniteStateMachine／MacroAction 積木組裝，實作既有 opponent 介面（`reset(seed)`、`act(state) -> Move`）。
- 該對手以 **FSM 切換兩種可讀具名模式**：
  - `aggressive`（攻擊）— 優先完成自己的三連、用較大子去吃以製造威脅。
  - `defensive`（防守）— 偵測對手即將連線的威脅時，優先擋掉或吃掉對手最上層的子。
  - 兩模式依**盤面威脅評估**（自己是否有一手致勝／對手是否有一手致勝）自動切換，全部為純函式分支，無任何學習。
- 新增 `src/hl_gobblet/controllers/` 套件（對齊 `hl_lander/controllers/` 慣例），放置該控制器與其純函式決策積木。
- 將 `FsmGobbletV1` 接進既有觀戰腳本 `watch_match.py` 的 opponent factory（`--p0 fsm` / `--p1 fsm`），讓它可與 `random` 對手對打觀看；觀戰腳本維持「只觀看、不寫報告」。
- 提供唯讀的 `decision_trace()`（對齊 `hl-procedural-policy`：每步含當前 state 名稱、觸發規則名稱、輸出 Move），並對固定 seed 凍結 golden trace 納入回歸測試。

## Capabilities

### New Capabilities
- `hl-gobblet-fsm-controller`: 在 `hl-gobblet-env` 上、用 `hl-procedural-policy` 積木組裝的免訓練 FSM 對手；定義「攻擊／防守」兩模式的切換條件、各模式的純函式決策規則、與隨機對手對打的觀戰整合，以及決策軌跡導出與 golden-trace 回歸。

### Modified Capabilities
<!-- 無：本變更只「使用」hl-gobblet-env 的環境介面與 hl-procedural-policy 的積木，兩者的需求皆不改變，故不列為修改。 -->

## Impact

- 新增程式碼：`heuristic-learning/src/hl_gobblet/controllers/`（新套件，含 `fsm_gobblet_v1.py` 與其決策輔助純函式）。
- 修改程式碼：`heuristic-learning/experiments/hl-gobblet/watch_match.py`（opponent factory 多支援 `fsm`）。
- 新增測試：`heuristic-learning/tests/hl_gobblet/`（兩模式切換、各模式規則、`decision_trace` 確定性、golden trace 回歸）。
- 依賴：不新增第三方依賴；僅依賴既有 `hl_core`、`hl_gobblet` 與 `numpy`。
- HL 紅線：MUST NOT import `optax`／`jax.grad`／`flax`／`torch` 等梯度或權重更新工具；策略必須全由可讀程式碼分支構成。
