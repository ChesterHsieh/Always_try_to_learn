## Context

奇迹连连（Gobblet Gobblers）規則（已查證，來源見 proposal）：

- 3×3 棋盤。每方 6 子，分大／中／小三種 size，各 2 個。子可疊放（gobble）：較大的可蓋住任何較小的（含對手與自己的）。
- 一回合二選一：(a) 從手牌放一個新子到「空格」或「較小子之上」；(b) 移動盤上自己**最上層**的一子到「空格」或「較小子之上」。
- 勝利：任一玩家的子（盤面最上層）在橫／直／斜連成三即獲勝。
- 隱藏資訊：大子蓋住小子後，小子被「記住」但不可見；移動大子會重新揭露底下的子。

本專案 `hl_lander` 已建立的慣例：`env.py` 是唯一命名環境的地方；controller 走 `reset(seed)/act(obs)` 介面；版本化 artefact 不可變。本 change 沿用「狀態不可變、純函式推進」的風格（呼應全域 coding-style：immutability）。

約束：JAX-only 生態、禁 TF/PyTorch；不引入新外部依賴（只用 numpy + gymnasium）；檔案小而聚焦。

## Goals / Non-Goals

**Goals:**
- 一個正確、可測試、決定性的 Gobblet Gobblers 規則引擎與環境。
- 狀態以不可變資料結構表示；推進為純函式（給定 state + action → new state），易於規則策略與測試。
- 合法步生成完整且唯一編碼（action 可被 index 化，方便日後 controller 與遮罩）。
- 勝負判定預設官方規則，並提供 `reveal_loses` 變體旗標。
- 一個 `random` 對手，讓 env 能自我對打跑完整局以驗證自洽。
- 一個用 `rich` 的 CLI 對戰觀戰器，把任意兩個對手物件的整局對打逐步渲染成漂亮的終端機畫面，讓人「看得懂兩個 AI 對戰時發生什麼事」。

**Non-Goals:**
- 不做手寫規則 controller（baseline/fsm）、不做 greedy 前瞻、不接 `hl_core` policy 管線。
- 不做評估管線（勝率對照）、REPORT.md、pygame 圖形視窗渲染（CLI 觀戰器有做）。
- 不做 RL 訓練或 PettingZoo 相容層。

## Decisions

### D1：狀態表示——每格一個 size→格主的不可變堆疊

每格用 size 為索引的固定長度三元組記錄各 size 上是誰的子（`None`/`P0`/`P1`），最上層 = 已被佔用的最大 size。整盤 9 格 + 各方手牌（每方各 size 剩餘數）構成 `GobbletState`，以 `@dataclass(frozen=True)` + tuple 表示，確保不可變。

- 為何不用「只存最上層」：移動大子需揭露底下的子，必須保留完整堆疊。記隱藏資訊正是這遊戲的核心 tricky 點。
- 為何用「size→格主」而非「stack list」：size 嚴格遞增的疊放規則讓 size-indexed 表示天然合法（不可能在大子上疊小子），判定與序列化都簡單；且固定長度利於日後向量化。

### D2：合法步生成與 action 編碼

兩類動作統一成一個 `Move` 值物件：
- `PLACE(size, to_cell)`：手牌還有該 size，且 `to_cell` 最上層為空或被更小的子佔用。
- `MOVE(from_cell, to_cell)`：`from_cell` 最上層是當前玩家的子，`to_cell` 合法承接（空或更小），且 `from != to`。

提供 `legal_moves(state) -> tuple[Move, ...]` 與穩定的 `move_to_index` / `index_to_move`（全動作空間固定編碼，方便日後動作遮罩）。生成順序固定 → 決定性。

- 替代方案：直接用 gym `Discrete` 扁平 id。仍會做扁平編碼，但以 `Move` 值物件為一等公民，可讀性與測試性更好。

### D3：推進為純函式 + 勝負判定時機

`apply_move(state, move) -> GobbletState` 回新狀態（不可變）。勝負判定：

- **官方規則（預設）**：在 `MOVE` 中，先從來源格「拿起」（揭露底下子），再「落下」到目標格，**只在落子完成後**檢查所有八條線的最上層連線。
- **`reveal_loses` 變體**：模擬「拿起的瞬間」——在來源子被拿起、尚未落下的中間狀態先檢查連線；若此時對方已成三連，則當前（拿起的）玩家立即判負。此旗標預設關閉。

`apply_move` 回傳的勝負資訊放在 step 結果裡（贏家 / 平局 / 進行中）。平局處理：Gobblet 理論上不必然終局，設可設定的 `max_moves` 上限，達上限判平局，避免無限對局。

- 為何把 `reveal_loses` 做成旗標：使用者描述的是進階 touch-move 變體，官方基本規則沒有；兩種都做才能做實驗對照（呼應 HL 的對照精神）。

### D4：環境介面——gymnasium 風格 + 對手 hook

`GobbletEnv` 提供 `reset(seed) -> obs` 與 `step(action) -> (obs, reward, terminated, truncated, info)`，從**單一受控玩家（P0）視角**出發；對手（P1）由注入的 `opponent` 物件在 P0 落子後自動走一步。reward 用稀疏訊號（贏 +1／輸 -1／平 0／非終局 0），`info` 帶 `legal_moves` 遮罩與當前 `GobbletState` 快照（供 trace／除錯）。

- 為何單視角 + 對手 hook 而非完整多智能體 API：本專案的 HL 命題是「單一受控策略」，與 LunarLander 一致；多智能體 API 是 non-goal。對手 hook 留好擴充點即可。

### D6：CLI 對戰觀戰器（rich）

用 `rich` 把一整局 AI vs AI 對打渲染成終端機畫面。職責分層：

- `render.py`（純函式、無 I/O）：把一個 `GobbletState` 轉成 `rich` 可渲染物件——3×3 棋盤格、每格最上層的子（以顏色區分 P0／P1、以符號大小區分 size）、被蓋住的底層子以提示標記、雙方手牌剩餘、以及「上一步動作」的人類可讀說明（含 `MOVE` 揭露了什麼）。純函式 → 可單元測試、不依賴終端機。
- `experiments/hl-gobblet/watch_match.py`（腳本、唯一做 I/O 的地方）：用 `GobbletEnv` 注入兩個對手物件，用 `rich.live.Live` 逐步刷新畫面跑完整局；支援 `--seed`、`--p0 / --p1`（對手名稱）、`--reveal-loses`、`--delay`（自動播放間隔）或 `--step`（按鍵逐步）。比照 lander 的 `play_gui.py`：只用於「看」，不寫 REPORT、不做統計。

為何把渲染抽成純函式而非寫死在腳本：渲染邏輯（誰被誰吃、揭露了什麼）正是這遊戲最容易看錯的地方，抽成純函式才能用 `test_render.py` 對「文字快照」做斷言；腳本只負責迴圈與刷新節奏。

替代方案：用 `textual` 做全螢幕互動 app——功能更強但相依更重、複雜度高，對「看兩個 AI 對打」這個唯讀需求過度設計，故不採用。

### D5：套件與檔案佈局（many small files）

```
src/hl_gobblet/
├── __init__.py
├── state.py        # GobbletState、Player/Size 列舉、堆疊與棋盤不可變模型
├── rules.py        # legal_moves、apply_move、勝負判定、line 檢查、reveal_loses
├── moves.py        # Move 值物件、move<->index 編碼、全動作空間
├── env.py          # GobbletEnv（gymnasium 風格）、obs 編碼、對手 hook、ENV 常數
├── render.py       # 純函式：GobbletState + 上一步 → rich 可渲染物件（無 I/O）
└── opponents/
    ├── __init__.py
    └── random.py   # RandomOpponent（決定性、吃 seed）
experiments/hl-gobblet/
└── watch_match.py  # CLI 觀戰器：兩個對手對打、rich.live 逐步刷新（唯一做 I/O 處）
tests/hl_gobblet/
├── test_state.py
├── test_rules.py        # 連線判定、gobble 合法性、apply_move 不可變性
├── test_moves.py        # move<->index round-trip、全動作空間
├── test_reveal_loses.py # touch-move 變體
├── test_render.py       # 渲染純函式的文字快照（棋盤、被吃、揭露說明）
└── test_env.py          # reset/step、self-play 跑完整局、determinism
```

## Risks / Trade-offs

- [隱藏資訊的觀測表示可能洩漏／不足] → obs 預設以 P0 視角提供「完整堆疊」（god view，利於規則策略），另記一個 flag 註明這是完整資訊；若日後要做記憶挑戰再加 partial-observation 變體。先求環境正確，不過早最佳化觀測。
- [`reveal_loses` 中間狀態判定容易出錯] → 用獨立 `test_reveal_loses.py` 針對「拿起即揭露對方連線」與「拿起後落子才連線」兩種情境做 golden 測試；明確定義「拿起的瞬間」= 來源格最上層被移除、目標格尚未變更的狀態。
- [對局可能不終局] → `max_moves` 上限 + 平局判定，並在測試中驗證上限會觸發 truncated。
- [action 空間隨疊放變化，遮罩維護成本] → 採固定全動作空間編碼 + `legal_moves` 遮罩；index 編碼穩定不隨局面變動，遮罩只是布林向量。
- [與 elliottower/gobblet-rl 規則細節差異] → 我們以查證到的官方規則為準並在 spec 寫死；不追求與該 repo 行為等價（它是 RL 導向、含我們不要的相依）。
- [CLI 渲染最容易看錯「誰被誰吃／揭露了什麼」] → 渲染抽成 `render.py` 純函式，用 `test_render.py` 對文字快照斷言（含一個有疊放、有 `MOVE` 揭露的局面）；觀戰腳本只負責迴圈與刷新，不含判斷邏輯。
- [rich 升為直接相依是否違反 JAX-only 約束] → `rich` 已是專案的傳遞相依（經 tensorstore），純 Python、無重量級或框架相依，不違反「禁 TF/PyTorch」的精神；於 proposal Impact 已載明。

## Open Questions

- obs 的精確張量佈局（god-view 維度）留到實作時定案，spec 只約束「足以重建最上層歸屬與手牌剩餘」。
- 平局上限 `max_moves` 的預設值（暫定一個保守值，如 60）於實作時依測試調整。
