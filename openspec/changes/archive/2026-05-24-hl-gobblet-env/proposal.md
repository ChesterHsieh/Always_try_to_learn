## Why

Heuristic Learning（HL）目前只有 LunarLander 一個連續控制環境作為「不訓練、用手寫規則打平 RL」命題的驗證場。要展示這個命題的普適性，需要一個**結構完全不同**的環境：離散、回合制、雙人對弈、含隱藏資訊（疊放遮蔽）。**奇迹连连（Gobblet Gobblers）** 正好符合——它是 3×3 井字棋的進階版，規則簡單但狀態空間因「大子可吃小子」而暴增，且帶有記憶／隱藏資訊成分，非常適合手寫規則策略發揮。

網路上雖已有 PettingZoo 的多智能體 RL 版本（[elliottower/gobblet-rl](https://github.com/elliottower/gobblet-rl)），但那是為梯度訓練設計的；本專案要的是一個**為 HL 範式（gradient-free、code-as-policy）量身打造**、且不依賴 PyTorch/TensorFlow 的純規則環境，這在現有生態中尚不存在。

## What Changes

- 新增 `hl_gobblet` 環境套件：3×3 棋盤、每方 6 子（小／中／大各 2）、大子可遮蔽（gobble）較小子的完整規則引擎。
- 實作**狀態表示**：每格是一個 size→owner 的堆疊（stack），最上層決定該格歸屬與連線判定；棋盤外手牌（off-board reserve）追蹤各 size 剩餘數。
- 實作**合法步生成**：(a) 從手牌放新子到空格或較小子之上；(b) 移動盤上自己的最上層子到別格（空格或較小子之上）。
- 實作**勝負判定**：預設採官方規則——一步完成後才檢查最上層連線；以及一條可設定的進階變體旗標 `reveal_loses`（touch-move：拿起／移動一子的瞬間，若揭露對方已成連線，則拿起者判負）。
- 提供 **gymnasium 風格的 step/reset 介面**（單一 agent 視角 + 對手 hook），並保留多種對手（random / greedy）的擴充點，但本 change 僅實作環境本體與一個 `random` 對手用於測試自洽性。
- 提供完整單元測試（規則、合法步、連線判定、reveal_loses 變體、determinism）。
- 提供 **CLI 對戰觀戰器**：用 `rich` 渲染漂亮的終端機介面，逐步播放兩個 AI（對手物件）對打的整局過程——棋盤格、疊放/被吃的子、手牌剩餘、每步動作說明、勝負結果，支援逐步/自動播放與決定性 seed。讓使用者能清楚看見「兩個 AI 對戰時發生什麼事」。

本 change **不含**：手寫規則 controller（baseline/fsm 版本）、greedy 對手前瞻、評估管線（勝率對照）、REPORT.md、pygame 圖形視窗——這些留待後續 change。（CLI 觀戰器**含**在本 change。）

## Capabilities

### New Capabilities
- `hl-gobblet-env`: 奇迹连连（Gobblet Gobblers）環境的核心——盤面狀態模型、合法步生成、回合推進、勝負判定（含官方規則與 `reveal_loses` 變體）、種子決定性、一個用於驗證自洽的隨機對手，以及一個用 `rich` 渲染的 CLI 對戰觀戰器。

### Modified Capabilities
（無——這是全新環境，不更動既有 spec 的需求。）

## Impact

- **新增程式碼**：`heuristic-learning/src/hl_gobblet/`（state、rules、env、opponents、render），對應測試 `heuristic-learning/tests/hl_gobblet/`；以及觀戰腳本 `heuristic-learning/experiments/hl-gobblet/watch_match.py`（比照 lander 的 `play_gui.py` 慣例）。
- **建置設定**：`heuristic-learning/pyproject.toml` 的 `[tool.hatch.build.targets.wheel].packages` 需加入 `src/hl_gobblet`。
- **依賴**：沿用既有 numpy + gymnasium，並把 `rich` 升為直接相依（它已是專案的傳遞相依，純 Python、無重量級相依）；維持 JAX-only／禁 TF・PyTorch 的專案約束。
- **共用原語**：盡量重用 `hl_core` 的 policy/trace 介面精神，但此 change 不強制接上 controller。
- **既有環境**：不更動 `hl_lander` 任何檔案。
