## 1. 套件骨架與建置設定

- [x] 1.1 建立 `src/hl_gobblet/__init__.py` 與 `src/hl_gobblet/opponents/__init__.py`
- [x] 1.2 在 `pyproject.toml` 的 `[tool.hatch.build.targets.wheel].packages` 加入 `src/hl_gobblet`
- [x] 1.3 建立 `tests/hl_gobblet/` 與 `experiments/hl-gobblet/` 目錄；確認 `make test` 能蒐集到新測試

## 2. 狀態模型（state.py）

- [x] 2.1 定義 `Player`、`Size` 列舉與每格 size→格主的不可變堆疊型別
- [x] 2.2 定義 `GobbletState`（frozen dataclass）：棋盤九格、雙方各 size 手牌數、當前玩家
- [x] 2.3 實作 `initial_state(seed)` 與「最上層歸屬」查詢輔助函式
- [x] 2.4 撰寫 `test_state.py`：初始局面、不可變性、疊放後最上層歸屬（對應 spec「盤面狀態模型」三情境）

## 3. 動作與編碼（moves.py）

- [x] 3.1 定義 `Move` 值物件（`PLACE` / `MOVE` 兩型）
- [x] 3.2 建立全動作空間與穩定的 `move_to_index` / `index_to_move`
- [x] 3.3 撰寫 `test_moves.py`：全動作空間大小固定、編碼/解碼 round-trip（對應 spec「動作索引雙向可逆」）

## 4. 規則引擎（rules.py）

- [x] 4.1 實作 `legal_moves(state)`：PLACE 與 MOVE 兩類，決定性順序
- [x] 4.2 實作 `apply_move(state, move)`：純函式推進、揭露底層子、切換玩家、拒絕非法動作
- [x] 4.3 實作八條線最上層連線檢查與官方規則勝負判定（含 `max_moves` 平局）
- [x] 4.4 撰寫 `test_rules.py`：放子/吃子/不可疊同或更大子、移動揭露底層子、拒絕非法動作、連線獲勝、達上限平局（對應 spec「合法步生成」「回合推進」「勝負判定（官方規則）」）

## 5. reveal_loses 變體

- [x] 5.1 在 `apply_move` 加入 `reveal_loses` 旗標：MOVE 的「拿起瞬間」中間狀態連線檢查
- [x] 5.2 撰寫 `test_reveal_loses.py`：啟用時拿起即揭露對方連線判負、關閉時同局面僅依落子後判定（對應 spec「reveal_loses 進階變體」兩情境）

## 6. 隨機對手（opponents/random.py）

- [x] 6.1 實作 `RandomOpponent`：吃 seed、從 `legal_moves` 決定性取樣一步
- [x] 6.2 撰寫對手最小測試：相同 seed 在相同局面選相同動作、永不回傳非法動作

## 7. 環境（env.py）

- [x] 7.1 定義 `GobbletEnv`：`reset(seed)` / `step(action)`、P0 視角、注入式對手 hook、`reveal_loses` 與 `max_moves` 設定
- [x] 7.2 實作觀測編碼（足以重建最上層歸屬與雙方手牌）與稀疏 reward、`info`（合法動作遮罩 + 局面快照）
- [x] 7.3 撰寫 `test_env.py`：reset 後有合法動作、step 推進並讓對手回應、相同 seed 決定性重現、雙 RandomOpponent self-play 正常終止（對應 spec「環境介面與隨機對手」四情境）

## 8. 渲染（render.py）

- [x] 8.1 渲染純函式（render.py）：把 `GobbletState` + 上一步動作轉成 `rich` 可渲染物件（棋盤格、歸屬/size、被蓋住底層子提示、雙方手牌、`MOVE` 揭露說明）；無 I/O
- [x] 8.2 撰寫 `test_render.py`：對含疊放與含 `MOVE` 揭露的局面做文字快照斷言（對應 spec「渲染含疊放與被吃的局面」「顯示移動揭露了什麼」）

## 9. CLI 對戰觀戰腳本

- [x] 9.1 把 `rich` 升為直接相依加入 `pyproject.toml` 的 `[project].dependencies`
- [x] 9.2 建立 `experiments/hl-gobblet/watch_match.py`：注入兩個對手物件、用 `rich.live.Live` 逐步刷新跑完整局，支援 `--seed`、`--p0/--p1`、`--reveal-loses`、`--delay`/`--step`，終止時顯示勝負（比照 lander `play_gui.py` 慣例）
- [x] 9.3 手動驗證：兩個 `RandomOpponent` 對打一整局能正常渲染並終止；相同 seed 重現相同對局（對應 spec「觀戰腳本跑完一整局」「觀戰決定性重現」）

## 10. 收尾

- [x] 10.1 跑 `make test` 全綠，並確認 `hl_gobblet` 覆蓋率達 80%+
- [x] 10.2 跑 `ruff` 通過；確認未引入 TF/PyTorch（`make hl-lander-deps-check` 仍綠）
- [x] 10.3 在 `heuristic-learning/README.md` 補一段 `hl_gobblet` 環境簡介、目錄結構與觀戰腳本用法（指向後續 controller change）
