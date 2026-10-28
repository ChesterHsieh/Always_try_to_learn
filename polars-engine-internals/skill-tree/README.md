# Polars 引擎內部技能樹

從欄式儲存、執行模型、optimizer 一路走到 Polars 2.0 的 streaming 引擎，目標是**追得到 py-1.0.0 → py-2.0.0 原始碼的每一行**。
50 節點（主線 48、支線 2）／11 章／153 題／145 小時（主線 136 小時，每週 5 小時約 27 週）。

- 線上開啟：<https://chesterhsieh.github.io/Always_try_to_learn/polars-engine-internals/skill-tree/>
- 搭配閱讀：[1.0 → 2.0 導讀報告](../report/index.html)

## 檔案

| 檔案 | 用途 |
|---|---|
| `tree_data.py` | **資料單一來源**：`META`（設定）、`N`（節點）、題庫輪轉 |
| `quiz_a0_a3.py` … `quiz_a9_a10.py` | 題庫，依章分檔。**正解一律寫在 `o[0]`**，由 `tree_data.py` 檔尾輪轉到 A/B/C/D |
| `build.py`、`index.template.html` | 從 `skill-tree` skill 複製的產生器、驗證器與 UI 樣板（與 `system-design/_tooling/` 一致） |
| `make.py` | 跑 `build.py` 後產生 `index.html` 獨立版，並清掉 `skill-tree.json` |
| `polars-engine.html` | 片段版，發布成 Artifact 用 |
| `index.html` | 獨立版，GitHub Pages 或瀏覽器直接開啟用 |

## 重新產生

```bash
python3 make.py
```

改動一律改 `tree_data.py` 或 `quiz_*.py`，**不要手改兩份 HTML**。驗證不過會 exit 1，自動擋下的反模式包含
重複 id、依賴循環、dod 不可判定、非四選一、答案位置偏斜、正解長度洩漏、深挖 prompt 重複。

## 十一章

| 章 | 主題 | 樞紐節點 ◆ |
|---|---|---|
| A0 | 序章 · 立足點 | Crate 地圖：一個查詢的生命週期 |
| A1 | 欄式資料 | Parquet 的檔案結構 |
| A2 | 執行模型（CMU 15-445/721） | Pipeline 與 Pipeline Breaker |
| A3 | 查詢計畫與 IR | Arena 與 Node(usize) |
| A4 | 規則式最佳化 | Predicate Pushdown 穿過 Join |
| A5 | in-memory 引擎（1.x 預設） | 峰值記憶體從哪裡來 |
| A6 | 舊 streaming 引擎（polars-pipe） | 白名單與 streaming 島 |
| A7 | 新引擎骨架（polars-stream） | Phase 迴圈 |
| A8 | 行順序語意 | simplify_ordering |
| A9 | Breaker 與 Out-of-Core | polars-ooc：計帳與 Spill |
| A10 | Cost-Based 與復盤 | Join Reordering |

支線兩條：A2-S1 玩具引擎三模型對打、A7-S1 用 asyncio 做 mini 管線。

## 摸底預填

建樹前的摸底結果：執行模型與 optimizer 的陳述都還沒把握，Rust 的「async 會編譯成 state machine」已經懂。
第一次開啟頁面時，按「進度」，貼上下面這段再按「載入」，A7-3 的第一個任務會預先勾好：

```json
{"done": {"A7-3": [0]}, "ans": {}}
```

## 連結與資料來源

- Polars 原始碼連結一律固定在 `py-1.0.0` 或 `py-2.0.0` tag，85 條都在本機用 `git cat-file` 核對過檔案存在、行號在範圍內。
- Arrow、Parquet、Rust async book 用它們在 GitHub 上的原稿，已確認檔案存在。
- CMU 15-445/721 課程首頁、Morsel-Driven 論文（db.in.tum.de）、X100 論文（cidrdb.org）這三類連結，
  建樹時的執行環境連不到這些網域，**沒有實際打過確認回 200**，若失效請改用課程或論文的最新網址。
- 導讀報告的 Pages 網址要等這個資料夾合併進 main、Pages 部署後才會生效。

## 進度保存

進度存在瀏覽器 `localStorage`（key：`polars-engine-skilltree-v1`）。頁面上的「進度」按鈕會秀出整包 JSON，
**定期複製一份存起來**；換瀏覽器或清快取後貼回去按「載入」即可還原。
試煉區的「重置作答」只清目前篩選範圍內的作答，任務勾選不受影響。
