# System Design 面試題庫

一題一個資料夾，每題產出一份**單檔自足的互動技能樹 HTML**（Path of Exile 式節點圖 ＋ 選擇題檢核點），可直接發布成 Artifact 或用瀏覽器開啟。

## 目錄

| 題目 | 資料夾 | 節點 / 題目 | 狀態 |
|---|---|---|---|
| URL Shortener（縮網址） | [url-shortener](url-shortener/) | 40 節點 / 120 題 | ✅ [線上開啟](https://chesterhsieh.github.io/Always_try_to_learn/system-design/url-shortener/) |

## 建置

```bash
python3 build.py url-shortener
```

產生器與 UI 樣板集中在 `_tooling/`，建置時暫時複製進題目資料夾，產生 `<slug>.html` 後移除。
驗證不通過會 `exit 1` —— **不要繞過去手改產物 HTML**，改 `tree_data.py` 後重跑。

自動擋下的反模式：重複 id、不存在的前置、依賴循環、空任務清單、不可判定的驗收條件、
非四選一、缺解釋或深挖 prompt、偷懶選項、**答案位置偏斜**、**正解長度洩漏**、深挖 prompt 重複。

## 新增一題

1. 建資料夾 `system-design/<題目 slug>/`
2. 寫 `tree_data.py`，三個頂層物件：
   - `META` — 標題、slug、localStorage key、每週時數、等級稱號、章節與配色
   - `N` — 節點陣列（`id`／`act`／`type`／`x`／`y`／`hours`／`deps`／`desc`／`why`／`tasks`／`dod`／`res`）
   - `QR` — 題庫，**正解一律寫在第 0 位**，檔尾用固定樣式輪轉到 A/B/C/D 並產生 `Q`
3. `python3 build.py <題目 slug>`
4. 把產出的 HTML 發布成 Artifact，並在上面的目錄表補一行

節點規格與出題規格見 `~/.claude/skills/skill-tree/references/`。

## 樣板來源

`_tooling/` 是從 `skill-tree` skill 的 `assets/` 複製過來的，兩邊目前一致。
skill 本身版控在 [ChesterHsieh/chester-skills](https://github.com/ChesterHsieh/chester-skills)
的 `plugins/skill-tree/`，日後 skill 更新時重新複製 `build.py` 與 `index.template.html` 即可。

試煉區的「重置作答」已經是 skill 的標準功能：清除範圍跟著目前的篩選走
（選了某章就只清該章、選了某節點就只清該節點），只動作答不碰任務勾選，需按兩次確認。

## 慣例

- **全篇繁體中文**（節點標題、任務、題幹、選項、解釋、深挖 prompt）
- 每章至少一個 `keystone`（◆ 樞紐）——通過它之後後續做法會改變，不是「比較難」
- `deps` 必須通過「不懂就做不出」測試，而不是「最好先懂」
- `dod` 要可判定：寫「能在白板上畫出並辯護 X」，不寫「理解 X」
- 每節點至少 3 題，且要跨題型（concept / debug / tradeoff / numeric / scenario）
- 外部連結交付前實際打過確認回 200

## 進度保存

進度存在瀏覽器 `localStorage`（key 由 `META.storage_key` 決定）。頁面上的「進度」按鈕會秀出整包 JSON，
**定期複製一份存起來**；換瀏覽器或清快取後貼回去按「載入」即可還原。
