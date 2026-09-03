# 技能樹

Path of Exile 風格的 ROS 2 學習技能樹。69 節點、11 章、443 小時。

## 開啟

```bash
# 建議：用 http server（localStorage 在 file:// 下 Safari 會擋）
python3 -m http.server 8080
# 開 http://localhost:8080

# 或直接開（Chrome 可以）
open index.html
```

## 操作

| 操作 | 說明 |
|---|---|
| 拖曳 / 滾輪 | 平移 / 縮放 |
| 點擊節點 | 開啟右側面板：說明、為什麼重要、任務清單、驗收條件、資源連結 |
| 勾任務 | 節點進度累積；全勾完 → 節點點亮、解鎖後續、跳出 toast |
| **▶ 下一步** | 自動跳到建議的下一個節點 |
| **定位** | 回到目前該做的節點 |
| **全覽** | 縮到看得見整棵樹 |
| 搜尋框 | 輸入關鍵字 + Enter 跳轉 |
| 左下角章節列 | 點擊跳到該章 |

## 節點視覺

| 樣式 | 意義 |
|---|---|
| 🔒 灰色 | 鎖定（前置未完成） |
| 藍色發光 | 可開始 |
| 橘色 + % | 進行中 |
| 金色 ✓ | 完成 |
| 小圓 / 大圓 / **菱形** | 一般 / 關鍵 / **樞紐** |
| 虛線邊框 | 選修支線 |
| 金色連線 | 兩端都完成 |
| 藍色連線 | 前置已完成，可往下走 |

## 進度

存在瀏覽器 `localStorage`（key: `ros2-skilltree-v1`）。

**定期按「匯出進度」** → 存下 `ros2-skilltree-progress.json` → commit 進 repo。
換瀏覽器或清快取後用「匯入」還原。

## 修改內容

單一資料來源是 `tree_data.py`：

```bash
vim tree_data.py       # 改節點
python3 build.py       # 重新產生 skill-tree.json + index.html
```

`build.py` 會驗證：重複 id、不存在的前置、依賴循環、空任務清單。
產生後會印出各章節點數與時數統計。

**進度以 node id 為 key，重新 build 不會遺失進度。**
（除非你刪掉節點或改了 id）

## 檔案

| 檔案 | 用途 |
|---|---|
| `tree_data.py` | ⭐ 單一資料來源。改這個 |
| `build.py` | 產生器 + 驗證器 |
| `index.template.html` | UI 樣板（含 `/*__TREE_JSON__*/` 佔位） |
| `index.html` | 產生物，**不要直接改** |
| `skill-tree.json` | 產生物，供程式讀取 |

## 等級

XP = 完成的任務比例 × 節點時數 × 10。每 120 XP 升一級，滿等約 33 級。

| 等級 | 稱號 |
|---|---|
| 1–5 | 見習生 · Apprentice |
| 6–10 | 節點工匠 · Node Artisan |
| 11–15 | 系統整合者 · Integrator |
| 16–20 | 模擬架構師 · Sim Architect |
| 21–25 | 具身智慧工程師 · Embodied AI |
| 26+ | 機器人大師 · Robotics Master |
