# 技術設計：Stacklands 風格 LoRA 資料準備流程

## Context

要訓練重現 Stacklands 卡牌插圖畫風的 SDXL LoRA，需要把「整張卡牌截圖」轉成「乾淨插圖 + 正確 caption」的訓練集。素材來源為遊戲 / wiki 截圖，特性是：版面規格一致（米色圓角框 + 中央插圖 + 卡名 + icon）、插圖本身低細節且尺寸偏小。

風格 LoRA 的兩條鐵律主導本設計：

1. 訓練圖須只含「要學的視覺特徵」——卡框與版面屬雜訊，必須移除，否則會被學成風格。
2. caption 描述「畫的是什麼物件」、不描述「怎麼畫」——畫風留白才能被 LoRA 綁定。

下游（放大、訓練、推論）已規劃在 ComfyUI on RunPod，不屬本流程範圍。

## Goals / Non-Goals

**Goals:**

- 把整張卡截圖批次轉成去框的中央插圖，過程可校準、可重現。
- 為每張插圖產生符合風格 LoRA 規範的 caption（觸發詞前綴、剝除畫風詞）。
- 最小相依：裁切僅用 Pillow，caption 純標準函式庫。
- 拆成單一職責的小工具，符合「many small files」原則。

**Non-Goals:**

- 自動蒐集 / 爬取卡牌圖（Fandom 擋抓取且有版權 / ToS 風險，維持手動）。
- 插圖放大（改在 ComfyUI 以 AI upscaler 處理，避免在此引入重相依）。
- LoRA 訓練本身與訓練參數調校。
- 影像偵測式自動去框（見下方決策，刻意不採用）。

## Decisions

### 決策 1：裁切採固定比例而非影像偵測

以相對比例（0~1）定義裁切框，對整批套用同一框；提供 `--preview` 先校準一張。

- **為何**：素材版面規格一致，固定比例足夠且行為可預期、無額外相依；OpenCV 邊框偵測複雜、易在低對比卡面失準。
- **替代方案**：OpenCV 輪廓 / 邊緣偵測自動找插圖區 —— 否決，理由是相依重、結果不穩、除錯成本高。
- **代價**：素材尺寸雜亂時需分批以不同比例校準。已於 spec 與 README 載明此前提。

### 決策 2：caption 來源用人工 CSV 對照表，而非自動 tagger

caption 物件描述來自使用者整理的 `cards.csv`（源自 wiki 的人寫描述），而非 WD14 / BLIP。

- **為何**：wiki 已有針對每張卡的精確描述，比自動 tagger 準；且免去 onnxruntime / torch 等重相依。
- **替代方案**：WD14 tagger（需 onnxruntime + 下載模型）或 BLIP（需 transformers + torch）—— 否決，相依過重且準確度未必勝過人寫描述。
- **代價**：需手動整理 CSV。以 `cards.csv.example` 降低門檻。

### 決策 3：以「整詞比對」自動剝除畫風詞，預設開啟

維護一份畫風詞集合（flat、cute、cartoon、crayon、illustration、stacklands、card、icon…），以正則整詞、忽略大小寫剝除，並清理殘留空白標點。

- **為何**：直接在工具層守住「caption 不描述畫風」鐵律，避免人為疏漏；同時剝除商標 / 版面詞（stacklands / card / icon）。
- **替代方案**：完全交給使用者人工把關 —— 否決，易遺漏且不可重現。
- **逃生口**：`--keep-style-words` 供進階使用者保留。

### 決策 4：兩支獨立腳本 + 共享流程文件

`crop_cards.py`（裁切）與 `make_captions.py`（打標）職責分離，各自可獨立執行與測試，以 `dataset_prep/README.md` 串接流程。

- **為何**：高內聚低耦合；裁切可重跑校準而不重打標，反之亦然。

### 決策 5：caption 與圖同名 `.txt`，圖統一輸出 PNG

採 kohya / ComfyUI 通用慣例：每張圖旁放同名 `.txt`；裁切輸出統一 PNG（保留透明、無壓縮 artifact）。

- **為何**：與下游訓練工具無縫銜接，無需額外轉換。

## Risks / Trade-offs

- [素材尺寸不一致導致固定比例切歪] → 提供 `--preview` 單張校準；README 載明分批校準做法。
- [插圖裁出後過小，放大會糊] → 放大移至下游 ComfyUI 以插畫專用 AI upscaler 處理；本流程僅負責乾淨裁切。
- [畫風詞集合不完整，漏剝某些詞] → 集合集中可擴充；`--dry-run` 讓使用者先檢視剝除結果再寫檔。
- [CSV 與圖檔名對不上，caption 只剩觸發詞] → 工具於結束時列出缺描述清單供補齊，不靜默略過。
- [版權] → 限個人學習 / 研究；公開或商用須另行評估（美術屬 Sokpop Collective）。工作目錄不入版控。
