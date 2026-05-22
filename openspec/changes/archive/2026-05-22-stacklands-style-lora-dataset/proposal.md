# 提案：Stacklands 風格 LoRA 資料準備流程

## Why

要訓練一個能重現 Stacklands 卡牌「插圖畫風」的 SDXL LoRA，成敗 80% 取決於資料品質。素材來源是整張卡牌截圖（含邊框、卡名、icon），若直接餵入訓練，模型會把卡框與版面也學成風格的一部分；同時 caption 若描述了畫風本身，會稀釋風格綁定。需要一套可重複、低人工的資料準備流程，把「整張卡截圖」轉成「乾淨的插圖 + 正確的 caption」訓練集。

## What Changes

- 新增**卡牌插圖裁切**能力：以固定比例（可校準、可預覽）批次把整張卡截圖裁出中央插圖，去除邊框 / 卡名 / icon；支援補白成正方形以利 SDXL bucketing。
- 新增**風格 caption 產生**能力：依「檔名→物件描述」對照表（CSV，源自 wiki）為每張裁好的圖產生 `.txt` caption，自動加觸發詞、自動剝除描述畫風的詞，並對缺描述的圖提出警告。
- 放大步驟**刻意不納入**本流程：插圖放大改在遠端 ComfyUI 以 AI upscaler 處理。
- 蒐集步驟維持手動：因 Fandom 擋自動抓取且有版權 / ToS 考量，僅規範產出物落地於 `raw_cards/`。

## Capabilities

### New Capabilities
- `card-illustration-cropping`: 把規格一致的整張卡牌截圖，依可校準的固定比例批次裁出中央插圖，去除卡框與版面元素，並可選擇補白成正方形輸出。
- `style-caption-generation`: 依 CSV 對照表為訓練圖產生符合風格 LoRA 規範的 caption（觸發詞前綴、剝除畫風詞、缺描述警告），輸出與圖同名的 `.txt`。

### Modified Capabilities
<!-- 無既有 capability 的需求變更 -->

## Impact

- 程式碼：`lora-image-gen/dataset_prep/`（`crop_cards.py`、`make_captions.py`、`cards.csv.example`、`README.md`）。
- 相依：僅 Pillow（裁切），caption 產生為純標準函式庫，無模型相依。
- 資料流：`raw_cards/`（手動蒐集）→ 裁切 → `cropped/` → caption → 訓練集；放大與訓練屬下游（ComfyUI on RunPod）。
- 版控：`raw_cards/`、`cropped/`、`cards.csv` 不入版控（大量圖片與個人整理資料）。
- 法務：素材屬 Sokpop Collective，限個人學習 / 研究用途；公開或商用需另行評估。
