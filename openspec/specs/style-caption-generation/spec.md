---
project: lora-image-gen
---

# style-caption-generation Specification

## Purpose

依 CSV 對照表為訓練圖產生符合風格 LoRA 規範的 caption（觸發詞前綴、剝除畫風詞、缺描述警告），輸出與圖同名的 `.txt`。

## Requirements

### Requirement: 依對照表產生 caption

系統 SHALL 讀取含表頭 `filename,description` 的 CSV 對照表，為輸出資料夾內每張支援格式的圖片產生與圖同名的 `.txt` caption，caption 內容為觸發詞前綴加上對應的物件描述。

#### Scenario: 圖片於 CSV 有對應描述

- **WHEN** 圖片檔名（去副檔名）在 CSV 找得到對應描述
- **THEN** 系統產出內容為「觸發詞, 物件描述」的同名 `.txt`

#### Scenario: CSV 缺少必要表頭

- **WHEN** CSV 缺少 `filename` 或 `description` 欄位
- **THEN** 系統 SHALL 報錯並以非零結束碼中止

#### Scenario: CSV 鍵以檔名比對且可含副檔名

- **WHEN** CSV 的 filename 欄含或不含副檔名
- **THEN** 系統 SHALL 以去除副檔名後的檔名作為比對鍵

### Requirement: 觸發詞前綴

系統 SHALL 在每個 caption 開頭加上使用者指定的觸發詞（預設為一個模型不認識的字串），用以在推論時召喚風格。

#### Scenario: 使用預設觸發詞

- **WHEN** 使用者未指定觸發詞
- **THEN** 系統使用預設觸發詞作為每個 caption 的前綴

#### Scenario: 自訂觸發詞

- **WHEN** 使用者指定觸發詞
- **THEN** 系統以該觸發詞作為每個 caption 的前綴

### Requirement: 自動剝除畫風描述詞

系統 SHALL 預設從物件描述中剝除描述「畫風」的詞（如 flat、cute、cartoon、crayon、illustration 等，比對忽略大小寫、整詞比對），以避免畫風被寫入 caption 而稀釋風格綁定；並 SHALL 提供旗標保留這些詞。

#### Scenario: 預設剝除畫風詞

- **WHEN** 描述含畫風詞且使用者未要求保留
- **THEN** 系統從 caption 移除這些詞、清理多餘空白與標點，並回報被剝除的詞清單與原因

#### Scenario: 要求保留畫風詞

- **WHEN** 使用者帶入保留畫風詞旗標
- **THEN** 系統不剝除任何詞，照原描述產生 caption

### Requirement: 缺描述警告與純觸發詞回退

系統 SHALL 對在 CSV 中找不到描述的圖片，產生僅含觸發詞的 caption，並在執行結束時列出這些缺描述的圖片以供使用者補齊。

#### Scenario: 圖片無對應描述

- **WHEN** 圖片在 CSV 找不到對應描述
- **THEN** 系統產生僅含觸發詞的 caption，並在結束時警告該圖缺描述

### Requirement: 乾跑預覽模式

系統 SHALL 提供乾跑模式，只印出每張圖將產生的 caption 與被剝除的詞，不寫出任何 `.txt` 檔。

#### Scenario: 啟用乾跑

- **WHEN** 使用者帶入乾跑旗標
- **THEN** 系統印出每張圖的預期 caption 與剝除的畫風詞，但不寫入任何檔案
