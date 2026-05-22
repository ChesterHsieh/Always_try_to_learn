## 1. 裁切工具（card-illustration-cropping）

- [x] 1.1 建立 `dataset_prep/crop_cards.py`，以相對比例（上/下/左/右）批次裁切，輸出同名 PNG
- [x] 1.2 實作比例值域驗證（0~1）與裁切框有效性檢查，無效時報錯中止
- [x] 1.3 實作 `--preview` 單張校準模式並印出校準提示
- [x] 1.4 實作 `--square` 正方形補白輸出
- [x] 1.5 實作來源資料夾不存在報錯、無支援格式圖片時警告
- [x] 1.6 補自動化測試：以合成卡牌圖驗證裁切框、square、值域錯誤、空資料夾各情境

## 2. 打標工具（style-caption-generation）

- [x] 2.1 建立 `dataset_prep/make_captions.py`，讀 CSV（filename,description）產同名 `.txt`
- [x] 2.2 實作觸發詞前綴（預設 `stcklnd`，可 `--trigger` 覆寫）
- [x] 2.3 實作畫風詞自動剝除（整詞、忽略大小寫）與 `--keep-style-words` 逃生口
- [x] 2.4 實作缺描述警告與「僅觸發詞」回退
- [x] 2.5 實作 `--dry-run` 乾跑預覽
- [x] 2.6 實作 CSV 缺表頭報錯、檔名鍵去副檔名比對
- [x] 2.7 補自動化測試：對應/缺描述/畫風詞剝除/dry-run/缺表頭各情境

## 3. 範本與文件

- [x] 3.1 建立 `dataset_prep/cards.csv.example` 範本
- [x] 3.2 撰寫 `dataset_prep/README.md`（流程、用法、整理 CSV 訣竅、版權注意）
- [x] 3.3 更新 `lora-image-gen/README.md` 串入 dataset_prep
- [x] 3.4 更新 `lora-image-gen/.gitignore` 排除 raw_cards/cropped/cards.csv

## 4. 驗證與整合

- [x] 4.1 以合成卡牌圖手動驗證裁切（去框正確、square 正確）
- [x] 4.2 以合成資料手動驗證打標（畫風詞剝除、缺描述警告、dry-run）
- [x] 4.3 收斂前述自動化測試成可重複跑的測試套件（pytest），達 rules 要求的覆蓋率（25 passed、覆蓋率 100%）
- [x] 4.4 以一小批真實 Stacklands 卡牌截圖端到端跑通：裁切 → 整理 cards.csv → 產 caption
      （33 張卡：比例校準 top0.23/bottom0.77/left0.11/right0.89 + square；檔名自動生成 cards.csv，人工修正語義卡；33 張全配對 caption，零缺漏）
