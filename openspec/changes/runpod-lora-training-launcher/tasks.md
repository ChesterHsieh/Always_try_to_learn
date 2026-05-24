## 1. 設定與機密載入

- [x] 1.1 建立 `lora-image-gen/runpod/.env.example`，列出所有設定鍵：`RUNPOD_API_KEY`、`RUNPOD_NETWORK_VOLUME_ID`、`RUNPOD_DATA_CENTER_ID`、`RUNPOD_GPU_TYPE`、`RCLONE_CONFIG_*`（或整段 rclone remote 設定）、`GDRIVE_DEST_PATH`、訓練超參數入口（觸發詞、rank/alpha/lr/steps）
- [x] 1.2 確認 `lora-image-gen/.gitignore` 已排除 `.env`（只提交 `.env.example`）
- [x] 1.3 在 lora-image-gen 的 uv venv 加入官方 `runpod` 套件相依（pyproject）
- [x] 1.4 實作設定載入與驗證模組：讀 `.env`、檢查必要鍵齊全、缺項時列出鍵名並以非零結束碼中止；確保進度/錯誤訊息不印機密明文

## 2. RunPod 互動薄封裝

- [x] 2.1 建立集中封裝 RunPod SDK 互動的模組（建 pod、查狀態、terminate），日後要換 REST/CLI 只改此處
- [x] 2.2 實作以既有 Network Volume + GPU 型別建立 pod：掛載 volume 到 `/workspace`、開 8188 (http) 與 SSH port、注入 env（含 rclone token）與訓練 start command
- [x] 2.3 實作 pod 狀態輪詢與逾時上限（避免無限等待）
- [x] 2.4 實作 terminate pod，並支援「保留 pod」旗標（保留時印出 pod 識別碼與回收指令）
- [x] 2.5 設定驗證：未提供 Network Volume 識別碼時報錯中止並指示先建立 volume；資料中心不符時給明確訊息

## 3. 訓練資料集上傳

- [x] 3.1 實作把本機 `dataset_prep/cropped/`（圖 + 同名 `.txt`）送至 pod `/workspace/datasets/<concept>/`
- [x] 3.2 上傳前驗證來源存在且含成對圖與 caption；不存在或為空則報錯中止、不啟動訓練
- [x] 3.3 回報已上傳的圖片數

## 4. Pod 端訓練流程

- [x] 4.1 撰寫 pod 端訓練啟動腳本 `scripts/train_lora.sh`（預設 kohya_ss / sd-scripts，SDXL LoRA），吃 `/workspace/datasets/<concept>/` 與設定帶入的超參數
- [x] 4.2 訓練輸出 LoRA 權重寫 `/workspace/models/loras/`、log 寫 log 目錄
- [x] 4.3 訓練超參數（觸發詞、rank/alpha/lr/steps）由 env / 設定帶入，不需改程式碼
- [x] 4.4 訓練流程結束寫「完成標記」（檔案或 log 關鍵字）供 launcher 輪詢；訓練非零結束碼視為失敗

## 5. 產出同步到 Google Drive

- [x] 5.1 撰寫 pod 端 rclone 同步腳本：用注入的 token 非互動授權，把 `/workspace/models/loras/` 與 log 同步到 `GDRIVE_DEST_PATH`
- [x] 5.2 在文件說明本機 `rclone authorize "drive"` 取得 token 並填入 `.env` 的步驟
- [x] 5.3 同步失敗時回報失敗、且不回收 pod；提示產出仍在 Network Volume 可手動取回

## 6. 除錯介面

- [x] 6.1 沿用 `scripts/start_comfy.sh` 讓 pod 起 ComfyUI（8188），確認經 RunPod proxy 可連
- [x] 6.2 launcher 啟動後印出 ComfyUI proxy URL 與可直接使用的 SSH 連線指令
- [x] 6.3 驗證能在該 ComfyUI 載入訓練輸出的 LoRA 產圖（肉眼驗證風格）

## 7. Launcher 主流程整合

- [x] 7.1 串起完整 ephemeral 流程：載入設定 → 建 pod → 上傳資料 → 觸發訓練 → 輪詢完成 → 同步 Drive → 依設定回收 pod
- [x] 7.2 失敗路徑處理：訓練失敗 / 同步失敗時的回報與「不回收 pod」行為
- [x] 7.3 提供一鍵入口（可執行腳本 / `python -m` 入口）並更新 `runpod/README.md`、`DEPLOY.md` 的自動化使用說明

## 8. 測試

- [x] 8.1 設定載入與驗證的單元測試（齊全 / 缺項 / 機密不外洩）
- [x] 8.2 資料集上傳前置驗證的單元測試（來源不存在 / 空 / 有效）
- [x] 8.3 RunPod 互動薄封裝以 mock 測試 pod 生命週期（建立 / 輪詢 / terminate / 保留旗標）
- [x] 8.4 launcher 主流程以 mock 測試成功路徑與失敗路徑（訓練失敗、同步失敗不回收）
- [x] 8.5 確認測試覆蓋率達標（沿用 lora-image-gen 既有 pytest 設定）
