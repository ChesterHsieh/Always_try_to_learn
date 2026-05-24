## Why

資料準備流程（`card-illustration-cropping`、`style-caption-generation`）完成後會在本機產出可訓練的資料集（cropped 圖 + 同名 `.txt` caption），但目前要在 RunPod 上訓練 LoRA 全靠手動：到 Console 點開 pod、貼腳本、複製 proxy URL、訓完手動拉檔。整個流程沒有可重現的入口，也沒有產出的離線備援。本變更要把「在 RunPod 上跑一次 LoRA 訓練」變成讀一個 `.env` 就能一鍵啟動、自動把產出同步到 Google Drive、並提供除錯介面的可重現流程。

## What Changes

- 新增**一鍵 launcher**（採 RunPod 官方 Python SDK）：讀 `.env` 取得 API key 與設定，建立帶 Network Volume 的 GPU pod、注入訓練啟動指令、訓練結束後自動回收 pod（ephemeral：launch → 訓練 → 存檔 → terminate）。
- 新增**訓練資料上傳**：把本機 `dataset_prep/cropped/` 的訓練集（圖 + caption）送上 pod 的 Network Volume，作為訓練輸入；訓練器當作可替換的一個 step（預設 kohya_ss / sd-scripts 跑 SDXL LoRA）。
- 新增**產出持久化到 Google Drive**：pod 上用 rclone 把訓練好的 LoRA、訓練 log（與可選的驗證產圖）同步到使用者的 Google Drive，本機不必常駐下載；rclone 的 OAuth token 由 `.env` 注入（headless，不在 pod 上互動授權）。
- 新增**除錯介面**：pod 對外開 ComfyUI Web UI（port 8188，經 RunPod proxy）與 SSH；launcher 啟動後印出兩者的連線資訊。
- 新增 **`.env.example`** 樣板，明列啟動所需的所有 key 與設定（RunPod API key、Network Volume / 資料中心、GPU、rclone 設定、訓練超參數入口）。

## Capabilities

### New Capabilities
- `runpod-training-launcher`: 從本機一鍵在 RunPod 上跑一次性 SDXL LoRA 訓練的可重現流程——含設定載入與驗證、Network Volume 與 pod 生命週期、訓練資料上傳、訓練執行、產出同步到 Google Drive，以及 ComfyUI/SSH 除錯介面。

### Modified Capabilities
<!-- 無：本變更不更動既有 card-illustration-cropping / style-caption-generation 的需求，只消費其產出。 -->

## Impact

- **新增程式碼**：`lora-image-gen/runpod/` 下新增 launcher（Python，使用官方 `runpod` SDK）、`.env.example`、pod 端訓練啟動腳本（kohya_ss）、rclone 同步腳本；沿用既有 `scripts/start_comfy.sh`、`client/comfy_client.py`。
- **新增相依**：本機 launcher 相依官方 `runpod` Python 套件（裝在 lora-image-gen 既有的 uv venv）；pod 端相依 rclone 與訓練框架（由 template / 啟動腳本安裝）。
- **外部服務 / 機密**：需要 RunPod API key、預先建立的 Network Volume（須與 pod 同資料中心、建立時掛載）、rclone 對 Google Drive 的 OAuth token——全部走 `.env`，不入版控。
- **消費既有產出**：輸入為 `dataset_prep/cropped/`（`card-illustration-cropping` + `style-caption-generation` 的產物），不更動其行為。
- **成本**：GPU pod 為計時計費，ephemeral 模式訓練完即回收以省費；Network Volume 按容量持續計費，由使用者自行決定保留或刪除。
