## Context

`lora-image-gen/` 的資料準備流程已完成並同步成正式 specs（`card-illustration-cropping`、`style-caption-generation`）：本機 `dataset_prep/cropped/` 內已有成對的裁切圖與 caption `.txt`，即訓練輸入。`runpod/` 目錄已有手動部署的素材——pod 端 `scripts/start_comfy.sh`（把 ComfyUI 指向 Network Volume）、本機 `client/comfy_client.py`（連遠端 ComfyUI 產圖）、以及 `DEPLOY.md` 的人工步驟。

目前缺口：**訓練本身全靠手動**。沒有可重現的入口把「建 pod → 上傳資料 → 訓練 → 存檔 → 回收」串起來，產出也只留在 Network Volume、沒有離線備援。本設計補上這個一鍵 launcher。

**約束：**
- 使用者只想提供一個 `.env`（API key 等機密）就能啟動，機密不入版控。
- 偏好把產出存到既有的 Google Drive（5TB），本機不必常駐下載。
- 需要 ComfyUI Web UI 與 SSH 兩種除錯介面。
- 這是**研究 / 學習用的一次性訓練**，不是長期共享的線上服務。

## Goals / Non-Goals

**Goals:**
- 一支本機 launcher，讀 `.env` 即可在 RunPod 上跑完一次 SDXL LoRA 訓練的完整 ephemeral 流程（launch → 上傳 → 訓練 → 同步 → terminate）。
- 產出（LoRA 權重、訓練 log）自動以 rclone 同步到 Google Drive，授權非互動式注入。
- pod 對外開 ComfyUI（8188，經 proxy）與 SSH，launcher 印出連線資訊。
- 訓練器、訓練超參數可由設定替換 / 覆寫，不改程式碼。

**Non-Goals:**
- 不做 Network Volume 的建立（官方 SDK 不支援；由使用者於 Console / CLI 預先建立並提供識別碼）。
- 不做宣告式 IaC 狀態管理（無 Terraform/Pulumi state、無 drift reconcile）——這是一次性 imperative 任務。
- 不做多 pod 編排、佇列、排程或長開推論服務（本變更只負責單次訓練；推論沿用既有 `comfy_client.py`）。
- 不改動 `dataset_prep` 既有行為，只消費其產出。
- 不處理資料蒐集 / 版權（`dataset_prep/README.md` 已說明，屬人工步驟）。

## Decisions

### 決策 1：launcher 採 RunPod 官方 Python SDK（`runpod`），而非 Terraform / Pulumi / 純 shell

RunPod 官方主推的 IaC 表面是 2025/03 推出的 **REST API**（`rest.runpod.io/v1`），`runpodctl` CLI 與官方 Python SDK 都是它的 first-party 包裝。對「launch → 訓練 → 存檔 → terminate」這種一次性 imperative 任務，官方自家的「AI on a schedule」指引就是建議用 launch + terminate API 呼叫實作 ephemeral pod，不是宣告式 state。

- **為何不用 Terraform**：Terraform Registry 上的 RunPod provider 是社群第三方（非 RunPod org / 非 HashiCorp partner），成熟度低；且一次性訓練用 state 檔反而是負擔。
- **為何不用 Pulumi**：`runpod/pulumi-runpod-native` 雖是 RunPod org 自家，但最後發版停在 2024/07，落後於 2025/03 的新 REST API，不夠 current。
- **為何不用純 shell + `runpodctl`/curl**：能做但 pod 狀態輪詢、錯誤處理、token 注入用 Python 較乾淨、可測；SDK 直接提供 `create_pod(...)` / `terminate_pod(...)`，型別清楚。lora-image-gen 已有 uv venv 可裝 `runpod`。
- **取捨**：SDK 目前底層走 GraphQL（非新 REST），且不支援建立 Network Volume——後者透過「由使用者預先建立」迴避，前者對本任務無影響。

### 決策 2：ephemeral pod，訓練啟動指令注入 container start command

pod 以 `create_pod(...)` 帶 `network_volume_id`、`gpu_type_id`、`ports`、`env`（注入 rclone token 等），訓練流程以 **container start command / docker args** 注入：開機即依序跑 `setup → 訓練 → rclone 同步`，跑完讓 launcher 偵測完成後 terminate。

- 訓練主邏輯放 pod 端腳本（如 `scripts/train_lora.sh`），由 start command 觸發；launcher 負責生命週期與輪詢，不把訓練邏輯塞進 launcher。
- **保留 pod 旗標**：失敗或使用者要求時不 terminate，方便 SSH 進去看 log。
- **替代方案**：用 `dockerEntrypoint` 覆寫成純 batch job 更乾淨，但會關掉 ComfyUI/SSH 這些除錯服務；本任務需要除錯介面，故選「template 既有服務 + 注入 start command 跑訓練」。

### 決策 3：產出持久化用 rclone → Google Drive，token 由 `.env` 注入（headless）

pod 是 headless 無瀏覽器，rclone 對 Google Drive 的互動式 OAuth 會失敗。標準解法是**在本機先 `rclone authorize "drive"` 拿到 token**，把整段 rclone remote 設定（含 token）透過 `.env` / 環境變數注入 pod，pod 端不互動授權。

- 同步在訓練成功後跑；**同步失敗則不回收 pod**（產出仍在 Network Volume，可手動取回），避免「pod 已砍、Drive 沒同步成功」兩頭空。
- **替代方案**：S3/R2 物件儲存對自動化更順，但使用者資產在 Google Drive（5TB 現成），故用 Drive。
- **替代方案**：只留 Network Volume——最簡但無離線備援、volume 持續計費，且本機要時再用 `runpodctl`/scp 拉，違背「不必常駐下載」需求。

### 決策 4：訓練框架預設 kohya_ss（sd-scripts），但當作可替換的 step

base 模型是 Dreamshaper XL（SDXL），kohya_ss / sd-scripts 是 SDXL LoRA 訓練的社群事實標準、資料最多、RunPod template 現成。spec 把「執行訓練」定義為吃資料集 + 設定、產出 LoRA 的一個 step，pod 端訓練腳本與 config 可替換成 ai-toolkit 等，不影響 launcher 與生命週期邏輯。

### 決策 5：Network Volume 路徑沿用 `/workspace` 既有約定

沿用既有 `start_comfy.sh` 的 `/workspace` 結構：資料集上傳到 `/workspace/datasets/<concept>/`，訓練輸出寫 `/workspace/models/loras/` 與 log 目錄，與既有 `extra_model_paths.yaml` 一致——訓練好的 LoRA 立刻能被同一個 pod 的 ComfyUI 載入驗證。

## Risks / Trade-offs

- **rclone refresh token 過期** → 對無人值守任務，user OAuth 的 refresh token 可能失效。緩解：文件指引可改用 service account；同步失敗時不回收 pod、明確報錯，產出不致遺失。
- **官方 SDK 底層仍走 GraphQL、非新 REST** → 未來 SDK 可能變動。緩解：把 RunPod 互動集中在一個薄封裝模組，日後要換 REST/CLI 只改一處。
- **Network Volume 資料中心鎖定** → volume 建立時綁資料中心，pod 必須開在同資料中心否則掛不上。緩解：設定驗證階段檢查 / 啟動失敗時給明確訊息指出資料中心不符。
- **start command 跑訓練、launcher 輪詢完成的判定** → 需可靠判斷「訓練+同步已結束」才 terminate，否則可能砍早或砍晚。緩解：pod 端流程結束寫一個完成標記（檔案 / log 關鍵字），launcher 輪詢該標記再回收；逾時上限避免無限等待。
- **機密外洩** → API key、rclone token 經 `.env` 與 pod env。緩解：`.gitignore` 排除 `.env`、只提交 `.env.example`、日誌不印機密明文。
- **GPU 計費忘了回收** → 保留 pod 旗標或同步失敗時 pod 不回收，可能持續計費。緩解：保留時明確提示需自行回收；輸出 pod 識別碼與回收指令。

## Migration Plan

- 純新增，無破壞性變更：不動既有 `dataset_prep`、`client/comfy_client.py`、`start_comfy.sh`。
- 前置：使用者於 RunPod Console 建立 Network Volume（記資料中心與識別碼）、本機 `rclone authorize "drive"` 取得 token、填好 `.env`。
- rollback：刪除 launcher 與新增腳本即可回到純手動流程；已建立的 pod 以 `terminate_pod` 或 Console 回收。

## Open Questions

- 訓練超參數的預設值（rank/alpha/lr/steps）以小資料集（數十張卡）為準，待 apply 時依實測校準。
- 是否在 launcher 內附「訓練後自動跑一張驗證圖」的選項，或留給使用者手動經 ComfyUI 驗證——傾向後者（除錯介面已提供），但可視需要加旗標。
