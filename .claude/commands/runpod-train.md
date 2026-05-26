---
name: "RunPod: Train LoRA"
description: 在 RunPod 上一鍵跑一次 SDXL LoRA 訓練（選區/卡、建唯一 volume、SECURE、launcher、輪詢）
category: lora-image-gen
tags: [runpod, lora, training, gpu]
---

在 RunPod 上跑一次 SDXL LoRA 訓練的完整流程。把過去踩過的坑固化成可重複步驟。

**前提**：`lora-image-gen/runpod/.env` 已存在（含 `RUNPOD_API_KEY`、`RCLONE_DRIVE_CONFIG`、
`GDRIVE_DEST_PATH` 等；機密不入版控）；本機 SSH public key 已註冊到 RunPod Console。
所有指令在 `lora-image-gen/runpod/` 下執行，Python 用 `../.venv/bin/python`。

**關鍵約束（別重蹈覆轍）**
- **cloud type 必須 SECURE**：掛 Network Volume 只能用 Secure Cloud。GPU 庫存/價格分 cloud
  type，用 COMMUNITY 去搶只有 SECURE 有貨的卡會一直假性「無容量」。確保 `.env` 有
  `RUNPOD_CLOUD_TYPE=SECURE`。（見記憶 runpod-cloud-type-secure）
- **volume 鎖區**：pod 掛 Network Volume 會被鎖在 volume 所在資料中心；只有
  `storageSupport=True` 的區能建 volume。所以「卡有貨的區」必須同時「能建 volume」。
- **同時只留一個 volume**：volume 按容量計費，換區時用 `ensure-single` 刪掉舊的。

**Steps**

1. **找有貨且能建 volume 的 GPU**（< 預算、足夠 VRAM、Secure 有貨）
   ```bash
   ../.venv/bin/python -m launcher.find_gpu --env .env --max-price 1.0 --min-vram 24 --min-stock medium
   ```
   - 看輸出挑一個（優先 High/Medium 庫存；Low 也行但要靠 launcher 重試搶）。
   - 記下它的「資料中心」與「GPU 型別」。若沒有 Medium 以上，放寬 `--min-stock low`。
   - 用 **AskUserQuestion** 讓使用者確認要用哪個區/卡（這會建立計費資源）。

2. **建唯一 volume**（在選定的區，並刪掉其他所有 volume）
   ```bash
   ../.venv/bin/python -m launcher.volume_admin --env .env ensure-single --dc <DC> --size 60 --name lora-vol
   ```
   - 從輸出取 `VOLUME_ID=...`。
   - 建立計費資源（Network Volume）前，先向使用者確認。

3. **更新 `.env`**（只動這幾行，不讀印機密值）：
   - `RUNPOD_DATA_CENTER_ID=<DC>`
   - `RUNPOD_NETWORK_VOLUME_ID=<新 volume id>`
   - `RUNPOD_GPU_TYPE=<選定的 GPU>`（可逗號分隔多款候選，launcher 會輪流搶）
   - `RUNPOD_CLOUD_TYPE=SECURE`
   用 Python 就地改檔，避免印出 API key / rclone token。

4. **啟動 launcher**（會建計費 GPU pod；先向使用者確認）
   ```bash
   ../.venv/bin/python -u -m launcher.launch --env .env --dataset ../dataset_prep/cropped --create-retries 40
   ```
   背景跑 + 用 **Monitor** 串流，依序會看到：搶到卡 → 建 pod → 印 ComfyUI/SSH →
   tar 上傳資料 → kickoff（pod 端：新 volume 缺模型則先下載 → 裝 sd-scripts → 訓練 →
   rclone 同步 Drive）→ 輪詢完成標記。

5. **輪詢 pod 端進度**（launcher 會自動輪詢；要手動看用 Monitor，判斷以「哪個程序在跑」為準，
   不要用易誤判的 pgrep 關鍵字）：
   - 下模型：`ps aux | grep [w]get`
   - 裝環境：`ps aux | grep "[p]ip install"`
   - 訓練中：`ps aux | grep "[s]dxl_train_network"`
   - 完成：`/workspace/training/<concept>.run.done`；失敗：`...run.failed`

6. **結果**
   - 成功：LoRA 在 `/workspace/models/loras/<concept>.safetensors`，且已同步到 Google Drive；
     依 `KEEP_POD` 決定回收。可用印出的 ComfyUI URL 載入 LoRA 肉眼驗證風格。
   - **訓練或同步失敗時 pod 不回收**（產出留在 volume），可 SSH 進去查或重跑同步。

**收尾提醒**
- 用完記得回收 pod（省 GPU 費）：`runpodctl remove pod <id>` 或 RunPod Console。
- 不再需要的 volume 要手動刪（持續按容量計費）。

**Guardrails**
- 建 volume / 建 pod 都是計費操作，執行前向使用者確認。
- 改 `.env` 時不要把 API key / rclone token 印到輸出。
- 一次只保留一個 volume（用 ensure-single）。
