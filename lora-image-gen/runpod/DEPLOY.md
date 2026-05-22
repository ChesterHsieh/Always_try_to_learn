# 部署指南：ComfyUI on RunPod（GPU + Model Service）

從零到「本機連遠端產圖」與「在 RunPod 上訓練 LoRA」的完整步驟。
架構說明見 [README.md](./README.md)。

---

## 階段 1：建立 Network Volume（持久化模型存放）

Network Volume 是 pod 之間共享、pod 關掉也不會消失的儲存。模型只下載一次。

1. RunPod Console → **Storage** → **Network Volume** → **+ New Network Volume**
2. 選一個 **資料中心（Data Center）**——記住它，**pod 必須開在同一個資料中心**才能掛載
3. 容量建議：
   - 只做推論：**30 GB**（SDXL checkpoint ~7GB + VAE + 幾個 LoRA）
   - 含訓練：**60–100 GB**（再加訓練資料集、訓練快取、產出的多個 LoRA）
4. 命名，例如 `lora-imggen-vol`，建立

> 💡 Network Volume 是按容量計費（約 $0.05–0.07/GB/月），即使沒開 pod 也會收。用不到時可縮容或刪除。

---

## 階段 2：開 Pod（掛上 Volume + ComfyUI template）

1. RunPod Console → **Pods** → **Deploy**
2. **GPU**：依用途選（見 README 的規格表）
   - 推論：RTX 4090 / A5000（24GB）
   - 訓練：A6000（48GB）較穩
3. **Network Volume**：選剛建立的 `lora-imggen-vol`，掛載點填 `/workspace`
   （⚠️ 必須選在 Volume 同一個資料中心，否則清單裡不會出現）
4. **Template**：搜尋 **ComfyUI**（社群有多個現成 template，如 `runpod/comfyui` 或 ashleykza 的版本）
5. 確認 **HTTP Ports** 有開 **8188**（ComfyUI 預設 port）
6. Deploy，等 pod 起來

---

## 階段 3：初始化 Volume + 下載模型（在 Pod 上做一次）

開 pod 的 **Web Terminal** 或用 SSH 連進去，把本資料夾的 scripts 弄上去執行。

把腳本貼上去最快的方式（在 pod terminal）：

```bash
# 在 pod 上，從本機 scp 或直接 git clone 你的 repo；這裡假設已取得 scripts/
cd /workspace
bash setup_volume.sh      # 建目錄結構 + 下載 Dreamshaper XL / VAE 到 Network Volume
```

`setup_volume.sh` 會在 `/workspace/models/{checkpoints,vae,loras,...}` 建好結構並下載基礎模型。
模型寫進 Network Volume，**之後換 pod 都不用重下**。

> 若 Civitai 模型需要 token，把 `setup_volume.sh` 裡的下載 URL 換成你自己的來源，或先在本機下載再 scp 上去（見階段 6）。

---

## 階段 4：啟動 ComfyUI 並取得連線 URL

多數 ComfyUI template 開機就自動跑 ComfyUI 了。若要讓它讀 Network Volume 上的模型，跑：

```bash
cd /workspace
bash start_comfy.sh       # 寫好 extra_model_paths.yaml 指向 Volume，並啟動 server
```

啟動後：
1. RunPod Console → 該 pod → **Connect**
2. 找 **HTTP Service → port 8188**，複製那個 URL，形如
   `https://<pod-id>-8188.proxy.runpod.net`

這個 URL 就是本機 client 要連的位址。

---

## 階段 5：本機連遠端產圖

本機（這個 repo 所在的電腦）：

```bash
cd lora-image-gen/runpod
export RUNPOD_COMFY_URL="https://<pod-id>-8188.proxy.runpod.net"

# 用現有的 Dreamshaper XL Turbo 產圖
python client/comfy_client.py workflows/sdxl_turbo_txt2img.json \
  --out ./out \
  --positive "a cinematic photo of a red fox in fresh snow, soft light" \
  --seed 42
```

圖片會被抓回本機的 `./out`。client 只用 Python 標準函式庫，不用裝額外套件。

> 也可以直接開瀏覽器到那個 proxy URL，用 ComfyUI 的網頁 UI（UI 在遠端 render）。
> 或用 **ComfyUI 桌面版** 設定遠端 server。client 腳本則適合做批次 / 自動化 / 實驗紀錄。

套用訓練好的 LoRA：把 LoRA 放到 `/workspace/models/loras/`，用
`workflows/sdxl_turbo_lora_txt2img.json`，改裡面的 `lora_name` 即可。

---

## 階段 6：把本機現有模型傳上去（可選）

你截圖裡的 Dreamshaper XL / VAE 若已在本機，不想重下，可直接傳上 Network Volume：

```bash
# 需先在 pod 設定 SSH（RunPod Connect 頁有 SSH 指令與 port）
scp -P <ssh-port> /path/to/DreamShaperXL.safetensors \
  root@<pod-ip>:/workspace/models/checkpoints/
```

或用 `runpodctl send` / `receive` 在無 SSH 設定時傳檔。

---

## 階段 7：訓練 LoRA（之後展開）

訓練建議另開一個帶訓練框架的 pod（**kohya_ss** 或 **ai-toolkit**），同樣掛 `/workspace`：

- 資料集放 `/workspace/datasets/<my-concept>/`
- 訓練輸出寫 `/workspace/models/loras/`
- 訓完的 LoRA 立刻能被推論 pod 用（共享 Volume）

詳細訓練設定（rank/alpha/lr、打標）待開訓練 pod 時補上 `scripts/train_lora.sh` 與 `configs/`。

---

## 成本與生命週期小抄

| 動作 | 說明 |
|------|------|
| 用完關 pod | **Stop** 省 GPU 費用，但 container disk 會清空；模型在 Network Volume 仍在 |
| 換更大/更小 GPU | 關 pod 重開，掛同一個 Volume，模型不用重下 |
| 長期不用 | 刪 pod；Network Volume 要留就留（持續收容量費），不留就刪 |
| 訓練省錢 | 用 Spot / Community Cloud；推論服務要長開用 On-Demand |
