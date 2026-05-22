# LoRA 產圖研究

研究 LoRA（Low-Rank Adaptation）在文生圖 / 圖生圖模型上的微調與應用，從原理到實作、訓練到推論。

## 研究主題

- **原理**：LoRA / DoRA / LoHa / LoKr 等低秩微調方法，rank、alpha、學習率的影響
- **基礎模型**：SD 1.5、SDXL、SD3、FLUX 等不同 backbone 上的 LoRA 行為差異
- **訓練**：資料集準備、打標（captioning）、訓練超參數、過擬合與風格遷移
- **推論**：LoRA 權重合併、多 LoRA 疊加、權重縮放（weight scaling）
- **評估**：產圖品質、風格一致性、prompt 對齊度的量化與主觀評估

## 目錄結構

| 目錄 | 用途 |
| --- | --- |
| `runpod/` | **ComfyUI on RunPod 部署**：遠端 GPU + model service，本機連 API（見下方） |
| `dataset_prep/` | **資料準備工具**：卡牌截圖裁框 + 自動打標（目前案例：Stacklands 風格） |
| `notebooks/` | 探索與實驗用的 Jupyter notebooks |
| `datasets/` | 訓練資料集（圖片 + caption），大檔不入版控 |
| `configs/` | 訓練 / 推論設定檔（kohya_ss、diffusers 等） |
| `scripts/` | 訓練、推論、資料前處理腳本 |
| `outputs/` | 產出的圖片與訓練好的 LoRA 權重，不入版控 |
| `docs/` | 研究筆記、論文摘要、實驗紀錄 |

## 執行環境：RunPod 遠端 GPU

採 **ComfyUI client-server 分離**架構——RunPod 跑 ComfyUI server 當 GPU + 模型服務，
模型放 RunPod Network Volume（只下載一次），本機透過 API 提交 workflow 產圖、訓練 LoRA。
完整部署步驟見 [runpod/DEPLOY.md](runpod/DEPLOY.md)，架構說明見 [runpod/README.md](runpod/README.md)。

目前基礎模型：Dreamshaper XL v2 Turbo（SDXL）+ sdxl-vae-fp16-fix。

## 環境

待補：列出使用的框架（diffusers / kohya_ss / ComfyUI）、Python 版本與相依套件。

## 參考資源

待補：論文連結、教學、模型來源。
