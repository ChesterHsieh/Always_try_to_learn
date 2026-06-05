# 03 — Decode memory-bound + ASR encoder/decoder（S3）

**展示概念**：為什麼自迴歸 decode 是 memory-bound、加大 batch 為何能提升吞吐；以及同一個 ASR 任務，attention decoder 為何比 CTC encoder 慢。

> ⚠️ 這兩個 demo 的效應需在 **GPU** 上才明顯（CPU 權重太小、會落在 cache，看不出 HBM 頻寬瓶頸）。建議用既有的 RunPod GPU 流程跑。

## A. `run.py` — Decode batch sweep

每一步把整疊權重「讀一遍」，量不同 batch 的單步延遲與吞吐。

```bash
python run.py --peak-bw 3.35           # 預設 32 層 x 4096²（約 1 GB 權重）
python run.py --d-model 4096 --layers 32 --batches 1,2,4,8,16,32,64,128,256,512
```

**預期觀察（GPU）**：

| 現象 | 解讀 |
|---|---|
| batch 小時 `step_ms` 幾乎不變 | 單步被「讀權重」主宰 → memory-bound |
| `tokens/s` 隨 batch 近乎線性上升 | 同一次權重讀取服務更多請求（攤平） |
| batch 跨過 ridge（H100 約 ~300）後 step_ms 才明顯上升 | 算力接手 → 轉 compute-bound |
| `step_ms` 逼近但不低於 `權重 / HBM 頻寬` | 頻寬下限就是 decode 的物理天花板 |

對應投影片：「為什麼 decode 是 memory-bound」「batch 的魔法（throughput vs latency）」。

## B. `asr_proxy.py` — encoder（平行）vs decoder（序列）

同樣的權重、層數、序列長度 T、總 FLOPs，比較「一次平行算」與「逐 token 序列算」的 wall-clock。

```bash
python asr_proxy.py --frames 256 --d-model 1024 --layers 12
```

**預期觀察（GPU）**：`decoder / encoder` 慢數倍～數十倍，**總 FLOPs 卻相同**。原因是序列相依讓每步重讀權重、又吃滿 kernel launch 開銷。

對應投影片：Whisper（attention decoder）vs wav2vec2/Conformer（CTC）為何推論速度差很多。

> 注意：`asr_proxy.py` 是「結構性」代理，不含真實 Whisper 的 cross-attention 與 KV cache；重點在「平行 vs 序列」的延遲差異。

## 共同約定

- 量測前 warmup、用 `torch.cuda.synchronize()` 圍住計時、多次取中位數。
- 非 CUDA 裝置會印警告，數字僅供功能驗證。
