# 01 — Roofline mini-demo（S1）

**展示概念**：同一張 GPU，矩陣的「形狀」就決定你被算力還是記憶體頻寬卡住。

## 怎麼跑

```bash
# 需要 torch（有 CUDA 的環境最有意義）
python run.py

# 指定 dtype
python run.py --dtype bf16

# 給定硬體峰值 → 標出 ridge point 與利用率（範例為 H100 SXM 約略值）
python run.py --peak-tflops 990 --peak-bw 3.35
```

## 量到什麼

每個形狀印出：算術強度 `AI`(FLOPs/Byte)、達成 `TFLOPS`、達成 `GB/s`、`bound`（給了峰值才判斷）、`util`。

| 欄位 | 意義 |
|---|---|
| `AI` | FLOPs ÷ Bytes；小 = memory-bound，大 = compute-bound |
| `TFLOPS` | 實際達成算力；瘦長矩陣會遠低於峰值 |
| `GB/s` | 實際達成記憶體頻寬 |
| `util` | compute-bound 看算力利用率、memory-bound 看頻寬利用率 |

## 預期觀察

- `GEMV (1×4096·4096)`、`skinny`：**AI 接近常數（fp16 ≈ 1–2）**，達成 TFLOPS 很低 → 落在 roofline 的斜線（memory-bound）區。
- `square 2048/4096/8192`：**AI 隨邊長線性變大**，達成 TFLOPS 逼近峰值 → 落在平頂（compute-bound）區。
- 給了 `--peak-*` 後，`bound` 欄會顯示每個形狀在 ridge point 的哪一側。

> 講解重點：**決定速度的不是 FLOPs 總量，而是「每搬一個 byte 能做幾次運算」。** 這正是後面 S3 解釋「為什麼自迴歸 decode（GEMV-like）是 memory-bound」的同一把尺。

## 量測模型與注意事項

- FLOPs = `2·M·K·N`；Bytes = `dtype_size·(M·K + K·N + M·N)`（理想下界，假設各讀一次、C 寫一次）。
- 真實 kernel 因 tiling 可能重讀資料，達成頻寬可能超過此模型估計；此 demo 用標準模型建立直覺即可。
- 量測有 warmup + `torch.cuda.synchronize()` + 多次取中位數；CPU 退化模式僅供功能驗證。
