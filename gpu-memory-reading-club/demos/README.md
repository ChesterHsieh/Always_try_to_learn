# Demos — 可重現的記憶體/搬遷效能量測

每個 demo 都對應主規劃裡的一個核心概念，目標是「用最少的程式，量到一個能講的數字」。
共用工具：`torch.cuda.Event` 計時、`torch.profiler`、`nvidia-smi dmon`、Nsight Systems（`nsys`）。

| 目錄 | 對應場次 | 展示概念 | 量測指標 | 預期觀察 |
|---|---|---|---|---|
| `01_roofline_mini/` | S1 | compute vs memory bound | 達成 TFLOPS vs 矩陣形狀 | 瘦長矩陣掉進 memory-bound 區 |
| `02_pinned_vs_pageable/` | S2 | 進出站 / DMA | H2D 頻寬 (GB/s) | pinned 快 ~1.5–2× |
| `03_decode_memory_bound/` | S3 | training vs inference 瓶頸 | tokens/s vs batch size | batch=1 受頻寬限；加 batch 吞吐線性升、單步延遲幾乎不變 |
| `04_prefetch_overlap/` | S4 | **預先搬資料的效益（壓軸）** | 吞吐 / step time | stream overlap、prefetch 明顯提升 |
| `05_flops_vs_parallelism/` | S5 | FLOPs ≠ 速度（平行度與 AI 才是） | GFLOPs / ms / 達成 TFLOPS | LSTM 輸給 FLOPs 更多的 transformer；depthwise FLOPs ÷8.7 但時間只 ÷3.7 |

## 共同約定

- 每個 demo 一個獨立資料夾，含 `run.py`（或 notebook）+ 簡短 `README.md` 記錄「怎麼跑、量到什麼、怎麼解讀」。
- 量測前先 warmup、用 `torch.cuda.synchronize()` 圍住計時區間，多次取中位數。
- 預設可在 RunPod GPU 上跑（沿用根目錄 `lora-image-gen` 的遠端流程）；Apple Silicon 可作統一記憶體對照。

> 進度：五個 demo（01–05）皆已完成。01／03 已 CPU smoke test；02／04 需 CUDA（CPU 上會優雅提示並結束）；05 已在 Apple M2（MPS）實測。真實記憶體效應一律需在 GPU 上才明顯。
