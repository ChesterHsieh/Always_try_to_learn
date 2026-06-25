# GPU 記憶體與資料搬遷讀書會

**從計算機組織與 GPU 架構，理解為什麼 inference / training 的「資料搬遷」與「記憶體相關速度」差這麼多。**

這是一個面向 data science 背景聽眾的技術讀書會系列。我們不停在「換更貴的卡就會更快」的結論，而是往下挖一層：**資料在硬體裡到底走了哪幾站、每一站的頻寬與延遲差幾個數量級、為什麼同一個任務換個架構速度差十倍。** 最後用幾個簡單、可重現的 demo，量化「預先把資料搬到對的地方（prefetch / staging）」帶來的效能差異。

---

## 1. 讀書會目標

讀完整個系列，聽眾應該能回答：

1. **為什麼會慢？** 一個運算是被 *算力（compute-bound）* 卡住，還是被 *記憶體頻寬（memory-bound）* 卡住？怎麼一眼判斷？
2. **資料走了哪幾站？** 從 SSD → CPU DRAM → PCIe → GPU HBM → L2 → shared memory → register，每一站的頻寬/延遲差幾個數量級？瓶頸通常在哪一段？
3. **training 跟 inference 差在哪？** 為什麼訓練多半 compute-bound、自迴歸推論（autoregressive decode）卻是 memory-bound？KV cache 扮演什麼角色？
4. **以 ASR 為例**：同樣是把語音轉文字，為什麼 attention decoder 架構比 CTC 架構慢？慢在哪一層？
5. **各種 GPU / 記憶體方案怎麼選？** HBM vs GDDR、PCIe vs NVLink、GPUDirect Storage、Unified Memory、Apple 統一記憶體、Grace Hopper——它們在「容量 / 頻寬 / 成本」三角上各站哪裡？
6. **怎麼動手優化？** pinned memory、CUDA stream overlap、prefetch、GPUDirect——預先搬資料實際能省多少？

## 2. 聽眾與前提

- **對象**：data science / ML 背景，會寫 Python、用過 PyTorch，但不一定碰過 CUDA、計算機組織。
- **可以講細**：因為聽眾有 DS 底，roofline、arithmetic intensity、記憶體階層這些可以認真推導，不需要過度簡化。
- **不假設**：不假設聽眾懂 GPU 微架構（SM / warp / tensor core）或 CUDA memory model——這些會從頭建立。
- **每場時長**：預設 60–90 分鐘（理論 40–50 分鐘 + demo / 討論 20–40 分鐘），可依場次合併或拆分。

## 3. 一條主線：兩個心智模型

整個系列只靠兩個模型撐起來，反覆套用到不同硬體與任務上。

### 模型 A — Roofline：你被誰卡住？

定義 **算術強度（Arithmetic Intensity, AI）= 完成運算所需 FLOPs ÷ 需搬動的位元組數（FLOPs/Byte）**。

- AI 高 → 每讀一個 byte 就做很多運算 → **compute-bound**，瓶頸是峰值算力。
- AI 低 → 大部分時間在等資料 → **memory-bound**，瓶頸是記憶體頻寬。
- 分水嶺（ridge point）= 峰值算力 ÷ 峰值頻寬。

> 範例（約略值，以官方規格為準）：H100 SXM ≈ 990 TFLOPS(BF16 tensor) ÷ 3.35 TB/s ≈ **~300 FLOPs/Byte** 才能餵飽算力。許多 inference 運算的 AI 只有個位數 → 注定 memory-bound，換更強的 tensor core 也沒用。

### 模型 B — 記憶體階層：資料離運算單元越遠，越慢一個數量級

| 層級 | 代表頻寬（約略，數量級概念） | 相對延遲 | 備註 |
|---|---|---|---|
| Register | 數十 TB/s | ~1 | 晶片內最快 |
| Shared memory / L1 | 數十 TB/s | ~數十 cycle | 程式可控（tiling 的關鍵） |
| L2 cache | 數 TB/s ~ 數十 TB/s | ~數百 cycle | |
| **HBM（GPU global memory）** | **2 ~ 4.8 TB/s** | ~數百 ns | A100 ~2TB/s、H100 ~3.35TB/s、H200 ~4.8TB/s |
| NVLink（GPU↔GPU / C2C） | ~900 GB/s（NVLink 4） | | 比 PCIe 快一個量級 |
| PCIe（Host↔Device） | Gen4 x16 ~32 GB/s、Gen5 ~64 GB/s | µs 級 | **最常被跨越、也最常見的瓶頸** |
| CPU DRAM（DDR5） | ~50 ~ 100+ GB/s | | |
| NVMe SSD | ~3 ~ 7 GB/s（PCIe Gen4） | µs~ms | 資料集 / 大模型權重來源 |

**一句話心法**：資料搬運的瓶頸＝它必須經過的「最慢那一段路」。HBM 內部很快，但只要被迫跨 PCIe 反覆進出，整體就被 PCIe 拖住——這就是「memory 站一進一出」的代價。

## 4. 課程地圖（4 場系列）

| 場次 | 主題 | 核心問題 | 對應 demo |
|---|---|---|---|
| **S1** | 為什麼會慢？Roofline 與記憶體階層 | compute-bound vs memory-bound 怎麼判斷 | Roofline 實測、latency 數量級 |
| **S2** | GPU 架構與 HBM：資料在晶片內怎麼走 | SM / warp / tensor core / HBM / shared memory | pinned vs pageable 傳輸、tiling 直覺 |
| **S3** | Training vs Inference 的瓶頸差異（以 ASR 為例） | 為什麼 decode 是 memory-bound、KV cache 的角色 | batch sweep decode、ASR encoder/decoder 剖析 |
| **S4** | 資料搬遷的關卡與記憶體方案 | PCIe / NVLink / GPUDirect Storage / Unified Memory | **prefetch / stream overlap 對照（壓軸 demo）** |
| **S5** | 平行運算與軟硬體共同演化（番外進階場） | 為什麼平行度就是一切、模型設計 ⇄ 計算機結構怎麼互相塑造 | FLOPs vs 平行度（LSTM vs Transformer、dense vs depthwise） |

> 彈性：若只辦一場 keynote，可走「S1 心智模型 → S3 ASR 案例 → S4 壓軸 demo」精簡線；完整讀書會則四場循序，S5 可作系列後的進階加場。
> **S1–S5 合輯**（[slides/full_series.pptx](slides/full_series.pptx)，34 頁）是目前唯一維護的投影片，**聚焦「硬體架構 × Transformer」**：重編去重後的單份，五篇章「機器 → 一把尺 → 模型上機 → 資料搬遷 → 共同演化」。相對早期版**已移除 ASR 案例、NVLink/GPUDirect、Unified Memory 三種、進出站(PCIe/pinned)、靜態的「CPU vs GPU」與「GPU 解剖」（GPU 結構改由互動地圖承擔），並把「心法」折進記憶體階層頁**（以下 §4/§5 為原始 S1–S5 場次大綱，屬內容來源；合輯為其聚焦衍生版）。兩個互動環節：第 4 頁搭配 [interactive/gpu_map.html](interactive/gpu_map.html)（Cluster 下鑽到 SM、再到 CUDA/Tensor core）、第 25 頁搭配 [interactive/transformer_map.html](interactive/transformer_map.html)（玩具級 Transformer，6 層 Encoder⟷Decoder 全景→Block→Attention→Head→計算子 matmul(L2⟷HBM)→硬體 × 訓練/Prefill/Decode × GPU/TPU/Groq，含 KV cache 串流與 tensor core tiling，報告見 [notes/transformer_interactive.md](notes/transformer_interactive.md)）。第 8 頁「三層記憶體每 GB 價格」+「各家加速器比較」、TPU/Groq 硬體專頁見第 27–28 頁。次序對照見 [notes/full_series.md](notes/full_series.md)。單場版 pptx 已刪除，可由 `slides/build/generate_sX.js` 重建。

## 5. 各場詳細大綱

### S1 — 為什麼會慢？Roofline 與記憶體階層

- **學習目標**：建立「先問是 compute-bound 還是 memory-bound」的反射動作。
- **大綱**：
  1. 一個熱身謎題：同樣的 GPU，為什麼 batch=1 的 LLM 解碼只用到 <5% 的算力？
  2. CPU vs GPU 設計哲學：latency-oriented（大 cache、亂序執行）vs throughput-oriented（大量 thread 藏延遲）。
  3. Arithmetic Intensity 推導 + Roofline 圖。
  4. 記憶體階層全景（模型 B 的表）＋「latency numbers every DS should know」。
- **關鍵數字**：ridge point 計算、各層頻寬數量級。
- **Demo**：`demos/01_roofline_mini` — 跑不同形狀的矩陣乘法，量到的 TFLOPS 對照 roofline，看到瘦長矩陣掉進 memory-bound 區。

### S2 — GPU 架構與 HBM：資料在晶片內怎麼走

- **學習目標**：知道一個 kernel 從 HBM 取資料、經 L2、進 shared memory、到 register 的路徑，理解 HBM 為何存在。
- **大綱**：
  1. GPU 微架構：SM、warp（32 thread 一組）、CUDA core vs tensor core。
  2. GPU 記憶體階層：global(HBM) / L2 / shared(L1) / register，各自誰能控制。
  3. **HBM vs GDDR**：3D 堆疊、寬匯流排、為什麼資料中心卡用 HBM、成本與功耗代價。
  4. Shared memory 與 tiling：為什麼「把資料留在晶片內重複用」能把 memory-bound 變 compute-bound（矩陣乘法 tiling 的直覺）。
  5. Host↔Device 的橋：pinned（page-locked）vs pageable memory、DMA、為什麼 pageable 要過 bounce buffer。
- **Demo**：`demos/02_pinned_vs_pageable` — 量 pinned vs pageable 的 H2D 頻寬差異（常見 ~1.5–2×）。

### S3 — Training vs Inference 的瓶頸差異（以 ASR 為例）

- **學習目標**：講清楚「為什麼訓練吃算力、自迴歸推論吃頻寬」，並用 ASR 落地。
- **大綱**：
  1. Training：大 batch → 大 GEMM → AI 高 → compute-bound；但 activations / gradients / optimizer states（Adam ≈ 2× 參數量 fp32）造成**容量**壓力。
  2. Inference 兩階段：
     - **Prefill**（吃整段 prompt）→ 大 GEMM → compute-bound。
     - **Decode**（一次一 token、batch 小）→ GEMV、AI≈1–2 → **memory-bound**：每產一個 token 要把整份權重從 HBM 讀一遍。
     - 粗估：7B fp16 ≈ 14 GB，÷ 3.35 TB/s ≈ 每 token ~4 ms 下限 → batch=1 約 ~250 tok/s 天花板。
  3. **KV cache**：避免重算，但隨序列長度變大、每步都要讀 → 同時是頻寬與容量壓力。
  4. **ASR 案例**（本場主軸）：
     - Whisper 類（attention encoder–decoder）：encoder 平行、快；**decoder 自迴歸、逐 token、memory-bound → 延遲主因**。
     - wav2vec2 + CTC：無自迴歸 decode、encoder 一次出結果 → 高度平行、推論快。
     - 結論：**決定速度的不是 FLOPs 總量，而是記憶體存取型態與可平行度**。
- **Demo**：`demos/03_decode_memory_bound` — 小模型 decode 的 batch sweep（batch=1 延遲≈權重/頻寬，加大 batch 幾乎不增加單步延遲 → 吞吐線性上升）；附 ASR encoder vs decoder 時間佔比剖析。

### S4 — 資料搬遷的關卡與記憶體方案（壓軸 demo 場）

- **學習目標**：盤點資料進出 GPU 的每一關，能在「容量/頻寬/成本」上替不同情境選方案。
- **大綱**：
  1. **進出站**：H2D / D2H copy、PCIe 為何是常見瓶頸、NVLink 如何改善多卡。
  2. **SSD → GPU**：傳統路徑（SSD→CPU bounce buffer→HBM，CPU 介入兩跳）vs **GPUDirect Storage**（DMA 直達 HBM、繞過 CPU）；適用大資料集載入、大權重載入、KV cache offload。
  3. **Unified Memory**：
     - CUDA UVM：單一指標、page fault 觸發遷移、可超額配置（oversubscription），`cudaMemPrefetchAsync` 預取藏延遲；存取型態差會 thrash。
     - **Apple 統一記憶體**：CPU/GPU 共用同一塊實體記憶體、**零複製**（與 NVIDIA「遷移式」UVM 本質不同）。
     - **Grace Hopper**：Grace(LPDDR5X) + Hopper(HBM3) 以 NVLink-C2C(~900GB/s) 硬體一致性連接 → 又大又快的統一記憶體。
  4. **GPU / 記憶體方案比較表**（見下節）。
  5. **壓軸 demo**：`demos/04_prefetch_overlap` — 用 CUDA stream 把「搬下一批資料」與「算這一批」重疊（prefetch），對照 naïve 序列版本的吞吐提升；延伸到 DataLoader 的 `num_workers` + `pin_memory` + prefetch 對訓練 step time 的影響。

### S5 — 平行運算與軟硬體共同演化：從 CNN 到 Transformer 到混合架構（番外進階場）

- **學習目標**：理解「GPU 不是快，是寬」；能用三個硬體問題（平行軸 / AI / 序列鏈長）解讀模型架構的演化與取捨。
- **大綱**：
  1. 熱身：GPU 的單執行緒比 CPU 慢——GPU 把電晶體全換成寬度，沒有平行度就沒有 GPU。
  2. 數字感 + Amdahl：H100 ≈ 16,896 條 lane、要 10⁵ 量級 thread 在飛；序列相依的 (1−p) 是加速天花板。
  3. 平行度是模型「暴露」出來的：DL 的平行軸（batch / pixel / channel / token / layer）。
  4. **硬體 → 模型**：CNN 等了 23 年等到 GPU（AlexNet）、hardware lottery；Transformer 的誕生動機就是平行化（取代 RNN 的序列鏈）。
  5. **模型 → 硬體**：tensor core、TPU systolic array、H100 Transformer Engine、H200 141GB、精度 fp32→fp4 的共同演化。
  6. **SOTA case studies**：MobileNet vs ConvNeXt（FLOPs ≠ 速度）；FlashAttention（演算法遷就記憶體階層）與 MQA/GQA（架構遷就頻寬）；Mamba/SSM 與混合架構（Jamba / Griffin / Conformer——呼應 S3 的 ASR）。
  7. 收束框架：設計/選模型前先問三個硬體問題。
- **Demo**：`demos/05_flops_vs_parallelism` — (A) 同規模 LSTM vs Transformer block：FLOPs 多 1.75× 反而快 ~2×；(B) dense vs depthwise conv：FLOPs ÷8.7 但時間只 ÷3.7（Apple M2 實測）。

### 各 GPU / 記憶體方案速覽（約略值，選型用，以官方規格為準）

| 方案 | 記憶體型態 | 容量級距 | 頻寬級距 | 定位 |
|---|---|---|---|---|
| RTX 4090 | GDDR6X | 24 GB | ~1 TB/s | 開發 / 本機實驗 |
| A100 80GB | HBM2e | 80 GB | ~2 TB/s | 訓練 / 推論通用 |
| H100 SXM | HBM3 | 80 GB | ~3.35 TB/s | 訓練 / 推論主力 |
| H200 | HBM3e | 141 GB | ~4.8 TB/s | LLM 推論（吃頻寬+容量） |
| Apple M 系列 | 統一 LPDDR | 可達 128–192 GB+ | ~400–800 GB/s | 本機跑大模型（容量夠、頻寬低 → 慢但跑得動） |
| Grace Hopper GH200 | HBM3 + LPDDR5X | 96 GB HBM + ~480 GB | HBM ~4 TB/s | 超大模型 / 大 KV cache |

> 選型心法：**自迴歸推論（memory-bound）看頻寬與容量，不是峰值 FLOPS。** 這就是為什麼 Apple 統一記憶體「跑得動大模型但慢」、H200/Grace Hopper 主打頻寬與容量。

## 6. Demo 總表

| Demo | 展示的核心概念 | 量測指標 | 預期觀察 |
|---|---|---|---|
| `01_roofline_mini` | compute vs memory bound | 達成 TFLOPS vs 矩陣形狀 | 瘦長矩陣掉進 memory-bound |
| `02_pinned_vs_pageable` | 進出站與 DMA | H2D 頻寬 (GB/s) | pinned 快 ~1.5–2× |
| `03_decode_memory_bound` | training vs inference 瓶頸 | tokens/s vs batch | batch=1 受頻寬限、加 batch 吞吐線性升 |
| `04_prefetch_overlap` | **預先搬資料的效益（壓軸）** | 吞吐 / step time | stream overlap / prefetch 明顯提升 |
| `05_flops_vs_parallelism` | FLOPs ≠ 速度（S5） | GFLOPs / ms / 達成 TFLOPS | LSTM 輸給 FLOPs 更多的 transformer；depthwise 省 FLOPs 不省時間 |

- **工具**：`torch.cuda.Event`（計時）、`torch.profiler`、`nvidia-smi dmon`、Nsight Systems（`nsys`）看時間軸上的 memcpy 停頓。
- **環境**：優先用既有 `lora-image-gen` 的 RunPod GPU 流程跑（見根目錄 README）；本機 Apple Silicon 可跑 Apple 統一記憶體對照組。

## 7. 產出物與資料夾結構

每場交付：**投影片（.pptx）+ 講稿（speaker script）+ demo 程式 + 筆記**。

```
gpu-memory-reading-club/
├── README.md          # 本檔：系列主規劃
├── slides/            # 投影片（pptxgenjs 腳本產生，見 slides/build/）
│   └── full_series.pptx        # S1–S5 合輯（唯一維護版本；單場版可由 build 腳本重建）
├── interactive/       # 互動教具
│   ├── gpu_map.html            # Cluster → Node → GPU → SM → 運算單元(CUDA/Tensor) 互動下鑽地圖（合輯第 4 頁指引開啟）
│   ├── transformer_map.html    # 玩具級 Transformer（T=5、d=6、2 heads）6 層 Encoder⟷Decoder→…→計算子 matmul(L2⟷HBM)→硬體 × 三模式 × GPU/TPU/Groq（KV 串流、tensor core tiling；第 25 頁指引）
│   └── interconnect_map.html   # NVIDIA 互連與通訊協定 6 層下鑽（全景階梯→NVLink/NVSwitch/C2C→PCIe→InfiniBand/Spectrum-X→GPUDirect→CMX）；scale-up/out 視角切換、IB↔乙太分頁、資料流動畫（獨立教具）
├── demos/             # 可重現的 demo 程式與量測腳本
│   ├── 01_roofline_mini/
│   ├── 02_pinned_vs_pageable/
│   ├── 03_decode_memory_bound/
│   ├── 04_prefetch_overlap/
│   └── 05_flops_vs_parallelism/
└── notes/             # 各場深入筆記 / 講稿 / 推導
```

- **目前輸出格式**：以 `.pptx` 與 Markdown（README / notes）為主；講稿可放 `notes/`。
- 投影片走「一頁一概念、圖優先、數字標清楚單位」風格。

## 8. 參考主題（待補來源連結）

> 📖 系列用到的所有縮寫（GEMM / HBM / KV cache / MMA / GQA…）的英文全稱 + 中文 + 一句話說明，見 [notes/glossary.md](notes/glossary.md)。

- Roofline model（Williams et al.）/ Arithmetic intensity
- GPU memory hierarchy 與 CUDA best practices（pinned memory、stream、tiling）
- LLM inference 的 memory-bound 本質、KV cache、prefill vs decode
- ASR 架構對照：Whisper（attention decoder）vs wav2vec2/Conformer + CTC
- GPUDirect Storage、CUDA Unified Memory、Apple 統一記憶體、Grace Hopper 架構
- NVIDIA 互連與通訊協定：NVLink / NVSwitch / NVLink-C2C（scale-up）、PCIe、InfiniBand（Quantum）vs Spectrum-X 乙太網（RoCE）、GPUDirect RDMA、NCCL
- NVIDIA CMX（Context Memory，2026）：KV cache 卸載到 G3.5 層（BlueField-4 DPU + Spectrum-X flash）、DOCA Memos / Dynamo / NIXL、長上下文與 agentic 推論
- The Hardware Lottery（Sara Hooker）／Attention is All You Need 的平行化動機
- FlashAttention（IO-aware exact attention）、MQA/GQA、Mamba/SSM 與混合架構（Jamba、Griffin、Conformer）

## 9. 里程碑與下一步

- [x] **M0**：系列主規劃（本檔）
- [x] **M1**：S1 roofline demo + 講稿（投影片已整併入合輯，可由 `slides/build/generate_s1.js` 重建）
  - Demo [demos/01_roofline_mini](demos/01_roofline_mini)（已 CPU smoke test）｜講稿見合輯 [notes/full_series.md](notes/full_series.md)（Part 1–2；單場講稿已併入後刪除）
- [x] **M2**：S2 pinned vs pageable demo + 講稿（投影片已整併，`generate_s2.js` 可重建）
  - Demo [demos/02_pinned_vs_pageable](demos/02_pinned_vs_pageable)（需 GPU）｜講稿見合輯 [notes/full_series.md](notes/full_series.md)（Part 1–2）
- [x] **M3**：S3 decode/ASR demo + 講稿（核心場；投影片已整併，`generate_s3.js` 可重建）
  - Demo [demos/03_decode_memory_bound](demos/03_decode_memory_bound)（batch sweep + ASR proxy，已 CPU smoke test）｜講稿見合輯 [notes/full_series.md](notes/full_series.md)（Part 3）
- [x] **M4**：S4 prefetch 壓軸 demo + 講稿（投影片已整併，`generate_s4.js` 可重建）
  - Demo [demos/04_prefetch_overlap](demos/04_prefetch_overlap)（需 GPU）｜講稿見合輯 [notes/full_series.md](notes/full_series.md)（Part 4）
- [x] **M5**：S5 FLOPs vs 平行度 demo + 講稿（番外進階場；投影片已整併，`generate_s5.js` 可重建）
  - Demo [demos/05_flops_vs_parallelism](demos/05_flops_vs_parallelism)（已在 Apple M2 / MPS 實測）｜講稿見合輯 [notes/full_series.md](notes/full_series.md)（Part 5）
- [x] **M6**：S1–S5 合輯 + 互動地圖
  - 合輯 [slides/full_series.pptx](slides/full_series.pptx)（34 頁，聚焦硬體×Transformer；次序對照 [notes/full_series.md](notes/full_series.md)）
  - 互動地圖 [interactive/gpu_map.html](interactive/gpu_map.html)（Cluster → Node → GPU → SM → 運算單元(CUDA/Tensor) 下鑽，合輯第 4 頁指引開啟）
- [x] **M7**：玩具級 Transformer 互動地圖 + TPU 專頁
  - 互動地圖 [interactive/transformer_map.html](interactive/transformer_map.html)（T=5、d=6、2 heads；六層 Encoder⟷Decoder 全景 → Block → Attention → Head → 計算子 matmul(L2⟷HBM 搬運) → 硬體；訓練/Prefill/Decode 三模式；GPU/TPU/Groq 切換含脈動陣列動畫）
  - 報告 [notes/transformer_interactive.md](notes/transformer_interactive.md)；合輯第 25 頁（互動環節②）、第 27 頁（TPU）、第 28 頁（Groq）、第 8 頁（每 GB 價格）
- [x] **M8**：NVIDIA 互連與通訊協定互動地圖（含 2026 CMX）
  - 互動地圖 [interactive/interconnect_map.html](interactive/interconnect_map.html)（六層下鑽：全景階梯 → NVLink/NVSwitch/C2C（scale-up）→ PCIe → InfiniBand/Spectrum-X（scale-out）→ GPUDirect → CMX；scale-up/scale-out 視角切換、IB↔乙太分頁、資料流動畫）
  - 把合輯為聚焦而移除的 NVLink/GPUDirect/網路等互連主題以獨立教具補回，並新增 NVIDIA 2026 的 **CMX（Context Memory）**：KV cache 卸載到 G3.5 層（BlueField-4 + Spectrum-X flash），呼應 Part 3 decode memory-bound 與 Part 4 prefetch。詞條見 [notes/glossary.md](notes/glossary.md) §12

> 🎉 全系列內容已整併為單份合輯 + 三張互動地圖。後續可選：把 demo 在 RunPod GPU 上實跑、補真實數據回填投影片的「示意」表格（demo 05 已有 M2/MPS 實測數據）；把 interconnect_map / CMX 接進合輯投影片（目前為獨立教具）。
