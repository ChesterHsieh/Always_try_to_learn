# S2 講稿 — GPU 架構與 HBM：資料在晶片內怎麼走

投影片：[../slides/s2_gpu_hbm.pptx](../slides/s2_gpu_hbm.pptx) ｜ Demo：[../demos/02_pinned_vs_pageable](../demos/02_pinned_vs_pageable)

> 前置：S1 的記憶體階層。節奏：理論 ~30 分（slide 1–9）+ demo ~8 分（slide 10）+ 收束 5 分。

## 一句話主旨

把放大鏡轉進晶片內：GPU 靠「大量 warp」藏記憶體延遲；晶片內階層是 HBM → L2 → shared → register；而 **tiling 與 pinned 是把資料「留在/搬到快的地方」的兩招**——tiling 提高算術強度把運算往 roofline 右推，pinned 讓 H2D 又快又能 async。

## 各頁講解重點

**Slide 1–2 — 標題 + 回顧**：S1 講了完整 8 層階層，本場聚焦晶片內 4 層（register / shared+L1 / L2 / HBM）。能「程式控制」的只有 shared 與 register，這是後面 tiling 的伏筆。

**Slide 3 — SM 解剖**：GPU = 一堆 SM；每個 SM 有 CUDA core（通用）、Tensor core（矩陣乘加、衝 AI）、shared memory/L1、register、warp 排程器。

**Slide 4 — warp / SIMT / 藏延遲**：32 threads = 1 warp。一個 SM 同時駐留很多 warp；某 warp 等 HBM（數百 cycle）時排程器切到別的 warp → SM 始終有人在算。這是 S1「GPU 靠同時跑很多藏延遲」的硬體機制。強調：藏延遲的前提是「有夠多 warp 且頻寬餵得上」。

**Slide 5 — 晶片內階層表**：四層的 scope / 頻寬 / 誰管理。重點：shared memory 是「程式可控」的高速暫存——tiling 的槓桿。

**Slide 6 — HBM vs GDDR**：HBM 用 3D 堆疊 + 超寬匯流排換 TB/s 頻寬與能效，但貴；GDDR 較窄匯流排、較高時脈、成本低（消費卡，約 HBM 的 1/3 頻寬）。資料中心吃頻寬 → 用 HBM。

**Slide 7 — shared memory + tiling**：naive matmul 每個輸出重讀整行整列 → 大量 HBM 讀；tiling 把 block 載入 shared memory 一次、在晶片內重複用 → HBM 讀大幅減少。重複用 = 同樣 FLOPs、更少 bytes。

**Slide 8 — tiling → roofline**：把它放回 S1 的 roofline：重複用提高 AI，operating point 從斜線（memory-bound）往右推進平頂（compute-bound）。數字感：tile 邊長 T → HBM 流量約 ÷T。這就是 cuBLAS/tensor core kernel 都重度 tiling 的原因。

**Slide 9 — pinned vs pageable**：H2D 過 PCIe。pageable 要先 staging 到 pinned bounce buffer（多一跳、不能 async）；pinned DMA 直達、可 async overlap。pinned 能 async 這點是 S4 壓軸 overlap 的前提。

**Slide 10 — Demo**：現場跑 `02_pinned_vs_pageable`，看同一條 PCIe pin 不 pin 差約 2×。

**Slide 11 — 收束 + S3**：三句帶走，預告 S3。

## 關鍵推導（撐住現場提問）

### 為什麼需要很多 warp 才藏得住延遲
- 要藏住 L（記憶體延遲，cycle）需要約 `L / 每 warp 指令間隔` 個可切換的 warp 來填空檔（little's law 的直覺）。HBM 延遲數百 cycle → 需要數十個 warp 駐留 → 這就是 GPU 要「大量 thread」的根本原因。

### tiling 為什麼提高 AI
- GEMM `C=A·B`，N×N。naive：每算一個 C 元素讀一行 A + 一列 B；總 HBM 讀 ~ O(N³)。
- tiled（tile 邊長 T）：把 A、B 的 T×T block 載入 shared memory，一個 block 的資料被重複用 T 次 → HBM 讀 ~ O(N³/T)。
- AI = FLOPs / HBM bytes ≈ O(N³) / O(N³/T) = O(T) → tile 越大 AI 越高，受 shared memory 容量限。

### pinned 為什麼快又能 async
- pageable 記憶體 OS 可換頁，DMA 引擎拿到的虛擬位址可能無效 → CUDA 先把資料複製到一塊固定的 pinned bounce buffer 再 DMA。多一次 host 內複製、且整個過程同步。
- pinned（page-locked）頁面鎖住、實體位址固定 → DMA 直接搬，並可 `non_blocking=True` 與 kernel 重疊。

## 預期提問（Q&A 準備）
- **Q：shared memory 多大？** A：每個 SM 數十 ~ 一百多 KB（依世代），且與 L1 共享配置。tile 大小受它限制，所以不能無限放大 T。
- **Q：tensor core 跟 CUDA core 差在哪？** A：tensor core 是專做小矩陣乘加（如 16×16）的單元，吞吐遠高於 CUDA core，但要餵對形狀與精度（fp16/bf16/fp8）；這也是為什麼 kernel 要 tiling 把資料排好餵給它。
- **Q：pinned 記憶體可以無限用嗎？** A：不行。pinned 記憶體鎖住實體頁、不能換出，用太多會擠壓系統可用記憶體；通常只 pin 傳輸用的 buffer。
- **Q：bank conflict 是什麼？** A：shared memory 分成多個 bank，多個 thread 同時存取同一 bank 會序列化。tiling kernel 要排好存取樣式避免它——屬於更深入的優化，這場點到為止。
