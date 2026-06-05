# S4 講稿 — 資料搬遷的關卡與記憶體方案（系列終章）

投影片：[../slides/s4_data_movement.pptx](../slides/s4_data_movement.pptx) ｜ Demo：[../demos/04_prefetch_overlap](../demos/04_prefetch_overlap)

> 前置：S1 階層、S2 晶片內。節奏：理論 ~30 分（slide 1–9）+ 壓軸 demo ~8 分（slide 10–11）+ 系列總結 5 分（slide 12）。

## 一句話主旨

把資料送進 GPU 要過幾道「晶片外」的關：PCIe、NVLink、SSD→GPU、各種統一記憶體。每道關都有取捨；最後用 **prefetch / overlap 把搬運藏在運算後面**，收束整個系列——**速度的故事，大半是「資料在哪、怎麼搬」的故事**。

## 各頁講解重點

**Slide 1–2 — 標題 + 全圖**：畫出 SSD → DRAM →[PCIe]→ HBM → L2/SM 全路徑。S1 教判斷瓶頸、S2 講晶片內；本場把「晶片外」這幾段講完。

**Slide 3 — H2D / D2H / PCIe**：host↔device 走 PCIe（Gen4 ~32 / Gen5 ~64 GB/s），比 HBM 慢約 100×。D2H（搬結果回去）也要算。關鍵：減少跨 PCIe 次數，或把它藏起來。

**Slide 4 — NVLink**：GPU↔GPU NVLink ~900 GB/s，比 PCIe 快約一個量級 → 多卡並行（tensor/pipeline parallel）搬權重/啟動值/梯度才划算。NVLink-C2C 也用來連 CPU–GPU（Grace Hopper）。

**Slide 5 — SSD → GPU**：傳統路徑 SSD→CPU bounce buffer→HBM（CPU 介入、兩跳）；GPUDirect Storage 讓 SSD→HBM DMA 直達、繞過 CPU。用途：大資料集/大權重載入、KV cache offload。資料越大、CPU 越是瓶頸。

**Slide 6 — Unified Memory（一）NVIDIA UVM**：單一指標、page fault 觸發頁面遷移、可超額配置、cudaMemPrefetchAsync 預取。好處（程式簡單、能跑超過 HBM）、壞處（fault 開銷、thrashing）。本質仍是「遷移」——資料還是要搬。

**Slide 7 — Unified Memory（二）Apple / Grace Hopper**：Apple 共用同一塊實體記憶體 = 零複製（根本不搬），但頻寬較低（LPDDR）。Grace Hopper 用 NVLink-C2C 硬體一致性連 LPDDR5X + HBM3 = 又大又快。強調三種「unified」機制不同：**遷移 / 零複製 / 一致性互連**。

**Slide 8 — 比較表**：RTX4090 → Grace Hopper 的記憶體型態/容量/頻寬/定位。選型心法：memory-bound 推論看頻寬 + 容量，不是峰值 FLOPS。

**Slide 9 — 心法決策樹**：裝得下 HBM？→ 純 GPU。裝不下但搬得動？→ offload/UVM/GDS（盡量 overlap）。要超大容量？→ 統一記憶體。沒有最好的記憶體，只有最 match 工作集的。

**Slide 10–11 — 壓軸 + Demo**：用第二條 stream 預取下一批、邊搬邊算 → 搬運被藏在運算後面（時間軸圖示「省下的時間」）。現場跑 `04_prefetch_overlap`，看 naive vs overlapped 的差。DataLoader 的 num_workers+pin_memory+prefetch 是同一招。

**Slide 12 — 系列總結**：把 S1–S4 串起來：判斷瓶頸 → 縮短/隱藏搬運 → 選對記憶體。收尾金句。

## 關鍵推導 / 數字（撐住現場提問）

### overlap 的理想上限
- 設搬一批時間 `C`、算一批時間 `K`。naive 總時間 ≈ `N·(C+K)`；完美 overlap ≈ `C + N·max(C,K)`（首批要先搬）。
- 當 `C ≈ K` 時，naive ≈ `2NK`、overlap ≈ `(N+1)K` → 大 N 時加速 ≈ **2×**。
- 若 `C ≪ K`（運算遠多於搬運）overlap 幫助小（本來就不被搬運卡）；`C ≫ K` 則被搬運卡、overlap 仍只能把運算藏進搬運，上限受 `C` 決定。

### 為什麼 GPUDirect Storage 有感
- 傳統路徑資料要進 CPU bounce buffer（一次 SSD→DRAM 的 DMA + 一次 DRAM→HBM 的 DMA + CPU 參與），CPU 與 DRAM 頻寬都可能成瓶頸。
- GDS 讓 NVMe 直接 DMA 到 HBM，省掉 CPU 那一跳——對「大到塞不進一次、要持續串流」的資料集/權重特別有感。

### 頻寬階梯（約略，串起整個系列）
HBM ~TB/s ≫ NVLink ~900 GB/s ≫ PCIe ~32–64 GB/s ≫ DRAM ~100 GB/s ≫ SSD ~7 GB/s。瓶頸＝資料必經的最慢那段（S1 心法）。

## 預期提問（Q&A 準備）
- **Q：統一記憶體是不是就不用煩惱搬運？** A：看機制。Apple 零複製確實不搬（但頻寬低）；NVIDIA UVM 仍會遷移（只是自動化、且可能 thrash）；Grace Hopper 靠高速一致性互連把「搬」變便宜。三者別混為一談。
- **Q：多卡一定要 NVLink 嗎？** A：不一定，但卡間通訊量大的並行（tensor parallel、頻繁 all-reduce）在純 PCIe 上會被卡間頻寬拖死；NVLink/NVSwitch 讓它可行。
- **Q：prefetch overlap 在訓練怎麼落地？** A：`DataLoader(num_workers>0, pin_memory=True)` + 適當 prefetch_factor，讓資料載入與 GPU 運算重疊；再進一步可手動 stream 預取 H2D。
- **Q：KV cache offload 到哪？** A：HBM 不夠時可 offload 到 CPU DRAM 甚至 NVMe（搭配 GDS），代價是每次取回要跨 PCIe——又回到「減少/隱藏搬運」這條主線。
