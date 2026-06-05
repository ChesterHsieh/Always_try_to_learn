# 02 — Pinned vs pageable H2D 頻寬（S2）

**展示概念**：主機 → GPU 的傳輸，記憶體是不是 pinned（page-locked）差很多。

> ⚠️ 需要 CUDA GPU（pinned 記憶體與 H2D 傳輸皆需 CUDA）。CPU 上會直接提示並結束。

## 怎麼跑

```bash
python run.py
python run.py --sizes-mb 1,4,16,64,256 --repeats 30
```

## 量到什麼

| 欄位 | 意義 |
|---|---|
| `pageable` | 一般 host 記憶體的 H2D 頻寬（要經 pinned bounce buffer） |
| `pinned` | page-locked 記憶體的 H2D 頻寬（DMA 直達） |
| `speedup` | pinned / pageable |

**預期觀察**：pinned 通常快 **~1.5–2×**，且大尺寸時頻寬會逼近 PCIe 上限（Gen4 ~32 GB/s、Gen5 ~64 GB/s）。

## 為什麼

- **pageable**：OS 可把該頁換出，DMA 引擎不能直接搬 → CUDA 先把資料 staging 到一塊 pinned bounce buffer 再 DMA → 多一次複製、且不能 async。
- **pinned**：頁面鎖住、實體位址固定 → DMA 直接搬，還能 `non_blocking=True` 與運算重疊。

對應投影片：S2「進出晶片的橋：pinned vs pageable」。pinned 能 async 這點，是 S4 壓軸 overlap demo 的前提。
