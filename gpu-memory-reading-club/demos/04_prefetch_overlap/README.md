# 04 — Prefetch / overlap（S4 壓軸）

**展示概念**：「預先把資料搬到對的地方」能省多少——用第二條 CUDA stream 預取（prefetch）下一批，讓搬運與運算重疊。

> ⚠️ 需要 CUDA GPU 與 pinned 記憶體。CPU 上會直接提示並結束。

## 怎麼跑

```bash
python run.py
python run.py --num 16 --rows 4096 --d 4096 --iters 6
```

## 量到什麼

| 做法 | 說明 |
|---|---|
| `naive` | 搬完這批才算這批；搬與算在同一條 stream 序列化 |
| `overlapped` | copy stream 預取後續批次、compute stream 同時運算 → 搬運被藏在運算後面 |

**預期觀察**：`naive / overlapped ≈` 數成～接近 2×。搬運時間與運算時間越接近，overlap 效益越大（理想上限約 ~2×）；可調 `--iters`（運算量）讓兩者接近。

## 怎麼做到的（重點）

- 用 **pinned host 記憶體** 才能 `non_blocking=True` async 搬。
- 兩條 stream：`copy_stream` 把每批搬進各自的 device buffer（每批一個 buffer，避免 WAR 衝突）並記錄 `Event`；`compute_stream` `wait_event` 該批搬完才算 → 與後續批的搬運重疊。
- 延伸到訓練：`DataLoader(num_workers>0, pin_memory=True)` + 預取，就是同一招在資料管線上的版本。

對應投影片：S4「壓軸概念：prefetch / overlap」。把 S1–S4 收束成一句——**速度的故事，大半是資料在哪、怎麼搬的故事**。
