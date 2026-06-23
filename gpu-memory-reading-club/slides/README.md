# Slides — 各場投影片

每場一份 `.pptx`。風格：深色「矽晶」主題、amber=compute／cyan=memory、一頁一概念、圖優先、數字標清楚單位。

| 檔案 | 場次 | 主題 | 狀態 |
|---|---|---|---|
| `full_series.pptx` | 合輯 | S1–S5 重編去重、**聚焦硬體×Transformer**（已移除 ASR / NVLink-GDS / UVM / 進出站 / CPU-vs-GPU / GPU 解剖、折入心法、方案表移 Part 1） | ✅ 唯一維護版本（34 頁） |
| ~~`s1`–`s5_*.pptx`~~ | S1–S5 | 單場版 | ♻️ 已整併入合輯後刪除；可由 `build/generate_s1.js`–`generate_s5.js` 重建 |

> 合輯第 4 頁是「互動環節」指引頁：講者切出去開 [../interactive/gpu_map.html](../interactive/gpu_map.html)（Cluster → Node → GPU → SM → 運算單元(CUDA/Tensor) 互動下鑽地圖）。次序對照見 [../notes/full_series.md](../notes/full_series.md)。

> 內容大綱見 [../README.md](../README.md) 第 5 節；四場講稿見 [../notes/](../notes/)。

## 如何重建 / 修改投影片

投影片由 `build/` 下的 pptxgenjs 腳本程式化產生（純文字、好版控、易改）：

```bash
cd build
npm install               # 首次：安裝 pptxgenjs
node generate_full.js     # 產生 ../full_series.pptx（主要維護版本）
node generate_s1.js       # （選用）重建單場版 s1–s5，同理 generate_s2..s5.js
```

視覺檢查（需 LibreOffice 的 soffice）：

```bash
cd build
soffice --headless --convert-to pdf --outdir . ../s1_roofline.pptx
pdftoppm -jpeg -r 130 s1_roofline.pdf slide   # 產生 slide-*.jpg 逐頁檢視
```

> `build/node_modules`、產生的 `*.pdf`／`slide-*.jpg` 皆已 gitignore。
