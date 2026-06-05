# Slides — 各場投影片

每場一份 `.pptx`。風格：深色「矽晶」主題、amber=compute／cyan=memory、一頁一概念、圖優先、數字標清楚單位。

| 檔案 | 場次 | 主題 | 狀態 |
|---|---|---|---|
| `s1_roofline.pptx` | S1 | 為什麼會慢？Roofline 與記憶體階層 | ✅ 已產生（11 頁） |
| `s2_gpu_hbm.pptx` | S2 | GPU 架構與 HBM：資料在晶片內怎麼走 | ✅ 已產生（11 頁） |
| `s3_train_vs_infer_asr.pptx` | S3 | Training vs Inference 的瓶頸差異（以 ASR 為例） | ✅ 已產生（13 頁） |
| `s4_data_movement.pptx` | S4 | 資料搬遷的關卡與記憶體方案 | ✅ 已產生（12 頁） |

> 內容大綱見 [../README.md](../README.md) 第 5 節；四場講稿見 [../notes/](../notes/)。

## 如何重建 / 修改投影片

投影片由 `build/` 下的 pptxgenjs 腳本程式化產生（純文字、好版控、易改）：

```bash
cd build
npm install               # 首次：安裝 pptxgenjs
node generate_s1.js       # 產生 ../s1_roofline.pptx
node generate_s2.js       # 產生 ../s2_gpu_hbm.pptx
node generate_s3.js       # 產生 ../s3_train_vs_infer_asr.pptx
node generate_s4.js       # 產生 ../s4_data_movement.pptx
```

視覺檢查（需 LibreOffice 的 soffice）：

```bash
cd build
soffice --headless --convert-to pdf --outdir . ../s1_roofline.pptx
pdftoppm -jpeg -r 130 s1_roofline.pdf slide   # 產生 slide-*.jpg 逐頁檢視
```

> `build/node_modules`、產生的 `*.pdf`／`slide-*.jpg` 皆已 gitignore。
