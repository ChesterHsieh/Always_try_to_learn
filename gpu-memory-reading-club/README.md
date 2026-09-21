# GPU 記憶體與資料搬遷讀書會

**網站入口：<https://chesterhsieh.github.io/Always_try_to_learn/gpu-memory-reading-club/>**

六堂課，給會寫 PyTorch、但沒碰過 CUDA 與計算機組織的 data science／ML 工程師。從一張 GPU 的記憶體階層出發，一路走到上百張卡的推論叢集。投影片、講稿、互動教具和複習測驗全部是網頁，從首頁 [index.html](index.html) 進入。

## 故事線

**同一個敵人，六個高度：decode 是 memory-bound，每產一個字，都要把整份權重和 KV 從記憶體搬一遍。**

整個系列只用兩個心智模型：

- **Roofline**：算術強度 = FLOPs ÷ 搬動的 Bytes。低於 ridge point（H100 約 300 FLOPs/Byte）就是 memory-bound，換更強的算力也沒用。
- **記憶體階層**：每離運算單元遠一站就慢一個數量級。瓶頸是資料必經的最慢那段路。

| 堂 | 主題 | 接住上一堂的什麼、留下什麼 | 教材 |
|---|---|---|---|
| 1 | 硬體 × Transformer | 用 roofline 和記憶體階層解開謎題：H100 在 batch=1 解碼時，算力利用率為什麼不到 5%。留下的問題：模型大到一張卡裝不下怎麼辦？ | [投影片](slides/full_series.html)・[講稿](notes/full_series.md) |
| 2 | Transformer × GPU：一張卡到多張卡 | 玩具 Transformer 逐 block 對到 GPU 單元；裝不下之後只能切模型（TP／PP／EP），切法可不可行由互連決定。留下的問題：decode 要把 batch 拉到幾百才吃得滿算力。 | [投影片](slides/class2_transformer_gpu.html)・[講稿](notes/class2_transformer_gpu.md) |
| 3 | 推論引擎單機篇：SGLang × vLLM | 退回一台機器，看引擎怎麼把 batch 從 1 撐到幾百。問題①–④：程式難平行、前綴重算、輸出不可控、CPU 成瓶頸，每題對照兩家寫法。 | [投影片](slides/class3_engine_single_node.html)・[講稿](notes/class3_engine_single_node.md) |
| 4 | SGLang 多機篇 | 用經典分散式系統的八類問題當影子，解問題⑤–⑧：大規模 EP、PD 分離、cache-aware router、容錯。壓軸是「KV 該搬還是該重算」。 | [投影片](slides/class4_sglang_multi_node.html)・[講稿](notes/class4_sglang_multi_node.md) |
| 5 | 中國開源模型的五個旋鈕 | 框架從外面調到頭了，改成模型從裡面改：壓 KV、少算、少看、一次多產、降精度。MiniMax 是「理論更省、實際不一定更快」的反例。 | [投影片](slides/class5_china_models.html)・[講稿](notes/class5_china_models.md) |
| 6 | 最終章：一個字，穿過一整排機櫃 | 跟著一個請求穿過 SGLang 96×H100 叢集，按「多常搬 × 一次多大」把資料分四種。真瓶頸是一個 scale-up 域裝得下幾張卡，由此推 NVIDIA 與 AMD 的選型。 | [投影片](slides/class6_multi_rack_inference.html)・[講稿](notes/class6_multi_rack_inference.md) |

一句話版：硬體的尺 → 單卡到多卡 → 單機引擎 → 多機服務 → 模型架構 → 裝回機櫃。

每堂 8 題的[複習測驗](quiz/index.html)，難度依序是辨識 → 邊界 → 遷移 → 取捨。答錯先給針對你所選選項的提示，再錯才公布答案，最後產生一段可以貼給 Claude 繼續深挖的 prompt。

## 資料夾

```
gpu-memory-reading-club/
├── index.html        # 網站首頁（GitHub Pages 單一入口）
├── slides/           # 六份 HTML 投影片（產生物，不要手改）
│   └── build/        # 投影片原始碼：generate_*.js + pptx-html.js 轉接層 + deck-template.html
├── notes/            # 各堂講稿（Markdown）、術語表 glossary.md；view.html 是講稿閱讀器
├── interactive/      # 六個互動教具（單檔 HTML）
├── quiz/             # 複習測驗：index.html + questions.js 題庫
├── demos/            # 五支可重現的 PyTorch demo
├── references/       # 外部參考資料
└── assets/site.css   # 首頁、講稿、測驗共用樣式
```

## 修改投影片

投影片內容寫在 `slides/build/generate_*.js`（沿用 pptxgenjs 的 API：`addText`／`addShape`／`addTable`／`addChart`，座標單位是英吋）。`pptx-html.js` 把這些呼叫轉成等比縮放的 HTML，不需要安裝任何套件：

```bash
cd slides/build
node generate_full.js      # → ../full_series.html（第一堂），其餘 generate_class2..6.js 同理
```

版面是固定尺寸的文字框，改字時注意長度，改完在瀏覽器看一眼有沒有溢出。投影片頁按 `F` 進簡報模式，網址加 `#s12` 可直接跳頁。

講稿是 Markdown，直接改 `notes/*.md`；網站上由 `notes/view.html?doc=<檔名>` 在瀏覽器端渲染。本機預覽要起一個伺服器（`fetch` 不能讀 `file://`）：

```bash
python3 -m http.server 8000   # 在本資料夾執行，然後開 http://localhost:8000/
```

## Demo

需要 PyTorch。記憶體效應要在 CUDA GPU 上才明顯；計時一律 warmup → `torch.cuda.synchronize()` 圍住 → 取中位數。

| Demo | 對應 | 怎麼跑 | 預期看到 |
|---|---|---|---|
| `01_roofline_mini` | 第一堂 roofline | `python run.py --peak-tflops 990 --peak-bw 3.35` | GEMV、瘦長矩陣的 AI ≈ 1–2，落在 memory-bound 斜線；方陣隨邊長逼近峰值 |
| `02_pinned_vs_pageable` | 第一堂 資料進出 GPU | `python run.py`（需 CUDA） | pinned 比 pageable 快約 1.5–2×，大尺寸逼近 PCIe 上限 |
| `03_decode_memory_bound` | 第一堂 decode | `python run.py --peak-bw 3.35`；`python asr_proxy.py` | batch 小時單步延遲幾乎不變、吞吐隨 batch 線性上升；同 FLOPs 下序列 decoder 比平行 encoder 慢數倍 |
| `04_prefetch_overlap` | 第一堂 prefetch 壓軸 | `python run.py`（需 CUDA） | 兩條 stream 把搬運藏在運算後面，理想上限約 2× |
| `05_flops_vs_parallelism` | 第一堂 共同演化 | `python run.py`（cuda > mps > cpu） | Apple M2 實測：Transformer FLOPs 多 1.75× 卻快 1.9×；depthwise FLOPs ÷8.7 但時間只 ÷3.7 |

每支 `run.py` 都有 `--help`，參數說明寫在檔頭註解。

## 待辦

- CMX 的能效，第二堂寫約 4×、術語表寫約 5×，開講前以 NVIDIA 官方頁統一。
- 第四堂的 `router_map.html`（三種路由策略下各機的命中率與負載）尚未製作。
- demo 01–04 在 RunPod GPU 上實跑，把實測數字回填投影片的示意表格。
