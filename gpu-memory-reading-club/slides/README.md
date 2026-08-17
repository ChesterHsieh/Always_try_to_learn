# Slides — 各場投影片

每場一份 `.pptx`。風格：深色「矽晶」主題、amber=compute／cyan=memory、一頁一概念、圖優先、數字標清楚單位。

| 檔案 | 場次 | 主題 | 狀態 |
|---|---|---|---|
| `full_series.pptx` | 第一堂課／合輯 | S1–S5 重編去重、**聚焦硬體×Transformer**（已移除 ASR / NVLink-GDS / UVM / 進出站 / CPU-vs-GPU / GPU 解剖、折入心法、方案表移 Part 1） | ✅ 維護中（34 頁） |
| `class2_transformer_gpu.pptx` | 第二堂課 | **Transformer × GPU 框架**：Part A 玩具 Transformer 逐 block × GPU 單元（單卡）｜Part B 多卡的資料平行問題 + NVIDIA 互連新技術（NVLink5/NVSwitch/NVL72、SHARP/NCCL、IB/Spectrum-X、Rubin/NVLink6、CMX 2026） | ✅ 維護中（24 頁） |
| `class3_sglang_single_node.pptx` | 第三堂課 | **SGLang 單機篇**：沿著 SGLang 遇到的問題走——①程式難平行 → 前端 DSL｜②前綴重算 → 分頁 KV + RadixAttention + continuous batching｜③輸出不可控 → Compressed FSM｜④CPU 成瓶頸 → zero-overhead scheduler / CUDA Graph / chunked prefill｜天花板：投機解碼·MTP、量化 | ✅ 維護中（24 頁） |
| `class4_sglang_multi_node.pptx` | 第四堂課 | **SGLang 多機篇**：Part A 用經典分散式系統的八類共同問題當影子對照 GPU 叢集｜⑤大 MoE → 大規模 EP（含資料傾斜/EPLB）｜⑥P/D 互擾 → PD 分離｜⑦多機調度 → cache-aware router（壓軸推導：該搬還是該重算）｜⑧容錯（KV 是可重算的快取 + 四個開放問題）｜橫向比較四家架構取向 | ✅ 維護中（20 頁） |
| `class5_china_models.pptx` | 第五堂課 | **中國開源模型的五個旋鈕**：壓 KV／少算／少看／一次多產／降精度 × DeepSeek／Kimi／MiniMax／Qwen／GLM。含 MiniMax 反例兩頁與「為什麼他們連 kernel 都開源」 | ✅ 維護中（16 頁） |
| ~~`s1`–`s5_*.pptx`~~ | S1–S5 | 單場版 | ♻️ 已整併入合輯後刪除；可由 `build/generate_s1.js`–`generate_s5.js` 重建 |

> 第二堂課第 23 頁是「互動環節③」指引頁：講者切出去開 [../interactive/parallelism_map.html](../interactive/parallelism_map.html)（單卡 → 裝不下 → DP → TP → PP → 互連硬體；TP/PP 兩層把「一層」的權重矩陣怎麼被切畫出來）。講稿見 [../notes/class2_transformer_gpu.md](../notes/class2_transformer_gpu.md)。

> 第三堂課第 14 頁是「互動環節」指引頁：講者切出去開 [../interactive/serving_map.html](../interactive/serving_map.html)（數字鍵 1–5 切換 Naive / Continuous batching / Paged KV / Radix 前綴共用 / PD 分離，右側面板同步顯示有效 batch、KV 浪費比例、算術強度與 roofline 上的位置）。**第三堂只用模式 1–4，模式 5（PD 分離）留給第四堂。** 講稿見 [../notes/class3_sglang_single_node.md](../notes/class3_sglang_single_node.md)。

> 第四堂課可搭配 [../interactive/serving_map.html](../interactive/serving_map.html) 的**模式 5（PD 分離）**——第三堂只用到模式 1–4。講稿見 [../notes/class4_sglang_multi_node.md](../notes/class4_sglang_multi_node.md)；規劃中的 `router_map.html`（三種路由策略對照）尚未製作。

> 合輯第 4 頁是「互動環節」指引頁：講者切出去開 [../interactive/gpu_map.html](../interactive/gpu_map.html)（Cluster → Node → GPU → SM → 運算單元(CUDA/Tensor) 互動下鑽地圖）。次序對照見 [../notes/full_series.md](../notes/full_series.md)。

> 內容大綱見 [../README.md](../README.md) §4.0；各堂講稿見 [../notes/](../notes/)。

## 如何重建 / 修改投影片

投影片由 `build/` 下的 pptxgenjs 腳本程式化產生（純文字、好版控、易改）：

```bash
cd build
npm install               # 首次：安裝 pptxgenjs
node generate_full.js     # 產生 ../full_series.pptx（S1–S5 合輯）
node generate_class2.js   # 產生 ../class2_transformer_gpu.pptx（第二堂課：Transformer × GPU）
node generate_class3.js   # 產生 ../class3_sglang_single_node.pptx（第三堂課：SGLang 單機篇）
node generate_class4.js   # 產生 ../class4_sglang_multi_node.pptx（第四堂課：SGLang 多機篇）
node generate_class5.js   # 產生 ../class5_china_models.pptx（第五堂課：中國開源模型）
node generate_s1.js       # （選用）重建單場版 s1–s5，同理 generate_s2..s5.js
```

視覺檢查（需 LibreOffice 的 soffice）：

```bash
cd build
soffice --headless --convert-to pdf --outdir . ../s1_roofline.pptx
pdftoppm -jpeg -r 130 s1_roofline.pdf slide   # 產生 slide-*.jpg 逐頁檢視
```

> `build/node_modules`、產生的 `*.pdf`／`slide-*.jpg` 皆已 gitignore。
