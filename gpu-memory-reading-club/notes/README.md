# Notes — 各場深入筆記與講稿

放系列的講稿（speaker script）、推導細節、延伸閱讀整理。投影片講重點，這裡放「為什麼」與數字來源。

系列已從「五個單場」更迭到「合輯 + 各堂獨立場」。原 `s1`–`s5` 單場講稿（及其 pptx）已併入合輯後刪除；五堂投影片與講稿均已齊備（對應關係見 [../README.md](../README.md) §4.0）。現存筆記：

- `full_series.md` — ✅ **第一堂課**／合輯索引（主檔）：S1–S5 重編去重成單份投影片（34 頁）的次序地圖、聚焦「硬體架構 × Transformer」、各頁整併/移除說明、精簡 keynote 頁碼建議，並含撐場用的關鍵推導與 Q&A 速查。
- `class2_transformer_gpu.md` — ✅ **第二堂課講稿／索引**：Transformer 逐 block × GPU（單卡）→ 多卡的資料平行問題與 NVIDIA 互連新技術（NVLink5/NVSwitch/NVL72、SHARP/NCCL、IB/Spectrum-X/GPUDirect、Rubin/NVLink6、CMX 2026）。含 24 頁頁面地圖、逐 block 速查、四種平行對照、互連數字與來源、關鍵推導與 Q&A。搭配 [../interactive/parallelism_map.html](../interactive/parallelism_map.html)。
- `class3_engine_single_node.md` — ✅ **第三堂課講稿／索引**：推論引擎單機篇，**八個問題 × 兩種解法（SGLang × vLLM）**。那八個問題不是某一家專有的，是任何推論引擎都會撞到的，所以每題都對比兩家的寫法：①程式難平行（SGLang 前端 DSL vs vLLM 沒有對應物）｜②前綴重算（共同：分頁 KV／索引：radix tree vs 鏈式雜湊表／排程：cache-aware 主動 vs 被動）｜③輸出不可控（共同：XGrammar FSM／差異：jump-forward 約 3× 吞吐）｜④CPU 成瓶頸（zero-overhead scheduler vs 多進程+async）｜天花板：投機解碼·MTP、量化。含 28 頁頁面地圖、哲學差異、「四問題 × 兩種寫法」總表、選型判準、Q&A。搭配 [../interactive/serving_map.html](../interactive/serving_map.html)（模式 1–4）。
- `class4_sglang_multi_node.md` — ✅ **第四堂課講稿／索引**：SGLang 多機篇，延續同一條主幹的問題 ⑤–⑧（大規模 EP／PD 分離／cache-aware router／容錯）。**開場框架是用經典分散式系統的八類共同問題當影子**，逐格對照 GPU 叢集（哪些被規避、哪些變形、哪些被放大），並指出推論服務比訓練更靠近經典分散式系統那一端。含壓軸推導「KV 該搬還是該重算」與「三個換了名字的老問題」。含 20 頁頁面地圖。互動教具 `router_map.html` 待做。
- `class5_china_models.md` — ✅ **第五堂課講稿／索引**：中國開源模型把推論成本寫進架構。五個旋鈕（壓 KV／少算／少看／一次多產／降精度）× 五個實驗室（DeepSeek／Kimi／MiniMax／Qwen／GLM），含 MiniMax M1→M2→M3 反例（No Free Lunch）與「為什麼他們連 kernel 都開源」。搭配 16 頁投影片。
- `transformer_interactive.md` — ✅ 報告：玩具級 Transformer 互動地圖（T=5、d=6、2 heads）— 結構、訓練/Prefill/Decode 三模式資料流 × GPU 結構、TPU systolic array、Groq LPU、展示動線。
- `glossary.md` — ✅ 術語與縮寫對照表：全系列縮寫（GEMM/HBM/KV cache/MMA…）的英文全稱 + 中文 + 一句話說明，分 12 類。

> 大綱見 [../README.md](../README.md) 第 5 節。講解細節與推導以 [full_series.md](full_series.md) 為準。


