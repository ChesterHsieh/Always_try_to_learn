# Notes — 各場深入筆記與講稿

放系列的講稿（speaker script）、推導細節、延伸閱讀整理。投影片講重點，這裡放「為什麼」與數字來源。

系列已從「五個單場」更迭到「一份聚焦合輯」。原 `s1`–`s5` 單場講稿（及其 pptx）已併入合輯後刪除；現存三份筆記：

- `full_series.md` — ✅ 合輯索引（主檔）：S1–S5 重編去重成單份投影片（37 頁）的次序地圖、聚焦「硬體架構 × Transformer」、各頁整併/移除說明、精簡 keynote 頁碼建議，並含撐場用的關鍵推導與 Q&A 速查。
- `transformer_interactive.md` — ✅ 報告：玩具級 Transformer 互動地圖（T=5、d=6、2 heads）— 結構、訓練/Prefill/Decode 三模式資料流 × GPU 結構、TPU systolic array、Groq LPU、展示動線。
- `glossary.md` — ✅ 術語與縮寫對照表：全系列縮寫（GEMM/HBM/KV cache/MMA…）的英文全稱 + 中文 + 一句話說明，分 11 類。

> 大綱見 [../README.md](../README.md) 第 5 節。講解細節與推導以 [full_series.md](full_series.md) 為準。


$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$
