# S3 講稿 — Training vs Inference 的瓶頸差異（以 ASR 為例）

投影片：[../slides/s3_train_vs_infer_asr.pptx](../slides/s3_train_vs_infer_asr.pptx) ｜ Demo：[../demos/03_decode_memory_bound](../demos/03_decode_memory_bound)

> 這是系列核心場。前置：S1 的 roofline／算術強度。節奏建議：理論 ~30 分（slide 1–11）＋ demo ~10 分（slide 12）＋ 收束 5 分。

## 一句話主旨

**訓練多半 compute-bound、自迴歸 decode 卻是 memory-bound**——因為 decode 每產一個 token 都要把整份權重從 HBM 讀一遍。用 ASR 把這件事落地：同一個轉錄任務，attention decoder（Whisper）比 CTC（wav2vec2）慢，慢在那段序列化的 decode。

## 各頁講解重點

**Slide 1–2 — 標題 + 回顧**：先把 S1 的尺帶回來（compute-bound vs memory-bound、AI）。本場全程用這把尺量訓練與推論。

**Slide 3 — Training**：大 batch → 大 GEMM → AI 高 → compute-bound，利用率高。但要同時放下：權重 / 啟動值（留著等 backward）/ 梯度 / optimizer states。強調：**訓練的痛點常是「裝不下」(容量)**，這解釋了為什麼有梯度檢查點、ZeRO、offload。

**Slide 4 — Inference 兩階段**：prefill（吃整段 prompt、大 GEMM、compute-bound）vs decode（逐 token、GEMV、memory-bound）。今天聚焦 decode。

**Slide 5 — decode 為何 memory-bound（核心）**：每 token 重讀整份權重。7B fp16 = 14 GB ÷ 3.35 TB/s ≈ 4.2 ms/token → batch=1 約 ~240 tok/s 上限。**直接連回 S1 的「<5% 利用率」**：時間都在搬權重。

**Slide 6 — KV cache**：救了重算（不必每步對整段序列重算 attention），但隨序列長度線性增長、每步都要讀（頻寬）又要存住（容量）。這是長 context 變慢、batch 受限的主因。

**Slide 7 — Batch 的魔法**：batch=1 讀一次權重只服務一個請求＝浪費；batch=N 同一次讀取服務 N 個 → 吞吐線性升、**單步延遲幾乎不變**。直到 compute-bound 或 KV cache 裝不下。強調 throughput ↑ 不等於 latency ↓。

**Slide 8 — 換場到 ASR**：ASR＝語音轉文字，DS 熟悉。同一任務、兩種架構、速度差很多 → 最適合示範「不是 FLOPs 決定速度」。

**Slide 9 — Whisper（attention）**：encoder 吃整段 spectrogram、平行、compute-bound、快；decoder 自迴歸逐 token → memory-bound → **延遲主因**。

**Slide 10 — CTC（wav2vec2）**：encoder only，一次輸出所有 frame 的字元機率、CTC 對齊，**無自迴歸 decode** → 高度平行、推論快、適合串流。

**Slide 11 — 對照總結**：把兩者擺一起。結論句：**決定速度的是記憶體存取型態與可平行度，不是 FLOPs 總量。**

**Slide 12 — Demo**：現場跑 `run.py`（batch sweep）看吞吐曲線、`asr_proxy.py`（平行 vs 序列）看延遲差。提醒效應需 GPU。

**Slide 13 — 收束 + S4**：三句帶走，預告 S4 把「搬權重／搬資料」的關卡（PCIe/NVLink/GPUDirect/Unified Memory）講完。

## 關鍵推導（撐住現場提問）

### decode 是 memory-bound
- 每步要讀的位元組 ≈ 權重（＋當前 KV cache）；做的 FLOPs ≈ `2 × params × batch`。
- AI ≈ `2 × params × batch / (2 × params)` ＝ **batch**（fp16）。batch 小 → AI 小 → memory-bound。
- batch=1 單步延遲下限 ＝ `權重位元組 / HBM 頻寬`。7B fp16：`14e9 / 3.35e12 ≈ 4.2 ms` → ~240 tok/s。

### KV cache 大小
`≈ 2 × layers × heads × head_dim × seq_len × batch × dtype_bytes`
（2＝K 與 V）。隨 `seq_len × batch` 線性長大；每個 decode step 都要讀它，所以長 context／大 batch 會把 decode 更往 memory-bound 推、也吃滿容量。

### batch 攤平的數學
- 單步延遲 `≈ max(權重/頻寬, 2·params·batch/算力)`。
- batch 小：被 `權重/頻寬`（常數）主宰 → step 幾乎不變、tokens/s ＝ batch/step 線性升。
- batch 跨過 ridge（≈ 算力·權重/(頻寬·2·params) ≈ 數百）：第二項接手 → 轉 compute-bound、tokens/s 趨平。

### ASR：為什麼架構決定速度
- 兩者 encoder 都平行（compute-bound）。差別在 Whisper 多了**自迴歸 decoder**：T 個輸出 token 必須一個一個來，每步是小 GEMV、重讀權重、又吃 kernel launch 開銷 → 即使總 FLOPs 不比 CTC 多，wall-clock 卻高出數倍。
- CTC 無此序列段 → 一次輸出所有 frame、對齊成字。這就是 demo `asr_proxy.py` 想讓你看到的「平行 vs 序列」差。

## 預期提問（Q&A 準備）
- **Q：那 decode 為什麼不也用大 batch？** A：server 端會（continuous batching）。但單一使用者的請求湊不到大 batch，且 KV cache 容量會先撐爆 → 這就是 LLM 服務的核心工程問題。
- **Q：Whisper 不是也很準？為何還要 CTC？** A：取捨。Whisper 準、能做語言模型式的生成與多語；CTC 低延遲、適合即時串流。速度差來自架構，不是誰比較「強」。
- **Q：speculative decoding／量化算不算解法？** A：算。投機解碼用小模型先猜、大模型一次驗多個 token，攤平權重讀取；量化（int8/fp8）直接減少要搬的位元組——兩者都是針對 memory-bound 下手，不是加算力。
- **Q：prefill 也會慢嗎？** A：prefill 是 compute-bound、相對好攤；長 prompt 的痛點通常在 KV cache 容量與 prefill 的 attention 複雜度，不是頻寬。
