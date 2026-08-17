# 第三堂課 講稿／索引 — SGLang 單機篇：沿著 SGLang 遇到的問題走（問題 ①–④）

投影片：[../slides/class3_sglang_single_node.pptx](../slides/class3_sglang_single_node.pptx)（24 頁）｜重建：`cd ../slides/build && node generate_class3.js`
互動地圖：[serving_map.html](../interactive/serving_map.html)（第 14 頁指引，**本堂只用模式 1–4**，模式 5 PD 分離留給第四堂）

> **主幹＝SGLang 一路上遇到的問題，不是功能表。** 每個機制都是被一個具體痛點逼出來的——先有痛點，機制才記得住。
> 八個問題由淺入深剛好走完「編程模型 → 單機記憶體 → 單機排程 → 多卡 → 多機」，**本堂做 ①–④（單機），⑤–⑧ 是[第四堂](class4_sglang_multi_node.md)（多機）**。
> 前置：第一堂的 roofline 與記憶體階層、第二堂的多卡平行與互連。

---

## 八個問題全景（第 2 頁，全系列的骨架）

| # | 遇到的問題 | SGLang 的解法 | 層級 | 場次 |
|---|---|---|---|---|
| ① | LLM 程式（多次呼叫、分支、工具）難寫又跑不快 | **前端 DSL**：把程式描述成執行圖 | 編程模型 | 本堂 |
| ② | 這些程式天然共享大量前綴，卻被反覆重算 | **RadixAttention**：前綴樹 + LRU + cache-aware 排程 | 單機記憶體 | 本堂 |
| ③ | 結構化輸出逐 token 檢查語法太慢、格式仍不保證 | **Compressed FSM**：預編譯 + 位元遮罩 | 單機生成 | 本堂 |
| ④ | GPU 一步只要 5–10 ms，CPU 排程反而成了瓶頸 | **zero-overhead scheduler** / CUDA Graph | 單機排程 | 本堂 |
| ⑤ | 大 MoE（DeepSeek）單機放不下、專家負載不均 | 大規模 EP + DeepEP / EPLB | 多卡 | 第四堂 |
| ⑥ | prefill 與 decode 互相干擾（TTFT vs ITL） | PD 分離 | 多卡 | 第四堂 |
| ⑦ | 多副本之間：快取局部性 vs 負載均衡此消彼長 | cache-aware router + KV 複製 | 多機 | 第四堂 |
| ⑧ | 副本掛掉，進行中的請求與它的 KV 怎麼辦 | 容錯 | 多機 | 第四堂 |

> **講法**：第 2 頁先把八個問題全攤出來，讓聽眾看到「這是一條線，不是一堆功能」。然後說「今天走前四個」。

---

## 頁面地圖（24 頁）

| 頁 | 內容 |
|---|---|
| 1 | 標題：一個推論引擎，是被哪些問題逼出來的？ |
| 2 | **全景**：八個問題 × 解法 × 層級，線以上本堂、線以下第四堂 |
| **地基** | |
| 3 | 地基①：batch=1 的 decode 只用 0.34% 算力 —— 所有問題的共同根源 |
| 4 | 地基②：AI ≈ B → 答案永遠是「把 batch 撐大撐滿」；四個問題各擋住 B 的哪一段 |
| **問題① 程式難平行** | |
| 5 | 問題①：Python 迴圈把依賴關係寫死在控制流裡 |
| 6 | 解法①：前端 DSL（`sgl.fork`）把程式描述成執行圖 |
| 7 | 橋：好寫的並行程式會製造大量重複計算 → 端出問題② |
| **問題② 前綴重算** | |
| 8 | 問題②：要複用前綴，得先解決「KV cache 怎麼放」（128 KB/token、預留浪費） |
| 9 | 第一層解法：分頁（KV 怎麼「放」）—— block table、浪費 <4%、**block 可跨請求共享** |
| 10 | 第二層解法：RadixAttention（block 怎麼被「找到」）+ ⚠️ 前綴匹配的常見誤解 |
| 11 | **澄清：「RadixAttention vs PagedAttention」是假對立**（兩層，不是兩派） |
| 12 | 撐大 B 的第三隻腳：continuous batching |
| 13 | 誠實話：RadixAttention 不是萬靈丹（收益邊界） |
| 14 | 🧭 互動環節：serving_map 模式 1–4（6.8% → 81%） |
| **問題③ 輸出不可控** | |
| 15 | 問題③：只靠 prompt 說「請輸出 JSON」不可靠（四種翻車） |
| 16 | 解法③：FSM 預編譯 + 非同步編譯 + 位元遮罩 + 確定路徑跳躍 |
| 17 | 問題② 與 ③ 的關係：正交，但有一條紅線（FSM 狀態不可共享） |
| **問題④ CPU 成瓶頸** | |
| 18 | 問題④：GPU 快到讓 CPU 變成瓶頸 → CUDA Graph / zero-overhead / async |
| 19 | 雙胞胎：TTFT vs ITL 打架 → chunked prefill |
| **天花板** | |
| 20 | 天花板①：投機解碼 / MTP（AI 從 1 變成 k） |
| 21 | 天花板②：量化（直接砍分母） |
| 22 | 單機篇彙整表 |
| 23 | 預告第四堂：⑤–⑧ 全都來自「不只一台」 |
| 24 | 帶走三句話 |

---

## 地基：為什麼四個問題都指向同一件事（第 3–4 頁）

### 推導 1 — batch=1 的 decode 只用 0.34% 算力

設參數量 $N$、權重 fp16。每產一個 token：

- 搬：整份權重讀一遍 ＝ $2N$ bytes
- 算：每個參數一次乘加 ＝ $2N$ FLOPs
- **AI ＝ 1 FLOP/Byte**

H100 ridge point ＝ 990 TFLOPS ÷ 3.35 TB/s ≈ **296 FLOPs/Byte** → 利用率上限 ≈ 1/296 ≈ **0.34%**。
（A100：312 ÷ 2.0 ≈ 156 → 0.6%。**卡越強，batch=1 越浪費。**）

### 推導 2 — AI ≈ B

batch = B 時，權重仍只讀**一遍**，FLOPs 變 B 倍 → **AI ≈ B**。既然 ridge ≈ 296，**B 要拉到幾百，GPU 才餵得飽**。

具體：Llama-3-8B fp16 on H100，權重 16 GB ÷ 3.35 TB/s ≈ 4.8 ms／步。
- batch=1 → ≈ 208 tok/s
- batch=64 → 步時幾乎不變 → ≈ **13,000 tok/s**（60×）

### 四個問題各自擋住 B 的哪一段（第 4 頁，本堂最重要的一張圖）

| 問題 | 它怎麼擋住 B |
|---|---|
| ① 程式難平行 | 根本沒有那麼多請求同時進來——runtime 看不見可平行的分支 |
| ② 前綴重算 | HBM 被浪費的 KV 佔住 → 放不下更多條；算力浪費在不必算的東西上 |
| ③ 輸出不可控 | 每步的語法檢查是 CPU 上的序列工作 → 拖慢整批 |
| ④ CPU 成瓶頸 | GPU 算完在等 CPU 決定下一步 → 有效 batch 再大也填不滿時間軸 |

> **講法**：把這張圖立起來，後面就不是在背功能表，而是「每個機制都在拆掉擋住 B 的一塊石頭」。

---

## 問題① 程式難平行 → 前端 DSL（第 5–7 頁）

### 痛點

多次呼叫、分支、工具使用、self-consistency 取樣——這些步驟裡很多其實可以同時跑，但用 `for` 迴圈寫出來之後，**runtime 只看得到「一個接一個的請求」**，看不見哪幾步彼此無關。加上 Python 的 GIL，前端本身也不擅長真正的並行。

三個痛：
1. **寫的人痛**：分支、重試、工具呼叫的膠水程式又臭又長
2. **跑的人痛**：runtime 收到一串互不相干的請求，無從批處理
3. **更痛的是**：這些請求其實高度共享前綴（同一個 system prompt、同一段歷史），但沒人告訴 runtime

### 解法：把程式描述成一張執行圖

前端 DSL（`sgl.gen` / `sgl.fork` / `sgl.select`）不是語法糖——它讓 runtime 看見「什麼依賴什麼」。

```
for 迴圈           →  runtime 看到：4 個先後到達的請求
sgl.fork()         →  runtime 看到：1 個共享前綴 + 3 條可平行分支
```

> **SGLang 的名字就是這裡來的**：**S**tructured **G**eneration **Lang**uage，論文標題是「Efficient Execution of Structured Language Model **Programs**」——**它先是一個語言，才是一個引擎。**
> 補充給會問的人：實務上多數人把 SGLang 當純服務器用、從沒碰過 DSL，但架構上這一層是真的，而且是理解它為什麼那麼在意「前綴共享」的鑰匙。

### 橋：DSL 立刻製造出下一個問題（第 7 頁）

一個 fork 出 8 條分支、system prompt 2,000 token 的程式 → **同一段 prefill 被算了 8 遍**。
而且真實流量本來就長這樣：RAG（同幾份文件反覆進 prompt）、多輪對話（每輪帶整段歷史）、few-shot（同一批範例）、agent（工具描述 + 前面所有步驟）。

> **所以問題②不是「順便做個快取」，它是 DSL 這條路能不能走通的前提。** 這個因果關係要講出來，否則 RadixAttention 聽起來只是一個優化。

---

## 問題② 前綴重算 → 分頁 + RadixAttention（第 8–14 頁）

### 先算 KV cache 有多大（第 8 頁）

```
每 token = 2 (K,V) × n_kv_heads × head_dim × n_layers × dtype_bytes
```

Llama-3-8B（GQA 8 KV heads、head_dim 128、32 層、fp16）：
- 每層每 token = 2 × 8 × 128 × 2 = 4 KB
- × 32 層 = **128 KB / token** → 一條 8K 序列就是 **1 GB**
- 若是 MHA 32 heads 就是 512 KB/token（**GQA 的價值就在這 4×**）

**問題出在「不知道會生成多長」**：照 `max_seq_len` 連續預留 → 多數請求只用到 20–30%，**60–80% 的 KV 記憶體是死重** → 放得下的條數被砍到 1/4 → B 上不去。

### 第一層：分頁——KV cache 怎麼「放」（第 9 頁）

把 1960 年代的虛擬記憶體分頁機制搬到 KV cache 上（vLLM 的 **PagedAttention** 讓它出名，**SGLang 底下同樣是分頁式記憶體池**）。

- 固定大小 block（16 token 一塊），用完再要，不用連續
- block table 做「邏輯序列 → 實體 block」映射
- 請求結束 block 直接還回 free pool；浪費從 60–80% 掉到 **<4%**
- **關鍵副產品：同一份 block 可被多個請求指向** → 打開了「複用」的門

### 第二層：RadixAttention——block 怎麼被「找到」（第 10 頁）

光能共享還不夠，得有人回答「這個新請求的前綴，之前算過嗎？算到哪？」——這是**索引問題**。

- KV cache 組成一棵 **radix tree**，鍵是 token 序列，分叉點出現在內容開始不同的地方
- 新請求做最長前綴匹配 → 複用命中的 block → 只 prefill 新增那一段
- LRU 淘汰最久未複用的節點；也可分層下放 CPU／磁碟
- 配 **cache-aware 排程**：把共用前綴的請求排在一起，讓命中率最大化

⚠️ **最常見的誤解（一定要講）**：前綴匹配是**嚴格按 token 順序從頭比對**。
「怎麼退貨」和「運費怎麼算」都含「怎麼」，但**不會被合併成同一個節點**——只有共同**開頭**才算數。
理由：KV 是按位置逐 token 累積出來的，第 3 個 token 的 K/V 依賴前面兩個。**位置一樣、前文一樣，K/V 才一樣。**

SGLang 論文宣稱：可共用前綴的工作負載最高 **6.4×** 吞吐。

### 澄清：「RadixAttention vs PagedAttention」是假對立（第 11 頁）

| 層 | 回答什麼問題 | 各家實作 |
|---|---|---|
| **索引 / 複用層** | 已經放好的 KV block 怎麼被找到並複用？ | SGLang：radix tree｜vLLM：16-token 區塊的滾動雜湊表 |
| **記憶體配置層** | KV cache 在 HBM 裡怎麼放？ | 兩邊都是分頁式：固定 block、不連續、用完才要 |

**真正的差異是索引結構與排程策略，不是「要不要分頁」。** 兩邊功能持續趨同——別把某一次 benchmark 當永久結論。

### 撐大 B 的第三隻腳：continuous batching（第 12 頁）

靜態批次（request-level）要**整批等最慢的那條生完**；改成 iteration-level：每一步結束就檢查，誰生完了退場還 block、佇列有人立刻補進來。

> **三隻腳一起看**：分頁撐大「B 的上限」、continuous batching 撐滿「B 的實際值」、radix 省掉「根本不必算的部分」。

### 誠實話：收益邊界（第 13 頁）

| 高收益 | 低收益 |
|---|---|
| RAG、多輪對話、few-shot、agent、`sgl.fork()` 的分支 | 客服（各種不相干問題）、批次翻譯、一次性摘要 |

**選框架看流量長相，不是看誰的 benchmark 數字大**：前綴共用 >60% → SGLang 的 TTFT 通常低 20–40%；每個 prompt 都獨立 → 兩者差 <5%，那就選生態成熟度。

### 互動環節（第 14 頁）

`serving_map.html` 模式 1–4：

| 模式 | 有效 batch | 算力利用率 |
|---|---|---|
| 1 Naive（靜態批次 + 預留 KV） | ≈ 20 | 6.8% |
| 2 Continuous batching | ≈ 60 | 20% |
| 3 Paged KV | ≈ 240 | 81% |
| 4 Radix 前綴共用 | ≈ 240 | 81%，且 TTFT ↓20–40% |

> 數值是教學用示意值（面板底部有註明），用來展示「有效 batch → 算術強度 → 利用率」的因果鏈。

---

## 問題③ 輸出不可控 → Compressed FSM（第 15–17 頁）

### 四種翻車

多餘文字（「當然！以下是您要的 JSON：」）、型別錯誤（`{"age": "twenty"}`）、語法小錯（多逗號／單引號）、幻覺欄位。
Agent／function calling 一旦解析失敗，整條鏈就斷了。

⚠️ **澄清**：這**不需要另一個小模型來審核**——純粹是符號計算／規則引擎的問題。

### 解法：把合法性變成物理限制

1. **預編譯成 FSM**：JSON schema / 正則 → 有限狀態機（SGLang 用 compressed FSM，vLLM 預設 XGrammar）
2. **非同步編譯**：vLLM 讓請求先進 `WAITING_FOR_FSM`，編好才轉 `WAITING`，不阻塞其他請求
3. **位元遮罩**：每步查表得出合法 token 集合，其餘 logit 設成 −∞，在合法子集重新歸一化採樣 → **物理上不可能生成非法內容**
4. **確定路徑跳躍**：FSM 上若某段路徑唯一確定（如 `{"name": "` 必然出現），一次吐掉多個 token

> **反直覺的一點**：加了語法約束之後，生成甚至可能**比自由生成更快**——有些 token 根本不用推理。

### 問題② 與 ③ 的關係：正交，但有一條紅線（第 17 頁）

| | RadixAttention | 約束生成 |
|---|---|---|
| 複用／裁剪什麼 | **算力**（避免重複矩陣運算） | **候選空間**（哪些 token 允許被選） |
| 作用在 | 「這段 token 的中間結果」 | 「這個請求走到語法的哪裡」 |
| 判定 | 與請求無關 → **可共享** | 屬於請求本身 → **不可共享** |

⚠️ **紅線**：即使兩個請求共享同一段前綴快取，**它們各自的 FSM 狀態仍必須獨立推進**。

> **這條紅線給了一個通用判準**：可以快取的是「計算結果」，不可以快取的是「請求狀態」。
> 之後看任何快取設計（包括第四堂的跨機 KV 複製）都先問這個問題。

---

## 問題④ CPU 成瓶頸 → 排程與圖執行（第 18–19 頁）

前面三個問題解完，batch 撐起來了、重複計算省掉了。這時每個 forward step 只剩 5–10 ms，而 tokenize、排程決策、取樣、序列化、**數百次 kernel launch** 全在 CPU 上 → GPU 開始出現「算完在等 CPU」的空窗。

| 招式 | 做什麼 |
|---|---|
| **CUDA Graph** | 初始化時對各種 batch size 做 dummy forward，把整串 kernel launch 錄成 DAG，之後直接 replay |
| **Zero-overhead scheduler**（SGLang） | 把 CPU 排程完全藏進上一步的 GPU 執行時間裡 |
| **多進程 + async 排程**（vLLM V1） | EngineCore 獨立進程只跑排程＋執行；tokenize／多模態前處理／串流輸出與它重疊 |

**vLLM V1 相對 V0 吞吐提升 ~1.7×，完全來自 CPU 開銷削減——一個 GPU kernel 都沒改。**

### 雙胞胎問題：TTFT vs ITL（第 19 頁）

| 指標 | 被誰決定 | 想要它小，你會想… |
|---|---|---|
| **TTFT** | prefill（compute-bound） | 讓 prefill 立刻插隊、獨佔 GPU |
| **ITL** | decode（memory-bound） | 別讓 prefill 打斷 decode |

衝突點是 **head-of-line blocking**：一個 32K prompt 的 prefill 要 0.5–2 秒，這段時間所有串流輸出的使用者都卡住。

**解法：chunked prefill** —— 每步固定 token 預算，剩下的預算拿去混 decode。
附帶好處：compute-bound 的 prefill 與 memory-bound 的 decode 混在同一批，**兩種瓶頸互補，利用率反而更高**。

> ⚠️ chunked prefill 只是**緩解**，沒有根治。根治要靠 PD 分離——那是第四堂的問題⑥。

---

## 單機的天花板（第 20–21 頁）

前面四個問題解完，B 撐大了。但**單一請求的延遲**仍被「讀一遍權重」鎖死。要打破它只有兩條路。

### 天花板① 投機解碼 / MTP

1. **Draft**：便宜地提出 k 個候選（小模型 / n-gram / Medusa 頭 / EAGLE / 模型自帶的 MTP head）
2. **Verify**：大模型把「context + k 個草稿 token」在**一次 forward** 裡跑完
3. **Accept**：由左而右比對機率；`large ≥ draft` 就收，否則按 `large/draft` 機率收；第一個被拒就停，並「免費」得到第 k+1 個 token

**為什麼會賺**：驗證 k 個 token 的 forward，權重讀取 = 1 次（不變），FLOPs = k 倍 → **AI 從 1 變成 k**。
＝用閒置算力換頻寬。這是**唯一能在不增加 batch 的前提下改善單請求延遲**的招式 → 低流量、互動式、本機部署特別有效。

- 輸出分佈**嚴格不變**（rejection sampling 保證），不是近似加速
- 代價：接受率低時純虧
- 期望接受數（接受率 α、草稿長度 k）＝ $(1-\alpha^{k+1})/(1-\alpha)$。α=0.8、k=4 → 約 3.4 個 token/步
- DeepSeek-V3 的 MTP head 報告第二 token 接受率 ~**85–90%**

### 天花板② 量化

AI = FLOPs ÷ Bytes。前面全在動分子，量化直接動分母。

| 對象 | 路徑 | 效果 |
|---|---|---|
| 權重 | fp16 → FP8（÷2）→ FP4/MXFP4（÷4） | decode 步時直接等比下降 |
| KV cache | fp16 → FP8 → 更低 | KV 變小 ⇒ 同 HBM 放得下更多序列 ⇒ **B 又能更大** |
| Activation | FP8 / MXFP8 | 決定 GEMM 能不能走 FP8/FP4 tensor core 路徑 |

硬體對齊：Hopper 有 FP8 tensor core、Blackwell 有 FP4。**精度格式是硬體規格表上的一行，直接決定模型能跑多快**——[第五堂](class5_china_models.md)會看到模型端已經開始「出廠就是 4-bit」。

---

## 帶走三句話（第 24 頁）

1. **每個機制都是被一個具體痛點逼出來的。** DSL 因為程式難平行、RadixAttention 因為 DSL 製造了大量共享前綴、FSM 因為 agent 需要能解析的輸出、zero-overhead scheduler 因為前三個解完 GPU 快到 CPU 跟不上。**順著問題走，就不用背功能表。**
2. **所有解法都在拆掉擋住 batch 的石頭。** batch=1 的 decode 只用 0.34% 算力、AI ≈ B。分頁撐大 B 的上限、continuous batching 撐滿實際值、radix 省掉不必算的、CPU 優化不讓時間軸留空。**一個 kernel 都沒改。**
3. **先看流量長相，再選框架。** RadixAttention 的收益完全取決於前綴重合度。

---

## 撐場用：Q&A 速查

**Q：vLLM 和 SGLang 是同一個 level 的工具嗎？**
A：**引擎這一層是，兩者是直接替代品**（都自己管 KV、自己排程、自己執行模型、自己開 OpenAI 相容 endpoint），部署時二選一。但 SGLang 範圍略大一點：多了**前端 DSL**（vLLM 沒有對應物）和**自帶 router**（vLLM 這層長在生態系裡：production-stack、llm-d、Dynamo）。另外 vLLM 生態更廣（模型與硬體後端更多）。
不同層的例子：FlashInfer / FlashAttention / DeepGEMM 是 kernel 庫（被引擎使用）；Dynamo / llm-d / Ray Serve 是編排層（驅動引擎當 worker）；llama.cpp / MLX 是本機單人推論，目標函數不同。

**Q：MoE 在單卡上不是不省記憶體嗎？稀疏化到底省什麼？**
A：單卡上不省**容量**（所有專家都要在 HBM），但省**每 token 的權重讀取量與 FLOPs**。要連容量也省，得靠 EP——那是第四堂的問題⑤。

**Q：既然 batch 越大越好，為什麼不無限加大？**
A：①KV 吃光 HBM；②超過 ridge point 後變 compute-bound，步時隨 batch 線性成長 → **ITL 變差**；③尾延遲與公平性。生產上是「在 SLO 之下把 batch 開到最大」的約束最佳化。

**Q：這些跟我只有一張 4090 有關嗎？**
A：非常有關。本機跑模型時 batch 幾乎恆等於 1 → 利用率 ~0.3%，**你買的算力 99% 在閒置**。所以本機端的加速幾乎全部來自天花板那兩條路：**量化**（砍分母）與**投機解碼**（把閒置算力換頻寬）。這也解釋了為什麼 llama.cpp / MLX 的核心賣點永遠是量化格式。

**Q：RadixAttention 會不會有安全問題（跨使用者共享快取）?**
A：值得警惕的方向。前綴快取本身只共享「相同 token 序列」的計算結果，內容相同才會命中；但**命中與否會反映在 TTFT 上**，理論上構成一個時間側信道（可探測「某段 prompt 是否被別人用過」）。多租戶場景通常會按租戶分隔快取命名空間。這題目前沒有標準答案，適合當開放討論。

---

## 資料來源

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)（RadixAttention、compressed FSM、前端 DSL、6.4×）
- [RadixAttention 部落格](https://www.lmsys.org/blog/2024-01-17-sglang/)
- [Efficient Memory Management for LLM Serving with PagedAttention（SOSP'23）](https://arxiv.org/abs/2309.06180)（60–80% 浪費 → <4%、2–4× 吞吐）
- [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)（排程器、KV cache manager、prefix caching、chunked prefill、CUDA Graph、投機解碼）
- [vLLM V1 架構升級公告](https://vllm.ai/blog/2025-01-27-v1-alpha-release)（~1.7× 吞吐、統一排程）
- Sarathi-Serve（chunked prefill）、[DeepSeek-V3 技術報告](https://arxiv.org/abs/2412.19437)（MTP 接受率）
