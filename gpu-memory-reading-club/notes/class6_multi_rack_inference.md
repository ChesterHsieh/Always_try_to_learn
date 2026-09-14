# 第六堂課（最終章）講稿／索引 — 一個字，穿過一整排機櫃

> **狀態：骨架已依 2026-09-14 四項決定修訂**——主角＝SGLang 96×H100；保留「四種資料」地圖；NVIDIA vs AMD 集中後段 3 頁；收尾只留全系列對照（1M context 移到附錄 A，不上投影片）。
> 互動教具 [../interactive/rack_journey_map.html](../interactive/rack_journey_map.html) 已完成（第 13 頁指引；第 7、9 頁可直接切到教具第 2、3 層）。投影片 `class6_multi_rack_inference.pptx` 待標題鏈確認後產出。
> 前置：[第一堂](full_series.md)（roofline、記憶體階層）、[第二堂](class2_transformer_gpu.md)（EP 與互連頻寬階梯）、[第三堂](class3_engine_single_node.md)（AI ≈ B、分頁 KV、CUDA Graph）、[第四堂](class4_sglang_multi_node.md)（大規模 EP、PD 分離、router、容錯）、[第五堂](class5_china_models.md)（MLA、MoE、降精度）。

---

## 0. 這一堂要回答什麼

**聽眾的原始問題**：當運行單位是數個機櫃、數十個 node 時，SGLang / vLLM 怎麼把 DeepSeek 這種 671B MoE 跑完「一輪」inference？換成 NVIDIA 或 AMD 硬體，差在哪？

**第四堂 vs 本堂的分工**：第四堂是「問題導向」——EP、PD 分離、router、容錯，一題一題拆開講。本堂是「組裝導向」——**把那些零件裝回一整排真實的機櫃，跟著一個請求從進門走到吐出最後一個字**，每一站算清楚：搬什麼、搬多大、走哪條線、花幾毫秒。

| 項目 | 內容 |
|---|---|
| **鉤子** | DeepSeek 尖峰開 278 台 8 卡機服務 V3/R1；SGLang 用 12 台開源復現。你螢幕上每跳出一個字，decode 池的 72 張卡要一起完成 116 次全員交換。**而且拆到越多卡上，每張卡的產量反而越高。** |
| **鉤子拋出的問題** | 通訊明明變多了，為什麼拆散反而便宜？ |
| **主軸（≤30 字）** | **拆散變便宜的前提：最常搬的資料，走最快的路。** |
| **軸型** | 拆解軸。拆法不是教科書的「router / prefill / decode」三分法，而是**按「多常搬 × 一次多大」把資料分成四種**，看每一種付得起哪一段路；中段照請求時序走。合回來時指出真瓶頸＝**一個 scale-up 域裝得下幾張卡**，再推論到 NVIDIA vs AMD 的選型（對比是次級結構，不換主軸）。 |
| **閉環** | 結尾回答鉤子：72 張卡同步踏一步之所以划算，是因為每字 116 次的 MoE 交換被 DeepEP 壓進 µs 級的特化 RDMA，而一次性的 KV 交接小到（MLA）可以走慢車道；再往上的 4.4×，買的是更大的 NVLink 域。 |

### 三種角色的案例分工

| 角色 | 用哪一套 | 為什麼 |
|---|---|---|
| 鉤子的規模感 | DeepSeek 官方（H800、尖峰 278 台、decode EP144） | 真實線上流量與成本，只出現在第 1、5 頁 |
| **跟著走的主角** | **SGLang 96×H100**（prefill 4 台 EP32、decode 9 台 EP72，在同一個 12 台叢集上分開量測） | 開源框架、可復現、每個手法都有單獨量過的增益 |
| 對照組 | B200 8 卡機、GB200 NVL72、AMD MI355X 8 卡機 | 只在第 13–18 頁與互動教具出現 |

### 閘門一自檢

| 題 | 結果 |
|---|---|
| 主軸 30 字內說得出來？ | ✅「拆散變便宜的前提：最常搬的資料，走最快的路」（20 字） |
| 標題鏈讀起來像摘要？ | ✅ 見 §1，把標題抽成一列讀一次 |
| 每頁拿掉主軸會斷？ | ✅ 第 5–12 頁是請求路徑（順序不可換）、13 是看現象、14–15 是合回來、16–18 是推論；第 19 頁（全系列對照）保留理由：把旅程每一站掛回前五堂，是最終章的資訊增量 |
| 相鄰兩頁能互換？ | ✅ 5→12 依請求時序；13（教具看現象）必須在 14（解釋）之前；16→17→18 是「軟體可搬 → 硬體不同 → 條件式判斷」 |

---

## 1. 頁面地圖＝標題鏈（20 頁）

> **快測**：只讀這一欄，應該就是一篇完整摘要。

| 頁 | 標題（斷言句） | 視覺 | 本頁回答 → 拋出 |
|---|---|---|---|
| **鉤子與拆法** | | | |
| 1 | 你看到的每一個字，背後有 72 張卡同時踏了一步 | — | → 為什麼要這麼多張卡一起踏？ |
| 2 | 把 DeepSeek 拆到越多卡上，每張卡反而產出越多 token | V2 長條：TP16 vs EP72 等三組 | 拆散真的更便宜 → 通訊變多了，怎麼會？ |
| 3 | 一個「模型服務」不是一台機器，是一個 router、兩個池子、三張網 | V1 機櫃拓樸圖 | 服務長什麼樣 → 請求在裡面搬了哪些東西？ |
| 4 | 機櫃群就是多了三層的記憶體階層：每種資料按「多常搬 × 一次多大」決定走哪條路 | V2 四種資料 × 四條路（教具第 1 層） | 拆法 → 從進門那一跳開始走 |
| **跟著一個請求走** | | | |
| 5 | 進門那一跳只搬幾 KB，卻決定了一半的 prefill 要不要算 | V3 router 決策流程 | router 選誰很重要 → 選好了，prefill 怎麼跨機算？ |
| 6 | Prefill 是整批的大塊通訊，跨 InfiniBand 也吃得消 | V2 機內 vs 跨機頻寬 | prefill 可以跨機 → 算完的 KV 怎麼交給 decode？ |
| 7 | KV 交接不是「傳過去」而已：SGLang 先預留再推，vLLM 先算完再拉 | V3 三泳道時序圖（教具第 2 層） | 交接流程 → 搬這一包要多久？ |
| 8 | 一份 5K token 的 KV 跨機只要 7 ms——MLA 在第五堂就替這一跳付了帳 | V2 MHA / GQA / MLA 傳輸時間長條 | KV 交接不是瓶頸 → 那 decode 一步裡在忙什麼？ |
| 9 | Decode 一步：attention 各算各的，MoE 大家一起交換——每個字 116 次 | V3 一層的資料流（教具第 3 層） | 熱迴圈長相 → 這 116 次要花多少時間？ |
| 10 | 不做重疊時，通訊佔掉 decode 一步約四成——引擎的工作變成「把等待藏進計算裡」 | V2 一步的時間堆疊長條 | 通訊很貴 → 那為什麼拆散還是比較便宜？ |
| 11 | 拆散之所以便宜：每張卡只放 4 個專家，省下的 HBM 拿去開大 batch | V2 單卡 HBM 佔用：TP16 vs EP72 | 回答鉤子的一半 → 代價是什麼？ |
| 12 | 代價是 72 張卡綁成一個節拍器：最慢那張決定速度，掉一張全部停 | V1 節拍器示意（barrier） | 代價 → 什麼條件下這個代價付得起？ |
| **看現象 → 合回來** | | | |
| 13 | 互動環節：同一個請求放到四種硬體上各走一次——只有一站的時間變了 | 教具指引頁 | 看到現象 → 為什麼只有那一站？ |
| 14 | 四種資料只有一種付不起跨機的路——真正的硬體瓶頸是「一個 scale-up 域裝得下幾張卡」 | V1/V2 跨機流量比例 | 真瓶頸 → 有實測證據嗎？ |
| 15 | 同一顆 GPU、同樣的 NVLink 頻寬，只把域從 8 擴到 72：每卡吞吐 4.4×、成本剩 1/4 | V2 B200 vs GB200 | 證據 → 換成 AMD 呢？ |
| **推論：NVIDIA vs AMD** | | | |
| 16 | 軟體圖搬得過去，零件得重寫一遍：AMD 落後的是「新模型上線那天」 | 零件對照表 | 軟體可搬但有時間差 → 那硬體差在哪？ |
| 17 | AMD 的強項是「一台裝得下」，弱項是「一個域只有 8 張」 | V2 單台總 HBM vs 對任一卡頻寬 | 硬體差異 → 所以怎麼選？ |
| 18 | 選型只問一題：你的最小單位是一台，還是一個機櫃？ | 條件式決策表 + 72 卡域時間線 | 條件式判斷 → 整個系列在這趟旅程裡的位置 |
| **收尾** | | | |
| 19 | 這趟旅程的每一站，都是前五堂的某一頁 | V1 旅程路線 × 堂次對照 | 全系列收束 |
| 20 | 72 張卡同步踏一步划算，是因為最常搬的東西走了最快的路 | — | 回答第 1 頁 |

---

## 2. 逐頁內容與事實原子

> 強度欄：**可查證**＝一手來源；**單一來源**＝廠商自報、只有一家；**推算**＝本講稿自己算的，標註假設。
> 頻寬一律寫**每方向**；NVIDIA 規格頁的 900 GB/s（NVLink 4）、1.8 TB/s（NVLink 5）是雙向合計，引用時會註明。

### 第 1 頁 · 鉤子

- DeepSeek 在 2025-02-27～28 的 24 小時裡：輸入 **608B** token（其中 **342B / 56.3%** 命中磁碟 KV 快取）、輸出 **168B** token；尖峰佔用 **278 個 node**、平均 **226.75 個**，每 node 8 張 H800；decode 部署單元＝18 node、EP144。〔可查證：DeepSeek open-infra-index day 6〕
- SGLang 在 **12 台 × 8 張 H100** 上開源復現：prefill 4 台（EP32）、decode 9 台（EP72）。〔單一來源：LMSYS 2025-05-05〕
- 「116 次」＝ DeepSeek-V3 共 61 層，前 3 層是 dense，**58 層 MoE × 每層 dispatch + combine 各一次**。〔可查證：config.json `first_k_dense_replace=3`〕

**講法**：「DeepSeek 自己開了 278 台機器在服務。今天我們不看它的黑盒子，而是看開源的 SGLang 怎麼用 12 台做出同一套架構。你手機上每跳出一個字，decode 那 72 張卡就同時做完 116 次全員交換。聽起來很浪費——下一頁會看到，這是最便宜的做法。」

### 第 2 頁 · 拆散反而更便宜（被誤解的常識）

常識：「模型拆越散，通訊越多，一定越慢越貴。」反例：

| 對照 | 每張 GPU 的產出 | 來源 |
|---|---|---|
| SGLang：PD 分離 + 大規模 EP vs TP16 | decode **5.2×**、prefill **3.3×** | LMSYS 2025-05-05〔單一來源〕 |
| TRT-LLM：EP16/32 vs EP4/8 | 每卡輸出**最高 6.17×**（含 MTP） | NVIDIA TRT-LLM tech blog〔單一來源〕 |
| InferenceX：GB200（decode EP16）vs B200（decode EP8），R1 FP4、125 tok/s/user | **4,130 vs 941 tok/s/GPU（4.4×）** | SemiAnalysis 2026-05-23〔可查證，第三方〕 |

> ⚠️ 三組條件不同，不能互相換算——共同點是方向一致：**拆散（EP 變大）→ 每卡產出變多。**

### 第 3 頁 · 服務長什麼樣（V1 拓樸圖）

圖要畫出（由外到內）：

1. **前端網路**（資料中心乙太）：使用者 → router
2. **Prefill 池**：4 台 × 8 張 H100，EP32
3. **Decode 池**：9 台 × 8 張 H100，EP72——兩個池子是在同一個 12 台叢集上**分開量測**的
4. **三張網**：機內 NVLink 4（每方向 450 GB/s、8 卡一域）／跨機 RDMA（每卡一張 400G ≈ 50 GB/s）／前端乙太（ms 級）
5. 儲存（權重 688.6 GB、磁碟 KV 快取）掛在旁邊

原子：
- DGX H100：8×80 GB、NVLink 900 GB/s（雙向）、10×ConnectX-7 400G（8 張算力網＋2 張儲存網）。〔可查證：NVIDIA DGX H100 頁〕→ LMSYS 原文**未載明網卡配置**，本堂以此典型配置估。
- DeepSeek-V3 權重在 HF 上 163 個 safetensors、**688.6 GB（641.3 GiB）**，其中 680.6B 參數是 FP8。〔可查證：HF repo〕→ **8×80 GB 的一台 H100 連權重都放不下。**

### 第 4 頁 · 本堂地圖：四種資料 × 四條路

| 資料 | 一次多大 | 多常搬 | 付得起的路 | 對應頁 |
|---|---|---|---|---|
| 請求本身（token ids、串流回傳） | KB | 每請求一次 / 每字一次 | 前端乙太（ms 級）就夠 | 5 |
| **KV cache**（prefill → decode） | ≈ 351 MB（MLA BF16，4,989 token） | 每請求一次 | 跨機 RDMA（~50 GB/s） | 7–8 |
| **MoE hidden states**（dispatch / combine） | 每卡每層 ≈ 14 MB ＋ 28 MB | **每層 × 每步 × 72 張卡同步**，每字 116 次 | 只有 scale-up 域或特化 RDMA 付得起 | 9–14 |
| 權重 | 688.6 GB 全部；單專家 FP8 ≈ 42 MiB | 幾乎不搬（啟動、EPLB 重排時） | 儲存網 / RDMA | 12 |

〔MoE 那一列為推算：每卡 256 序列 × 平均送往 7.5 張卡 × hidden 7168，dispatch FP8、combine BF16〕

**講法**：「第一堂那張記憶體階層表，暫存器 → L1 → L2 → HBM → PCIe → SSD，每往外一層慢一個數量級。機櫃群只是**再往外加三層**：scale-up 域 → 跨機 RDMA → 前端乙太。推論引擎的全部工作，就是讓每一種資料只走它付得起的那一段路。」

### 第 5 頁 · 第一站：router

- **SGLang PD 模式**：router（`sglang_router --pd-disaggregation`）挑一組 prefill／decode，塞入 `bootstrap_host / bootstrap_port / bootstrap_room`（隨機 63-bit ID），**同時 POST 給兩邊**；串流從 decode 回來。〔可查證：sgl-model-gateway `mini_lb.py`〕
- **vLLM / llm-d**：Gateway → Endpoint Picker（EPP，KV-aware 排程）→ sidecar → prefill pod → NIXL → decode pod。〔可查證：llm-d docs v0.9〕
- **NVIDIA Dynamo**：Smart Router（KV-aware），Dynamo 1.0 於 2026-03-16 發布，整合 SGLang / vLLM / llm-d / LMCache。〔可查證〕
- **為什麼這一跳重要**：DeepSeek 線上 **56.3%** 輸入 token 命中快取——router 送錯地方，這一半就得重算。（接第四堂⑦ cache-aware router）

### 第 6 頁 · 第二站：prefill

- DeepSeek-V3 prefill 是 compute-bound（第一堂：大 GEMM、AI 高）；通訊是「一批 token 一次送」的大塊流量。
- **DeepEP normal 模式**（prefill 用）：機內 NVLink 轉發 **153 GB/s**、跨機 RDMA **43–58 GB/s**；不支援 CUDA Graph。〔可查證：DeepEP v1.2.1 README〕
- **Two-batch overlap**：把一批切成兩個 micro-batch，一個在通訊時另一個在算 → prefill 吞吐 **+27–35%**。〔單一來源：LMSYS〕
- 數字錨：prefill 4 台（EP32）在 1K / 2K / 4K 輸入下分別 **57,674 / 54,543 / 50,302 tok/s／node**。〔單一來源〕
- EPLB 共 288 個專家槽（256 + 32 冗餘）→ prefill 每張卡放 **9 個 routed 專家**（288 ÷ 32）。〔可查證：LMSYS〕

### 第 7 頁 · 第三站：KV 交接的流程（V3 三泳道）

> 🎛 **教具對應**：第 2 層「KV 交接」，上方可切 SGLang / vLLM 逐步播放。

**SGLang——並行送、先預留、再推**（源碼 main branch）：

1. router 同時送請求給 prefill 與 decode，帶同一個 `bootstrap_room`
2. decode 查 prefill 端的 bootstrap server → 握手（Bootstrapping）
3. decode **預先配置 KV 槽位**（PreallocQueue）
4. prefill 算 forward（Waiting → Inflight，非阻塞輪詢 KV sender）
5. KV 經 RDMA 推送（TransferQueue；後端 Mooncake 或 NIXL）
6. decode 拿到 KV，**跳過 prefill forward** 直接組成 decode batch（PrebuiltExtendBatch → RunningBatch）
7. 串流從 decode 回傳；逾時預設 300 s

〔可查證：`python/sglang/srt/disaggregation/{prefill,decode}.py`、`base/conn.py`〕

**vLLM + NIXL——串行送、先算完、再拉**：

1. proxy（或 llm-d sidecar）先把請求送給 prefill，prefill 只產第一個 token
2. prefill 把 KV 留在自己 GPU 上，回傳 KV 的位址（block ids、engine id、host／port）
3. proxy 把位址附在請求上轉給 decode
4. 第一次接觸時經 ZMQ side channel 交換 NIXL metadata（lazy handshake）
5. **decode 用單邊 RDMA read 從 prefill 的 GPU 記憶體「拉」KV**
6. decode 繼續生成並串流回傳

〔可查證：vLLM NIXL connector 文件〕

**講法**：「同一件事兩種寫法——這就是第四堂說的『KV 所有權轉移』。SGLang 讓 decode 先把位子佔好再開算；vLLM 讓 prefill 先算完、把 KV 留著等人來拉。」

### 第 8 頁 · KV 交接要多久（V2 長條）

以 DeepSeek 官方統計的**平均 KV 長度 4,989 token** 計算：

| 注意力 | 每 token KV | 一個請求 | 跨機 400G（49.5 GB/s） | NVL72 域內 NVLink 5（每方向 900 GB/s） |
|---|---|---|---|---|
| MHA（同骨架假想：128 頭 × 128 維） | ~4.0 MB | ~20 GB | **~400 ms** | ~22 ms |
| GQA（Llama-3-70B：8 KV heads × 80 層） | 320 KB | 1.6 GB | ~33 ms | ~1.8 ms |
| **MLA BF16（DeepSeek-V3）** | **70 KB** | **351 MB** | **~7 ms** | ~0.4 ms |
| MLA FP8 | 35 KB | 175 MB | ~3.5 ms | ~0.2 ms |

〔推算：MLA ＝ 61 層 ×（kv_lora_rank 512 ＋ rope 64）＝ 35,136 值／token；跨機頻寬取 llm-d 2026-06-23 400G IB 實測（UCX）；未計握手、排程與協定開銷〕

對照：SGLang 96×H100 的 TTFT 是 **2–5 秒**（含排隊）→ **7 ms 的交接在 TTFT 裡是雜訊**。主角的 H100 叢集 prefill 與 decode 在不同台機器，只能走跨機那一欄。

> **講法**：「第五堂說 MLA 是為了 decode 的 HBM 頻寬發明的。它還順手付了另一張帳：**讓 prefill 和 decode 可以放在不同機櫃**。換成 MHA，光交接就要 0.4 秒。」

### 第 9 頁 · 第四站：decode 一步（V3 一層的資料流）

> 🎛 **教具對應**：第 3 層「Decode 一步」，可拉 EP 大小、切 dispatch 精度，看一個 token 的 8 個專家落在哪幾張卡。

一層 MoE layer 在 DP-attention + EP 下：

1. **Attention 資料平行**：每張卡只算自己那批請求的 MLA（FlashMLA kernel），KV 只存在自己卡上 → **不重複**（TP 下 MLA 的 latent 無法按 head 切，每張卡都得各存一份）
2. **Gate**：每個 token 選 8 個 routed expert
3. **Dispatch**（DeepEP low-latency，FP8）：token 送去專家所在的卡——**同一張目的卡只送一次**，所以平均送往約 7.5 張卡
4. **Experts**：各卡用 grouped GEMM 算自己的專家（DeepGEMM）
5. **Combine**（BF16）：結果送回原卡
6. × 58 層 → LM head → 取樣 → 串流回使用者

- **DeepEP low-latency 模式**：純 RDMA（NVSHMEM + IBGDA，GPU 直接敲網卡門鈴、不經 CPU）、固定預配置 buffer → 可被 CUDA Graph 錄下來（接第三堂④）。〔可查證：DeepEP README〕
- decode 每張 GPU 只放 **4 個 routed 專家**（288 槽 ÷ 72 卡）+ 1 個 shared。〔推算自 LMSYS 的 288 槽〕
- 前 3 層 dense FFN 也走資料平行。〔可查證：LMSYS〕

### 第 10 頁 · 通訊的時間帳（V2 堆疊長條）

DeepEP low-latency 實測（H800 + CX7 400G、每批 128 token、hidden 7168、top-8、FP8 dispatch）：

| EP 大小 | dispatch | combine | ×58 層＝每步通訊 |
|---|---|---|---|
| 8 | 77 µs | 114 µs | ~11 ms |
| 32 | 155 µs | 273 µs | ~25 ms |
| 128 | 192 µs | 369 µs | ~33 ms |
| 256 | 194 µs | 360 µs | ~32 ms |

〔單層數字可查證：DeepEP v1.2.1 README；×58 為推算〕

**主角的帳**（EP72、每卡 256 序列，套教具模型，見 §6）：

| 項目 | 數字 | 強度 |
|---|---|---|
| 實測一步（TBO 開） | 8 × 256 ÷ 22,282 ≈ **92 ms**（原文 ITL ≈ 100 ms） | 單一來源 |
| 模型推算：每層通訊 | ≈ 0.86 ms → × 58 ≈ **50 ms** | 推算 |
| 反推：不開 TBO 的一步 | 92 × 1.35 ≈ **124 ms** → 通訊佔 **約四成** | 推算 |
| 反推：計算本身 | 124 − 50 ≈ **75 ms** | 推算 |

所以引擎在做的三件事，全都是「把等待藏起來」：

| 手法 | 做什麼 | 數字 |
|---|---|---|
| Two-batch overlap（SGLang）／dual-batch overlap（vLLM）／5 段 pipeline（DeepSeek） | 一個 micro-batch 通訊時，另一個在算 | decode 吞吐 **+35%**（LMSYS，128 序列／卡、模擬 MTP 條件） |
| 通訊降精度 | dispatch 用 FP8 → NVFP4 | NVFP4 dispatch 讓 all-to-all 流量 **÷4**（vLLM GB200 blog，相對 BF16）、**減半**（LMSYS GB200 Part II，相對 FP8） |
| 固定 buffer + CUDA Graph | 每步不重新配置、不回 CPU | 接第三堂④ |

> 💡 **一個 2026 年的轉折**：DeepEP V2（2026-04-29）改用 NCCL 後端、支援到 EP2048，並**拿掉了「零 GPU 開銷」的純 RDMA low-latency 模式**。〔可查證：DeepEP repo〕→ 開講前確認你用的框架版本走哪條路。

### 第 11 頁 · 為什麼拆散便宜（回答鉤子的一半）

**單卡 HBM 佔用（80 GB H100）**：

| | TP16（2 台一組） | DP attention + EP72（9 台） |
|---|---|---|
| 每卡權重 | ≈ 43 GB（688.6 GB ÷ 16） | ≈ 29 GB：4 routed + 1 shared 專家 × 58 層 ≈ 12.8 GB（FP8）；attention 等非專家權重整份複製 ≈ 12.6 GB（FP8）；embedding + LM head ≈ 3.7 GB（BF16） |
| 每卡剩給 KV | ≈ 37 GB，但 MLA latent 無法按 head 切，**16 張卡各存同一份** | ≈ 51 GB，**只存自己那批請求** |
| 72 張卡能放的「不重複」KV | 4.5 組 × 37 GB ≈ **170 GB** | 72 × 51 GB ≈ **3.6 TB** |

〔推算；未扣 activation、CUDA Graph 與 DeepEP buffer，只看量級〕

- **省下的 HBM 拿去放 KV → batch 開大**：decode 每張 H100 跑 **256 條序列**。〔單一來源；交叉驗算：22,282 tok/s ÷ 8 卡 ÷ 256 ≈ 每步 92 ms，與原文 ITL ≈ 100 ms 吻合〕
- **接第三堂的尺**：decode 是 memory-bound，AI ≈ B；H100 的 ridge point ≈ 296，**B 要拉到幾百**才吃得滿算力——大規模 EP 是把 B 推到幾百的方法。
- **EPLB**：熱門專家做副本（32 個冗餘槽）→ decode **2.54×**、prefill **1.49×**。〔單一來源：LMSYS〕
- 實測結果：decode **vs TP16 為 5.2×**。〔單一來源：LMSYS〕

### 第 12 頁 · 代價：節拍器

- DP-attention + EP 下，**同一個 decode 單元的所有 rank 必須同步進入每一層的 all-to-all**——沒請求的卡也得跑空批次陪跑（集合通訊的本質）。
- → **最慢那張卡決定整體 ITL**（第四堂 Part A 的 barrier／straggler）
- → **掉一張卡，整個 72 卡單元停擺**（第四堂⑧：爆炸半徑隨 EP 變大）
- → 專家負載傾斜時，熱門專家那張卡就是 straggler → EPLB 定期重排，**這是權重唯一會「搬家」的時候**：TRT-LLM 實測單一 FP4 專家 24 MiB，最多重排 **348 GiB** MoE 權重。〔單一來源：TRT-LLM blog〕

> **講法**：「拆散換來了大 batch，代價是 72 張卡變成一支軍樂隊——所有人踩同一個節拍。節拍能踩多快，取決於傳令兵跑多快。接下來我們換幾種硬體，看傳令兵的速度差多少。」

### 第 13 頁 · 互動環節指引頁

切出去開 [../interactive/rack_journey_map.html](../interactive/rack_journey_map.html)，建議動線（約 6 分鐘）：

| 步 | 教具操作 | 要讓聽眾看到的現象 |
|---|---|---|
| 1 | 第 1 層全景，停在 H100，點「③ MoE 交換」 | decode 池的交換九成要出機器、走 RDMA |
| 2 | 按 `G` 切 GB200 NVL72 | 同一種資料，整段路縮回機櫃內 |
| 3 | 第 2 層 KV 交接，attention 切 MHA → MLA | 400 ms → 7 ms；在 TTFT 的 2–5 s 裡變成雜訊 |
| 4 | 第 3 層 Decode 一步，EP 從 8 拉到 144（H100） | EP8 權重放不下；EP 越大、跨機比例越高、每卡權重越少 |
| 5 | 第 4 層時間帳，依序看 H100 → B200 → GB200 | **NVLink 快一倍（H100 → B200），通訊一毫秒都沒省；域從 8 變 72（B200 → GB200），通訊 50 → 9 ms** |

> **講法**：「四種硬體跑同一個請求，router、prefill、KV 交接幾乎不動——只有 decode 的交換那一站變了。而且讓它變的不是頻寬，是『一個域裡有幾張卡』。下一頁解釋為什麼。」

### 第 14 頁 · 合回來：真瓶頸

回到第 4 頁的四種資料：

- 請求本身 → 前端乙太就夠 ✅
- KV 交接 → MLA 讓它 7 ms，跨機櫃也行 ✅
- 權重 → 幾乎不搬 ✅
- **MoE 交換 → 每字 116 次、每層都要全員同步 ❌ 只有它付不起慢車道**

量化「慢車道佔多少」（均勻路由假設）：

| 部署 | 一個 token 送往的卡裡，落在同一個 scale-up 域的比例 | 走跨機 RDMA 的流量 | 每步通訊（教具模型） |
|---|---|---|---|
| H100 8 卡機 × 9 台，EP72（主角） | (8 − 1) ÷ (72 − 1) ≈ 10% | **≈ 90%** | ≈ 50 ms |
| B200 8 卡機 × 9 台，EP72 | 同上 | **≈ 90%** | ≈ 50 ms（NVLink 快一倍也沒用） |
| GB200 NVL72，EP72 | 100% | **0%** | ≈ 9 ms |

- 頻寬差：NVLink 5 每方向 **900 GB/s**，是 400G 網卡（50 GB/s）的 **18 倍**。（NVIDIA 行銷寫「36×」，是拿雙向合計比單向。）〔可查證：NVIDIA 2025-06-06 blog〕

〔推算：DeepSeek-V3 有 group-limited routing（8 組選 4 組），實際跨機比例依部署拓樸與 EPLB 擺法而定；此處只為建立量級直覺〕

> **本堂最重要的一句**：MoE 推論的硬體選型，第一個要看的數字不是 FLOPs、也不是單卡 HBM，而是**一個 scale-up 域裝得下幾張卡**。

### 第 15 頁 · 證據（V2）

SemiAnalysis InferenceX（DeepSeek-R1 FP4、1K/1K、Dynamo + TRT-LLM、MTP、**125 tok/s/user**）：

| | 同一顆 GPU？ | 每卡 NVLink | NVLink 域 | 拓樸 | tok/s／GPU | $／1M token |
|---|---|---|---|---|---|---|
| B200（HGX 8 卡） | Blackwell | 1.8 TB/s（雙向） | **8** | 4 prefill + 40 decode（EP8） | 941 | $0.576 |
| GB200 NVL72 | Blackwell | 1.8 TB/s（雙向） | **72** | 8 prefill（TP8）+ 16 decode（EP16） | **4,130** | **$0.149** |

〔可查證：inferencex.semianalysis.com 2026-05-23〕

→ **同一顆晶片、同樣的每卡頻寬**，差別幾乎只剩域的大小（另有 Grace CPU vs x86 的差異，影響較小）：**每卡 4.4×、成本 ÷3.9**。
→ 教具模型只算出通訊那一段的差（一步約 1.2×（TBO 開）～1.5×（TBO 關））；其餘來自域變大後 EP 能開更大、每卡權重更少、batch 更大（第 11 頁的機制）。

旁證：LMSYS GB200 NVL72 Part II，FP8 attention + NVFP4 MoE：**26,156 prefill / 13,386 decode tok/s／GPU**，相對 H100 為 3.8× / 4.8×（這組同時換了 GPU 世代，只當旁證）。

### 第 16 頁 · 軟體圖搬得過去，零件得重寫一遍

**同一張圖**（第 5–10 頁）在 AMD 上的零件對照：

| 旅程的一站 | NVIDIA 生態 | AMD / ROCm 對應物 |
|---|---|---|
| 集合通訊 | NCCL | **RCCL**（ROCm 10 併入上游 NCCL 2.30.4） |
| MoE dispatch / combine | DeepEP（NVSHMEM + IBGDA） | **MoRI-EP**；另有 ROCm 版 DeepEP（rocSHMEM，只列 MI300X/MI308X） |
| KV 交接 | NIXL、Mooncake | **MoRI-IO**（LMSYS 實測比 Mooncake 快約 10%） |
| MLA / MoE kernel | FlashMLA、DeepGEMM、FlashInfer | **AITER**（vLLM `ROCM_AITER_MLA` 比 Triton MLA 快 1.2–1.6×） |
| 共享記憶體通訊 | NVSHMEM | rocSHMEM / MoRI-SHMEM |
| 服務層編排 | Dynamo | **沒有完整上游對應物**（靠 SGLang router / llm-d） |
| 算力網卡 | ConnectX-7/8 | Pensando **Pollara 400**（UEC）、Vulcano 800 |

〔可查證：github.com/ROCm/mori、ROCm 文件 SGLang + MoRI recipe、vLLM ROCm attention backend blog 2026-02-27〕

**搬得過去的證據（最佳情境）**：
- ROCm 官方 recipe：DeepSeek-V3 / R1 在 8×MI355X + 每 node 8 張 RDMA 網卡上跑 SGLang PD 分離 + EP + DP attention（範例：1 prefill node + 2 decode node）。〔可查證〕
- LMSYS × AMD（**廠商合寫**，2026-05-28）：DeepSeek-R1-0528 MXFP4、24 張 MI355X、129 tok/s/user → **2,436 tok/s/GPU、$0.169／1M token**；對照 B200 + TRT-LLM **3,128 tok/s/GPU、$0.178**。〔單一來源；只比 B200，沒比 NVL72〕

**但「搬得過去」不等於「一樣成熟」**（第三方）：
- InferenceX v2（2026-02-16）：FP8 PD 分離 + wide EP 能追上 NVIDIA 上的 SGLang；**FP4 被 B200 大幅甩開**；高互動性區間 PD 分離反而**比單機還慢**；AMD 在開源分散式推論上「**落後超過六個月**」。
- DeepSeek-V4 上線：SemiAnalysis 2026-06-09 仍回報 **V4-Pro 在 ROCm 上的分散式推論跑不起來**；單機 SGLang 在 26 天內從 20.4 → 2,256 tok/s/GPU（**110×**），仍落後 B200 約 5×。

> **講法**：「架構圖一模一樣——落後的不是設計，是**新模型上線那天，零件有沒有人寫好**。對 DeepSeek-R1 這種跑了一年的模型，AMD 已經追到同價位；對剛出的 V4，差距是好幾倍。」

### 第 17 頁 · AMD 的強項是「一台裝得下」，弱項是「一個域只有 8 張」

| | AMD MI355X 8 卡機 | NVIDIA HGX B200 8 卡機 | NVIDIA GB200 NVL72 |
|---|---|---|---|
| 單卡 HBM | **288 GB** HBM3E | 180 GB | ~186 GB（13.4 TB ÷ 72） |
| 一台／一櫃總 HBM | **~2.3 TB** | 1.44 TB | 13.4 TB |
| scale-up 拓樸 | **8 卡全網狀**（xGMI：超頻 PCIe 5，無交換器） | 8 卡、NVSwitch 交換 | **72 卡**、NVSwitch 交換 |
| 對任一張卡的頻寬（每方向） | **~77 GB/s**（7 條合計 ≈ 538 GB/s） | 900 GB/s | 900 GB/s |
| all-to-all（SemiAnalysis） | NVL72 的 **1/18** | — | 基準 |

〔MI355X 容量：TechRadar / Crusoe；xGMI 數字：SemiAnalysis 2025-06-13，「每條 76.8 GB/s」是研究推論；B200：NVIDIA DGX B200 頁〕

- **強項**：DeepSeek-V3 FP8 權重 688.6 GB，**一台 MI355X（2.3 TB）裝得下還剩 ~1.6 TB 給 KV**；一台 H100（640 GB）連權重都放不下。〔推算〕
- **弱項**：EP 一旦超過 8，dispatch 就得走 RDMA——MoRI-EP 在 MI355X 上實測 dispatch **xGMI 345 GB/s vs RDMA 54 GB/s**（6.4×）、combine **420 vs 71 GB/s**。〔可查證：MoRI README〕→ 回到第 14 頁：**付不起慢車道的那種資料，在 AMD 現役機器上只有 8 張卡的快車道**——教具第 4 層裡，MI355X 在 EP72 的通訊和 H100 一樣是 ≈ 50 ms。

### 第 18 頁 · 選型只問一題：你的最小單位是一台，還是一個機櫃？

**條件式判斷**：

| 如果你的單位是… | 決定性的數字 | 2026 年 9 月的答案 |
|---|---|---|
| **一台 node**（中低 QPS、想避開跨機複雜度） | 單台總 HBM | 大容量 8 卡機：**MI355X（2.3 TB）** > B200（1.44 TB）> H200（1.13 TB）→ AMD 在這一格有結構優勢 |
| **數個機櫃**（追 $／token、大規模 EP） | **scale-up 域大小** × 軟體成熟度 | 現役：GB200 / GB300 NVL72 領先；AMD 8 卡網狀在這一格吃虧 |

**機櫃級 scale-up 域的時間線**：

| 系統 | 域大小 | 每卡 scale-up（雙向） | 每卡 HBM | 狀態（2026-09） |
|---|---|---|---|---|
| GB200 NVL72 | 72 | 1.8 TB/s（NVLink 5） | ~186 GB | 主流部署，~120 kW／櫃 |
| GB300 NVL72 | 72 | 1.8 TB/s | ~288 GB | 部署中，最高 142 kW／櫃，每卡 800G 網卡 |
| **Vera Rubin NVL72** | 72 | **3.6 TB/s**（NVLink 6） | 288 GB HBM4 | 2026-05-31 宣布量產，秋季開始出貨（Dell 已交付 CoreWeave） |
| **AMD Helios** | 72 | **3.6 TB/s**（UALink over Ethernet，Tomahawk 6） | **432 GB** HBM4（櫃 31 TB） | AMD：Q3 末開始出貨；SemiAnalysis：2027 Q2 才量產 |

〔NVIDIA：產品頁與 NVL72 參考架構；Rubin：NVIDIA 2026-01-05、2026-05-31；Helios：AMD newsroom 2026-07-23、StorageReview（225–245 kW 等細節為單一來源）〕

- **Helios 是 AMD 第一次在「域大小」這一格和 NVIDIA 對等**，而且每卡 HBM 多 50%。
- 為什麼是乙太網隧道而不是真 UALink：UALink 交換器趕不上 2026 年底，原生 UALink（256 卡）要等 MI500、2027 年底。〔SemiAnalysis〕

**什麼會推翻這個判斷**：
1. Helios 出貨時，MoRI／RCCL 在 72 卡 UALoE 域上**還沒有公開的大規模 EP 實測**——硬體對等 ≠ 軟體對等（第 16 頁的半年差）。
2. 工作負載轉向超長上下文、KV 容量變成主角時，**單卡 HBM 大的一方優勢放大**。

> **講法**：「不要問『AMD 還是 NVIDIA』，要問『我的最小單位是什麼』。單位是一台，AMD 的 288 GB 是真優勢；單位是機櫃，2026 年底前 NVL72 是唯一成熟的 72 卡快車道——Helios 和 Rubin 會在 2027 年正面對撞，而那一仗的勝負會落在軟體。」

### 第 19 頁 · 旅程 × 全系列（V1）

| 旅程的一站 | 決定它快慢的東西 | 在哪一堂 |
|---|---|---|
| router 選誰 | cache-aware、局部性 vs 均衡 | 第四堂⑦ |
| prefill 跨機 | compute-bound、大 GEMM | 第一堂 roofline |
| KV 交接 7 ms | MLA 把 KV 壓到 70 KB | 第五堂旋鈕① |
| KV 放得進 HBM | 分頁 KV、continuous batching | 第三堂② |
| decode 為什麼要大 batch | memory-bound、AI ≈ B | 第一堂、第三堂地基 |
| 116 次 all-to-all 走哪條線 | scale-up 域 vs RDMA 頻寬階梯 | 第二堂 Part B |
| 每步不回 CPU | CUDA Graph、固定 buffer | 第三堂④ |
| 專家熱點、掉卡 | EPLB、爆炸半徑 | 第四堂⑤⑧ |
| 每字少搬幾個 byte | MoE 稀疏、FP8/NVFP4 | 第五堂旋鈕②⑤ |

> 「第一堂那張記憶體階層表，在第六堂長成了一座資料中心。」

### 第 20 頁 · 帶走三句話（閉環）

1. **72 張卡同步踏一步划算，是因為最常搬的東西走了最快的路。** 每字 116 次的 MoE 交換被 DeepEP 壓到每層不到 1 ms；一次性的 KV 交接被 MLA 壓到 7 ms，可以跨機櫃。
2. **拆散的收益在 HBM，代價在節拍。** 每卡只放 4 個專家、每卡跑 256 條序列；但整個單元被最慢的卡和最脆弱的卡綁住。
3. **選型先問最小單位。** 單位是一台，看總 HBM（MI355X 一台 2.3 TB 裝得下 DeepSeek）；單位是機櫃，看域的大小（同一顆 Blackwell，域從 8 到 72，每卡 4.4×）——而 2027 年 Helios 對 Rubin 的勝負會落在軟體。

---

## 3. 視覺決策（閘門：V1–V4）

| 頁 | 圖 | 通過條件 | 刪掉要多講幾句？ |
|---|---|---|---|
| 2 | 每卡吞吐長條 | V2（三組以上數字比較） | ≥3 句 |
| 3 | 機櫃拓樸 | V1（池子、網路、機櫃的位置關係） | ≥5 句方位詞 |
| 4 | 四種資料 × 四條路 | V2（量級差 + 頻率差） | ≥4 句 |
| 7 | 三泳道時序圖 | V3（>4 步、有並行分支、兩種寫法） | ≥6 句 |
| 8 | KV 傳輸時間長條 | V2（跨兩個數量級） | ≥3 句 |
| 9 | 一層資料流 | V3（6 步、含跨卡回流） | ≥5 句 |
| 10 | 一步時間堆疊 | V2 | ≥3 句 |
| 11 | 單卡 HBM 佔用 | V2 | ≥3 句 |
| 13 | 教具截圖（第 4 層時間帳） | V2 | 指引頁本身 |
| 14 | 跨機流量比例 | V1 + V2 | ≥4 句 |
| 15 | B200 vs GB200 | V2 | ≥2 句 |
| 19 | 旅程路線 × 堂次 | V1 | ≥5 句 |

不放：機櫃照片（除非講者有現場照並圈出 NVLink spine／compute tray）、廠商 logo、裝飾圖。

---

## 4. 待補來源 / 開講前要再對一次的

- InferenceX 儀表板數字持續更新，**開講前對一次**。
- DeepEP V2 拿掉純 RDMA low-latency 模式後，SGLang / vLLM 的 decode 預設走哪個後端——確認版本。
- 公開文獻只有 KV 傳輸**頻寬**，沒有端到端交接**毫秒數**；第 8 頁的 ms 全是推算。
- LMSYS 96×H100 原文**未載明網卡配置**；本堂以 DGX H100 典型的每卡一張 400G 估。
- TBO 的 decode +35% 是在「128 序列／卡、模擬 MTP」條件下量的；第 10 頁用它反推 124 ms 是跨條件套用。
- Vera Rubin 的命名在 CES 2026 前從 NVL144 改回 **VR NVL72**（72 個封裝），第二堂講稿寫的是舊名，需同步。
- AMD xGMI「每條 76.8 GB/s × 7 條」是研究推論（SemiAnalysis 原文寫 per GPU）；Helios 的 225–245 kW、12 顆 Tomahawk 6 只見於 StorageReview。
- LMSYS × AMD 的 $0.169／1M 是廠商合寫、只對照 B200；24 張 MI355X 的 EP/DP 切法原文不明確。
- Helios 截至 2026-09-14 **查不到公開的客戶實際到貨紀錄**。

---

## 5. 資料來源

- [DeepSeek-V3/R1 Inference System Overview](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)（608B/342B/168B、278 node、EP32/EP144、平均 KV 長度 4,989）
- [DeepSeek-V3 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json)（61 層、前 3 層 dense、256+1 專家選 8、MLA 512+64）
- [DeepEP v1.2.1 README](https://github.com/deepseek-ai/DeepEP/blob/v1.2.1/README.md)（low-latency dispatch/combine µs、normal 模式頻寬）｜[DeepEP V2](https://github.com/deepseek-ai/DeepEP)
- [LMSYS：96×H100 大規模 EP + PD 分離](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)（本堂主角）｜[GB200 Part I](https://www.lmsys.org/blog/2025-06-16-gb200-part-1/)｜[GB200 Part II](https://www.lmsys.org/blog/2025-09-25-gb200-part-2/)｜[GB300 長上下文](https://www.lmsys.org/blog/2026-02-19-gb300-longctx/)｜[DeepSeek-V4 day-0](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
- [SGLang PD 分離文件](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/pd_disaggregation.mdx)｜[sglang_router mini_lb.py](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py)
- [vLLM：大規模服務（H200 2.2k tok/s/GPU）](https://vllm.ai/blog/2025-12-17-large-scale-serving)｜[vLLM：DeepSeek-R1 on GB200](https://vllm.ai/blog/2026-02-03-dsr1-gb200-part1)｜[vLLM：DeepSeek-V4](https://vllm.ai/blog/2026-04-24-deepseek-v4)｜[NIXL connector](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/)
- [llm-d wide EP](https://llm-d.ai/docs/well-lit-paths/foundations/wide-expert-parallelism)｜[llm-d 網路實測](https://llm-d.ai/blog/networking-for-distributed-inference-llm-d)
- [NVIDIA Dynamo 1.0](https://nvidianews.nvidia.com/news/dynamo-1-0)｜[TRT-LLM 大規模 EP](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.html)
- [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)｜[GB200 NVL72 × Dynamo（36× 說法）](https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models)｜[NVL72 AI factory 參考架構（GB300）](https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/components.html)｜[Vera Rubin 平台](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)｜[Vera Rubin 量產宣布](https://nvidianews.nvidia.com/news/vera-rubin-full-production-agentic-ai-factory)｜[DGX H100](https://www.nvidia.com/en-eu/data-center/dgx-h100/)｜[DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)
- [InferenceX：GB200 NVL72 vs B200](https://inferencex.semianalysis.com/blog/gb200-nvl72-vs-b200-disagg-deepseek-r1-fp4-dynamo-trt)｜[InferenceX v2（AMD 首次多機 PD + wide EP）](https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs)｜[AgentX／InferenceX v3](https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat)｜[MI355X × DeepSeek-V4-Pro 26 天 110×](https://inferencex.semianalysis.com/blog/mi355x-deepseek-v4-pro-sglang-110x-in-26-days)
- AMD：[MoRI](https://github.com/ROCm/mori)｜[SGLang + MoRI recipe（ROCm 文件）](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/distributed/sglang-mori-recipe.html)｜[LMSYS × AMD：MI355X 上的 PD 分離 + MoRI](https://www.lmsys.org/blog/2026-05-28-mori/)｜[vLLM + MoRI wide EP on MI300X](https://rocm.blogs.amd.com/software-tools-optimization/wide-ep-deepseek/README.html)｜[vLLM ROCm attention backend](https://vllm.ai/blog/2026-02-27-rocm-attention-backend)｜[ROCm DeepEP port](https://github.com/ROCm/DeepEP)
- AMD 硬體：[MI325X 發表](https://ir.amd.com/news-events/press-releases/detail/1220/amd-delivers-leadership-ai-performance-with-amd-instinct-mi325x-accelerators)｜[MI355X 規格（TechRadar）](https://www.techradar.com/pro/amd-releases-details-of-288gb-mi355x-accelerator-80-percent-faster-than-mi325x-8tb-s-memory-bandwidth)｜[SemiAnalysis：MI355X xGMI 與 UALoE72](https://newsletter.semianalysis.com/p/amd-advancing-ai-mi350x-and-mi400-ualoe72-mi500-ual256)｜[Helios 2026 更新（AMD newsroom）](https://newsroom.amd.com/news/aai-2026-helios-update/)｜[StorageReview：MI455X 與 Helios](https://www.storagereview.com/news/amd-mi455x-and-helios-432gb-hbm4-72-gpu-racks-and-a-real-answer-to-vera-rubin)

---

## 6. 互動教具的計算模型（講者備查）

教具 [rack_journey_map.html](../interactive/rack_journey_map.html) 的數字都由下面幾條算式產生；被問「這數字哪來的」時照這裡回答。

**KV 交接時間**＝每 token KV × token 數 ÷ 鏈路頻寬（未計握手與協定開銷）。

**MoE 每層通訊**（decode、每卡 B = 256 序列）：

1. 一個 token 送往的不同卡數 u ＝ EP ×（1 −（1 − 1/EP）⁸）；扣掉自己，遠端約 u ×（EP − 1）/ EP 張（EP72 時 ≈ 7.5）
2. 其中落在同一個 scale-up 域的比例 ＝（域大小 − 1）/（EP − 1）
3. 每段（dispatch、combine）時間 ＝ max（域內位元組 ÷ scale-up 每方向頻寬，跨域位元組 ÷ 網卡頻寬）
4. 每層通訊 ＝ dispatch ＋ combine ＋ 固定開銷 0.11 ms；每步 ＝ × 58 層

**校準**：拿 DeepEP README 的實測條件（128 token、FP8、純 RDMA 50 GB/s）代入第 1–3 步：

| EP | 模型純線速 dispatch / combine | DeepEP 實測 dispatch / combine |
|---|---|---|
| 32 | 128 / 255 µs | 155 / 273 µs |
| 128 | 142 / 283 µs | 192 / 369 µs |
| 256 | 144 / 288 µs | 194 / 360 µs |

→ 純線速解釋了實測延遲的 75–95%，殘差取整成每層 0.11 ms 固定開銷。

**一步的時間帳**：計算時間固定為 74.5 ms、TBO 能藏住 65% 的通訊——兩個數由主角實測反推（一步 92 ms、TBO +35%、模型通訊 50 ms）。**換硬體時刻意不改計算時間**，只改互連，用來隔離「域大小」這個變數。

**每卡權重**＝（288 ÷ EP ＋ 1）× 單專家槽 2.55 GB（44M 參數 × 58 層，FP8）＋ 非專家權重 16.3 GB。

| 硬體 | 域大小 | scale-up 每方向 | 網卡 | HBM |
|---|---|---|---|---|
| H100 8 卡機 | 8 | 450 GB/s（NVLink 4） | 400G ≈ 50 GB/s | 80 GB |
| B200 8 卡機 | 8 | 900 GB/s（NVLink 5） | 400G | 180 GB |
| GB200 NVL72 | 72 | 900 GB/s（NVLink 5） | 400G | ~186 GB |
| MI355X 8 卡機 | 8 | 7 × 76.8 ≈ 538 GB/s（xGMI 全網狀） | Pollara 400 | 288 GB |

---

## 附錄 A：未上投影片的延伸素材——1M context

> 2026-09-14 決定不上投影片，保留在這裡供 Q&A。

- DeepSeek-V4-Pro **1.6T-A49B**、V4-Flash **284B-A13B**，都支援 1M context。〔可查證：NVIDIA blog 2026-04-24〕
- vLLM day-0：1M context 下 **每條序列 KV 9.62 GiB**（BF16），V3.2 式架構則要 **83.9 GiB**。〔可查證：vLLM blog 2026-04-24〕
- 換算跨機 400G（49.5 GB/s）：V4 **~0.2 s**、V3.2 式 **~1.8 s**。〔推算〕→ 第 4 頁 KV 那一列，在 1M context 下從「搬得起」退回「要算一算」。
- SGLang day-0 對 V4 的回應正在這一列：**ShadowRadix** 前綴快取、**HiSparse**（KV 卸載到 CPU）、**context parallel** attention。〔可查證：LMSYS 2026-04-25〕
- 硬體端：NVIDIA CMX（第二堂）把 KV 卸載到 BlueField-4 + flash；AMD 的回應是容量——Helios 每卡 432 GB HBM4。
- 軟體缺口：vLLM 的 context parallel 目前**所有 AMD 後端都不支援**（InferenceX v3，2026-08-24）。
