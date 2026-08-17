# 第五堂課 講稿／索引 — 中國開源模型：把推論成本寫進架構本身

投影片：[../slides/class5_china_models.pptx](../slides/class5_china_models.pptx)（16 頁）｜重建：`cd ../slides/build && node generate_class5.js`

> **主幹＝五個旋鈕**（壓 KV / 少算 / 少看 / 一次多產 / 降精度）。每一家實驗室只是在這五個旋鈕上轉了不同組合。
> **承接第三、四堂**：那兩堂是「框架從**外面**調」（排程、記憶體管理、路由）——一個模型權重都沒改；這一堂是「模型從**裡面**改」。**兩邊打的是同一個敵人**：decode 的 memory-bound 與 KV cache。

---

## 頁面地圖（16 頁）

| 頁 | 內容 |
|---|---|
| 1 | 標題：把推論成本寫進架構本身 |
| 2 | 命題：效率不是加分項，是生存條件（三個共同特徵） |
| 3 | 承接前兩堂：框架動分子、模型動分母，同一個敵人 |
| 4 | **五個旋鈕**（本堂骨架） |
| 5 | 旋鈕① 壓 KV：MHA → GQA → MLA |
| 6 | 旋鈕② 少算：MoE 稀疏度一路往上推 |
| 7 | 旋鈕③ 少看：稀疏 vs 線性，兩條不同的路 |
| 8 | **MiniMax 反例（1）**：M1→M2→M3 與「No Free Lunch」的前兩個理由 |
| 9 | **MiniMax 反例（2）**：卡在三個生產系統上（回扣第三堂） |
| 10 | 旋鈕④ 一次多產（MTP）、旋鈕⑤ 降精度（FP8 → MXFP4 QAT） |
| 11 | DeepSeek：五個旋鈕全轉的教科書 |
| 12 | Kimi：K2 → K3 |
| 13 | Qwen / GLM：混合注意力成為主流 |
| 14 | 全景：五個實驗室 × 五個旋鈕 |
| 15 | **為什麼他們連 kernel 都開源？**（與第三、四堂真正的接縫） |
| 16 | 帶走三句話 + 全系列收束 |

---

## 命題（第 2 頁）

出口管制下算力受限，使得**效率不是加分項，是生存條件**。結果是這批實驗室的三個共同特徵：

1. **架構層就為推論成本設計** —— 不是「先訓練完再想怎麼服務」。MLA 這種注意力機制的發明動機，直接就是 HBM 頻寬帳單。
2. **系統零件跟著開源** —— FlashMLA / DeepEP / DeepGEMM / EPLB / 3FS。新架構必須被 vLLM、SGLang 支援才有人用（見第 15 頁）。
3. **技術報告寫得很細，含失敗嘗試** —— MiniMax 那篇尤其可貴。

---

## 五個旋鈕（第 4 頁，全堂骨架）

| 旋鈕 | 打擊的瓶頸 | 代表技術 |
|---|---|---|
| **① 壓 KV** | decode 讀 KV 的頻寬 + 容量 | MHA → GQA → **MLA**（低秩 latent）→ Gated MLA |
| **② 少算** | 每 token 的 FLOPs 與權重讀取 | **MoE 稀疏化**：細粒度專家、共享專家、極高稀疏比 |
| **③ 少看** | 長 context 的 O(n²) 與 KV 線性增長 | **稀疏注意力**（DSA/CSA/MSA）、**線性注意力**（Lightning/GDN/KDA）、混合層 |
| **④ 一次多產** | 單請求延遲（memory-bound 天花板） | **MTP** ＋ 投機解碼 |
| **⑤ 降精度** | 搬的位元組數 | FP8 訓練、**MXFP4 權重 / MXFP8 activation 的 QAT** |

> 每個旋鈕都在回答同一個問題：**怎麼讓每產一個 token，少搬一點位元組？** 這正是第一堂 roofline 的分母。

---

## 旋鈕① 壓 KV（第 5 頁）

| | 做法 | KV 每 token |
|---|---|---|
| **MHA** | 每個 head 各存 K/V | Llama 式 32 heads：**512 KB** |
| **GQA** | 多個 query head 共用一組 K/V | Llama-3-8B（8 KV heads）：**128 KB**（÷4） |
| **MLA** | K/V 投影成低秩 latent 再存，用時解回 | DeepSeek-V3：**≈ 70 KB**（576 維 latent × 61 層 × 2B；同規模 MHA 推算 ~3.8 MB） |

DeepSeek-V2 論文自陳：MLA 讓 KV cache 相對 MHA **減少 93.3%**。Kimi K3 用 Gated MLA、GLM-5 也採用 MLA——**這個旋鈕已經是共識**。

> **最該記的一句**：MLA 是「為了 decode 的 HBM 頻寬」而發明的注意力機制——**架構決策就是硬體帳單**。

---

## 旋鈕② 少算（第 6 頁）

決定 decode 速度的是**活躍參數 + KV**，不是總參數。所以趨勢是「總參數大幅變大、活躍參數只小幅變大」。

| 模型 | 總參數 / 活躍 | 專家配置 | 活躍比例 |
|---|---|---|---|
| DeepSeek-V3 | 671B / 37B | 256 routed + 1 shared，選 8 | 5.5% |
| Kimi K2 | 1T / 32B | 384 專家，選 8 | 3.2% |
| Kimi K3 | 2.8T / 104B | 896 專家，選 16（Stable LatentMoE） | 3.7% |
| Qwen 3.5 | 397B / 17B | 極高稀疏 | 4.3% |
| GLM-5 | 744B / 40B | — | 5.4% |

- **細粒度專家**：切更小、選更多 → 組合數變多，表達力上升
- **共享專家**：每 token 必經，承接共通知識 → 讓 routed 專家專心學差異
- **無輔助損失負載均衡**（DeepSeek）：傳統 aux loss 逼均衡會傷品質 → 改成動態調整路由 bias

⚠️ **但 MoE 在單卡上不省容量**（專家都得在 HBM），只省每 token 的 FLOPs 與權重讀取。**要連容量也省，得靠[第四堂](class4_sglang_multi_node.md)的大規模 EP**——而 EP 的代價是 all-to-all，以及專家負載不均（＝資料傾斜）。

---

## 旋鈕③ 少看：稀疏 vs 線性（第 7 頁，本堂最重要的分類）

| | 稀疏注意力 | 線性注意力 |
|---|---|---|
| 做法 | **保留完整 KV**，但每個 query 只看 top-k 個位置 | **不存 KV**，改成固定大小的遞迴狀態 |
| 代表 | DeepSeek DSA / CSA+HCA、MiniMax MSA、GLM-5 | MiniMax Lightning、Qwen Gated DeltaNet、Kimi KDA |
| 生態影響 | **KV 還在 → prefix caching / 投機解碼還能用** | **KV 沒了 → 三個生產系統全部要重做** |

> **這個分類是理解 2026 的鑰匙**：共識不是「線性取代 full attention」，而是「**混合 + 稀疏**」——因為稀疏保留了「KV 還在」這個前提。

---

## MiniMax：最有價值的反例（第 8–9 頁，本堂重點）

| 版本 | 做法 | 結果 |
|---|---|---|
| **M1**（2025-06） | Lightning Attention 混合（線性為主）、CISPO RL | 宣稱長生成場景 FLOPs 大幅低於同級 |
| **M2**（2025-10） | **退回 full attention** | 品質優先；並公開說明為什麼 |
| **M3**（2026） | **MSA**（MiniMax Sparse Attention）——改走稀疏 | 宣稱 1M ctx 下 prefill **9×**、decode **15×** 快於 M2 |

### 「No Free Lunch」的三個理由

1. **評測會騙人**：混合注意力在 MMLU / BBH / LongBench 上看起來沒問題，**放大後才發現多跳推理明顯退化**——把散落在長文件裡的線索串起來的能力壞掉了。而要在困難任務上得到統計顯著訊號，所需算力是天文數字。
   > **弔詭**：「省算力的方法」，需要巨量算力才驗證得了。這是效率研究最大的結構性障礙。
2. **理論 FLOPs ≠ wall-clock**：線性注意力的實作**本身就是 memory-bound**，即使在訓練時也吃不滿算力——完全是第一堂 roofline 的教訓：省下的是紙上的 FLOPs，不是牆上的時間。
3. **卡在三個生產系統上**（第 9 頁，第三堂與第五堂的接縫）：

| 系統 | 對應第三堂 | 打壞了什麼 |
|---|---|---|
| **KV cache** | 問題② | 線性狀態對數值精度遠比 full attention 敏感 → 低精度存不了，**旋鈕⑤ 跟著失效** |
| **Prefix caching** | 問題② | 線性狀態不像 KV 可以直接切片複用 → **RadixAttention 的整套價值歸零** |
| **投機解碼** | 天花板① | 在線性骨幹上「仍是未解問題」→ **單請求延遲的唯一解法沒了** |

他們也試過滑動窗口混合，調過比例、RoPE 設定、層內/層間配置、sink token——**在 agent 任務與複雜長文評測上一致地很差**。

> **帶走**：「理論複雜度更低」離「生產環境更快」隔著三層：**kernel 效率、評測有效性、生態相容性**。
> 這也順帶回答聽眾常問的「Mamba／線性注意力不是早就贏了嗎？」——沒有，而且原因非常具體。

---

## 旋鈕④⑤（第 10 頁）

### ④ MTP（Multi-Token Prediction）

訓練時多預測幾步當額外訊號（更密的監督），推論時那些 head **直接當投機解碼的 draft**。

驗證 k 個草稿 token：權重讀取 = 1 次（不變），FLOPs = k 倍 → **AI 從 1 變成 k**（第三堂天花板①的同一把尺）。

DeepSeek-V3 報告第二 token 接受率 ~**85–90%**；Qwen3-Next 也內建 MTP。
> 這是模型端**主動配合投機解碼**的做法——訓練時就把 draft 模型長在自己身上。

### ⑤ 降精度：從「部署後處理」變成「訓練的一部分」

| 階段 | 做法 | 意義 |
|---|---|---|
| 以前 | 訓練用 bf16 → 社群事後量化成 GGUF/AWQ | 品質掉多少看運氣 |
| **DeepSeek-V3** | **FP8 訓練**（首個大規模開源前沿模型） | 細粒度 scaling + 高精度累加解決數值問題 |
| **Kimi K3** | **MXFP4 權重 / MXFP8 activation，從 SFT 起 QAT** | **出廠就是 4-bit**，直接對齊 Blackwell FP4 |

---

## 五個實驗室（第 11–14 頁）

### DeepSeek（第 11 頁）

| 技術 | 旋鈕 | 重點 |
|---|---|---|
| MLA | ① | V2 自陳 KV cache 相對 MHA 減少 93.3%；V3 ≈ 70 KB/token |
| DeepSeekMoE | ② | 細粒度 + 共享專家；V3：671B-A37B |
| 無輔助損失均衡 | ② | 動態調整路由 bias，均衡且不干擾主目標 |
| MTP | ④ | 第二 token 接受率 ~85–90% |
| FP8 訓練 | ⑤ | 首個大規模用 FP8 完成訓練的開源前沿模型 |
| DualPipe + 通訊 kernel | 系統 | all-to-all 與計算重疊（第四堂⑤） |
| **DSA**（V3.2-Exp, 2025-09） | ③ | 細粒度稀疏注意力 → API 降價 >50%（$0.27/M input） |
| **V4**（2026-04 預覽） | ③ + 系統 | CSA + HCA 逐層交錯、mHC 殘差、Muon。1.6T-A49B / 284B-A13B，1M ctx；**1M ctx 下 FLOPs 只需 V3.2 的 27%、KV cache 只需 10%** |

### Kimi（第 12 頁）

| | K2（2025-07） | K3（2026-07） |
|---|---|---|
| 規模 | 1T / 32B 活躍 | **2.8T / 104B 活躍** |
| MoE | 384 專家 / 選 8 | **896 專家 / 選 16**（Stable LatentMoE，latent 3584） |
| 注意力 | MLA | **93 層 ＝ 69 KDA ＋ 24 Gated MLA** |
| 訓練 | **MuonClip**（Muon + QK-Clip）；15.5T tokens 無 loss spike | 宣稱 scaling efficiency ≈ **2.5× K2** |
| 精度 | — | **MXFP4 / MXFP8 QAT** |
| 其他 | — | 原生多模態（401M vision encoder）、1M ctx |

### Qwen / GLM（第 13 頁）

- **Qwen3**（2025）：GQA + MoE（235B-A22B 等）
- **Qwen3-Next**：Gated DeltaNet : full ＝ **3:1**，加 MTP，極高稀疏（80B-A3B）
- **Qwen 3.5**（2026-02）：397B-A17B，延續 GDN 3:1；宣稱 256K ctx decode 比 Qwen3-Max 快 **19×**
- **GLM-5**（2026-02）：744B-A40B，**MLA ＋ DSA 式稀疏同時用上**，200K ctx

> **注意「混合比例」已經變成一個新的超參數**：Qwen 3:1、Kimi K3 約 3:1（69:24）——大家收斂到差不多的比例，這本身就是訊號。

### 全景對照（第 14 頁）

| | ① 壓 KV | ② 少算（MoE） | ③ 少看 | ④ 一次多產 | ⑤ 降精度 |
|---|---|---|---|---|---|
| **DeepSeek** | MLA（−93.3%） | 671B-A37B 細粒度+共享 | DSA → CSA+HCA（1M：FLOPs 27%、KV 10%） | MTP 85–90% | FP8 訓練 |
| **Kimi** | MLA → Gated MLA | 1T-A32B → 2.8T-A104B | KDA 線性 ×69 + full ×24 | — | MXFP4 QAT |
| **MiniMax** | — | MoE | Lightning → 退回 full → MSA | — | — |
| **Qwen** | GQA | 397B-A17B | Gated DeltaNet : full ＝ 3:1 | MTP | — |
| **GLM** | MLA | 744B-A40B | DSA 式稀疏 | — | — |

> ⚠️ 以各家技術報告／官方部落格公開數字為準；2026 上半年版本迭代極快，**開講前請對一次官方頁面**。

---

## 為什麼他們連 kernel 都開源？（第 15 頁，全系列的接縫）

**困境**：新架構如果沒有 kernel，就沒有人跑得動。MLA 不是標準 attention，vLLM / SGLang 原本的 FlashAttention kernel 直接用不了；MoE 的 all-to-all、FP8 GEMM、專家負載均衡也一樣。**開源模型權重卻沒有配套 kernel，等於發布了一台沒有輪子的車。**

| 零件 | 是什麼 | 讓哪個旋鈕真的跑得動 |
|---|---|---|
| **FlashMLA** | MLA 的 decode kernel | ① 壓 KV |
| **DeepEP** | 專家平行的 all-to-all 通訊庫 | ② 少算（在多機可行；第四堂⑤） |
| **DeepGEMM** | FP8 GEMM | ⑤ 降精度（吃到 tensor core） |
| **EPLB** | 專家平行負載均衡器 | 專家熱點＝資料傾斜（第四堂⑤） |

> **開源 kernel 是讓自家架構進入 vLLM / SGLang 生態的手段——模型與框架是共生的，不是上下游。**
> 這一頁把第三、四、五堂縫起來：第三堂講框架怎麼榨單機、第四堂講怎麼榨叢集、第五堂講模型怎麼改自己**以及怎麼讓框架接得住**。

---

## 帶走三句話（第 16 頁）

1. **架構決策就是硬體帳單。** MLA 是為了 decode 的 HBM 頻寬、MoE 稀疏度是為了每 token 的權重讀取量、MXFP4 QAT 是為了對齊 Blackwell 的 tensor core。成本從第一天就寫在架構裡。
2. **五個旋鈕，一個目標。** 全都在回答「怎麼讓每產一個 token 少搬一點位元組」——第一堂 roofline 的分母，也是第三堂框架優化的另一半。
3. **理論上更省 ≠ 實際上更快。** MiniMax M2 退回 full attention 是 2025–26 最重要的負面結果，而**驗證它需要的算力，正是它想省下來的那些**。

> **全系列收束**：第一堂硬體 → 第二堂多卡 → 第三堂單機引擎 → 第四堂多機服務 → 第五堂模型架構。**同一個敵人，五個高度。**

---

## Q&A 速查

**Q：MoE 在單卡上不是不省記憶體嗎？稀疏化到底省什麼？**
A：單卡不省容量（專家都要在 HBM），省的是每 token 的權重讀取量與 FLOPs。要連容量也省得靠 EP（第四堂⑤），代價是 all-to-all 通訊與專家負載不均。

**Q：長 context 那麼貴，稀疏注意力是不是必然的未來？**
A：方向上是，但注意 MiniMax 的教訓：**線性**注意力目前在多跳推理與生態相容性上仍有實證問題；**稀疏**比較被接受，因為它不改變「KV 還在」這個前提。2026 的實務共識是**混合 + 稀疏**。

**Q：這些模型我在本機跑得動嗎？**
A：總參數是容量門檻、活躍參數是速度門檻——兩者都要看。K3 的 2.8T 即使 MXFP4 也要 ~1.4 TB 才裝得下權重，本機無望；但 Qwen3-Next 80B-A3B 這種「大總參數、小活躍」的設計，在統一記憶體機器（Apple M 系列大容量）上是可行的——這正好呼應第一堂的「容量夠、頻寬低 → 慢但跑得動」。

---

## 資料來源

- [DeepSeek-V3 技術報告](https://arxiv.org/abs/2412.19437)（MLA、DeepSeekMoE、無 aux loss 均衡、MTP、FP8、DualPipe）
- DeepSeek-V2（MLA −93.3%）｜DeepSeek-V3.2-Exp（DSA，2025-09，API 降價 >50%）｜[DeepSeek-V4](https://arxiv.org/abs/2606.19348)（CSA/HCA/mHC、1M ctx 下 FLOPs 27%、KV 10%）
- [Kimi K2 技術報告](https://arxiv.org/abs/2507.20534)（1T-A32B、MuonClip）｜[Kimi K3 模型卡](https://huggingface.co/moonshotai/Kimi-K3)（2.8T-A104B、69 KDA + 24 Gated MLA、896/16、MXFP4/MXFP8 QAT、1M ctx）
- [LMSYS：No Free Lunch — Deconstruct Efficient Attention with MiniMax M2](https://www.lmsys.org/blog/2025-11-04-miminmax-m2/)｜[MiniMax 官方說明](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model)｜[MiniMax-M1 論文](https://arxiv.org/abs/2506.13585)
- Qwen3 / Qwen3-Next / Qwen 3.5、GLM-5 —— 各家官方部落格；綜覽見 [Interconnects 開源模型整理](https://www.interconnects.ai/p/latest-open-artifacts-19-qwen-35)
- DeepSeek 開源週零件：FlashMLA / DeepEP / DeepGEMM / EPLB / 3FS（[open-infra-index](https://github.com/deepseek-ai/open-infra-index)）
- 前置：[第三堂](class3_sglang_single_node.md)（KV cache、prefix caching、投機解碼、量化）、[第四堂](class4_sglang_multi_node.md)（大規模 EP）
