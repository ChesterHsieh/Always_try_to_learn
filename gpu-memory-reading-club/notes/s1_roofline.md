# S1 講稿 — 為什麼會慢？Roofline 與記憶體階層

投影片：[../slides/s1_roofline.pptx](../slides/s1_roofline.pptx) ｜ Demo：[../demos/01_roofline_mini](../demos/01_roofline_mini)

> 講解節奏建議：理論 ~30 分鐘（slide 1–9）＋ demo 實跑 ~10 分鐘（slide 10）＋ 收束 5 分鐘（slide 11）。

## 一句話主旨

「慢」有兩種：**算不夠快（compute-bound）** 與 **資料搬不夠快（memory-bound）**。先學會用「算術強度」這把尺判斷自己在哪一種，後面三場（HBM、訓練 vs 推論、資料搬遷）才有共同語言。

## 各頁講解重點

**Slide 1 — 標題**：開門見山把全系列的核心問句丟出來：一個運算是被算力還是被記憶體頻寬卡住？右上角的折線就是今天的主角 roofline。

**Slide 2 — 開場謎題**：拋出反直覺事實——H100 跑 batch=1 解碼，tensor core 利用率常 <5%。讓聽眾先「不解」，製造張力。强調：問題不在算力，在搬運。（這也是 S3 的伏筆。）

**Slide 3 — 兩種「慢」**：給出兩個對立卡片。重點在「症狀」那行——**加大 batch 會不會變快**是現場最容易自我診斷的訊號：compute-bound 加 batch 不會更快，memory-bound 加 batch 吞吐會上升。

**Slide 4 — CPU vs GPU**：用核心數量的視覺差異解釋設計哲學。關鍵句：GPU 靠「同時跑很多」來藏記憶體延遲；但 thread 再多，資料餵不上就是空等。承接到「那要怎麼判斷資料餵不餵得上」。

**Slide 5 — 算術強度**：給定義。把 AI 唸成「每搬 1 個 byte，能換到幾次浮點運算」。這是全場的尺。

**Slide 6 — Roofline**：把尺畫成圖。斜線＝頻寬上限（你被頻寬限），平頂＝算力上限（你被算力限），轉折＝ridge point。資料點落在斜線上 → 換更貴的算力沒用。

**Slide 7 — Ridge point 範例**：用 H100 把抽象數字落地：≈300 FLOPs/Byte。强調這個門檻很高，很多日常運算根本到不了 → 注定 memory-bound。

**Slide 8 — 記憶體階層**：拉遠看整個階層。最該記住的對比：**HBM 是 TB/s、PCIe 是 GB/s，差約 100×**。為 S4「資料搬遷」鋪路。

**Slide 9 — 心法 + pipeline**：把階層連成一條搬運路徑，點出瓶頸＝最慢那段（常是 PCIe）。「memory 一進一出」這句要講清楚：HBM 內部再快，反覆跨 PCIe 就被拖死。

**Slide 10 — Demo**：現場跑 `01_roofline_mini`，讓聽眾親眼看到瘦長矩陣掉進 memory-bound、大方陣逼近算力上限。把抽象 roofline 變成終端機裡的數字。

**Slide 11 — 收束**：三句帶走。最後一句最重要：**很多 inference 的慢是頻寬問題、不是算力問題**——直接連到 S2/S3。

## 關鍵推導（撐住現場提問）

### 算術強度與 ridge point
- GEMM `C[M,N]=A[M,K]·B[K,N]`：FLOPs `= 2·M·K·N`；理想 Bytes `= s·(M·K + K·N + M·N)`（s = dtype 位元組）。
- 大方陣 `M=N=K=n`：AI `≈ 2n³ / (3·s·n²) = 2n/(3s)` → 隨 n 線性變大 → compute-bound。
- GEMV `M=1`：AI `≈ 2NK / (s·NK) = 2/s` → fp16 (s=2) 約 **1** → 永遠 memory-bound。
- ridge point `= 峰值算力 / 峰值頻寬`。H100：`990 TFLOPS / 3.35 TB/s ≈ 296 FLOPs/Byte`。

### 「為什麼 batch=1 解碼這麼慢」的粗估（S3 會展開）
- 每產 1 個 token，要把整份權重從 HBM 讀一遍。
- 7B 模型 fp16 ≈ 14 GB；`14 GB / 3.35 TB/s ≈ 4.2 ms/token` → 上限約 **~240 tok/s**（batch=1，只算讀權重）。
- 這就是 slide 2「<5% 利用率」的根源：時間都花在搬權重，算力閒置。

## 數字使用原則
- 投影片所有硬體數字標「約略值，以官方規格為準」。現場若被追問精確值，回到官方 datasheet。
- 記憶體階層頻寬以「數量級」呈現即可，重點是層與層之間 ~10× 的落差，不是精確數。

## 預期提問（Q&A 準備）
- **Q：那加大 batch 不就解決了？** A：對 throughput 有效（攤平權重讀取成本），但對單一請求的 latency 沒幫助；且 batch 受 KV cache 容量限制（S3 展開）。
- **Q：tiling／cache 不是能省頻寬？** A：能。tiling 把資料留在 shared memory 重複用，等於提高 AI、把運算往 roofline 右邊推（S2 展開）。
- **Q：unified memory 是不是就沒有搬運問題？** A：看架構。NVIDIA UVM 是「遷移式」仍會搬；Apple 統一記憶體是「零複製」共用同一塊（S4 展開）。
