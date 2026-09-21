// 讀書會複習題庫：六堂 × 8 題（辨識／邊界／遷移／取捨 各 2 題）。規格見 concept-check。
window.QUIZ = [
  // ───────────────────────────── 第一堂 ─────────────────────────────
  {
    id: 1, title: "第一堂 · 硬體 × Transformer", concept: "roofline 與記憶體階層",
    questions: [
      {
        level: "辨識", type: "concept", concept: "算術強度與 ridge point",
        q: "H100 峰值約 990 TFLOPS、HBM 頻寬約 3.35 TB/s。某個 kernel 每從 HBM 讀 1 byte 只做 1 次浮點運算。它的速度主要被什麼決定？",
        options: [
          { label: "Tensor Core 的峰值算力", desc: "990 TFLOPS 是這張卡能做運算的天花板，算得再快也不會超過它。", hint: "如果每讀 1 byte 能做幾百次運算，那確實是算力說了算。但題目裡這個 kernel 每 byte 只做 1 次運算，遠低於 H100 約 296 的轉折點。再想一次？" },
          { label: "L2 快取裝得下多少資料", desc: "資料若能留在 L2，就不必每次回 HBM 拿，容量決定命中率。", hint: "如果同一份資料會被重複用，L2 容量確實重要。但題目說的是每讀 1 byte 只用一次、做 1 次運算，沒有重複使用可以省。再想一次？" },
          { label: "HBM 能多快把資料送進來", desc: "資料送進來的速度跟不上時，運算單元只能閒著等。", hint: null },
          { label: "kernel launch 的 CPU 開銷", desc: "每次啟動 kernel 都要 CPU 下指令，小 kernel 會被這段時間主宰。", hint: "當 kernel 很小、跑幾微秒就結束時，launch 開銷確實可能是主角。但題目問的是這個 kernel 本身的運算和搬運比例，那才決定它在 roofline 上的位置。再想一次？" }
        ],
        answer: 2,
        explain: "算術強度 AI=1 FLOP/Byte，遠低於 ridge point（990 TFLOPS ÷ 3.35 TB/s ≈ 296）。所以它落在 roofline 的斜線段，是 memory-bound，速度等於 AI × 頻寬，算力上限最多用到約 0.34%。出處：第一堂第 10–13 頁（兩種慢、AI、roofline、ridge point）。",
        dig: "如果把同一個 kernel 搬到 A100（ridge point 較低），利用率上限會變高還是變低？為什麼「卡越強越浪費」？"
      },
      {
        level: "辨識", type: "concept", concept: "GPU / TPU / Groq 怎麼面對 decode 的記憶體牆",
        q: "在 GPU 和 TPU 上，decode 每一步都要把權重和 KV cache 從 HBM 串流進晶片。下面哪一種設計是直接把「從 HBM 串流」這條路拿掉？",
        options: [
          { label: "Groq LPU 全用片上 SRAM", desc: "晶片上不放 HBM，資料全部存在 SRAM，再用很多顆晶片切開模型。", hint: null },
          { label: "TPU 的 systolic array", desc: "權重釘在格子裡、資料流過去，XLA 事先把整張計算圖排好。", hint: "systolic array 讓矩陣乘法的資料流很有效率，這點沒錯。但 TPU 仍然掛著 HBM，decode 時權重與 KV 還是要從 HBM 串流進來。再想一次？" },
          { label: "FlashAttention 的 tiling", desc: "把 attention 切塊、線上 softmax，T×T 的矩陣不寫回 HBM。", hint: "FlashAttention 確實大幅減少 attention 中間矩陣進出 HBM 的量。但它只省下 S 矩陣那一塊，權重和 KV cache 仍然住在 HBM。再想一次？" },
          { label: "GQA 讓多個 head 共用 KV", desc: "KV head 從每個 query 一組改成共用，KV 位元組數大減。", hint: "GQA 讓要搬的 KV 變少，確實減輕了記憶體牆。但它只是讓那條路上的貨變少，貨還是從 HBM 送進來。再想一次？" }
        ],
        answer: 0,
        explain: "Groq 砍掉 HBM、全用 SRAM（每顆約 230 MB、約 80 TB/s），所以沒有那條 HBM 串流；代價是單顆容量很小，要幾十到幾百顆。FlashAttention、GQA 是在 HBM 上想辦法省搬運量，TPU 則是換一種運算陣列，三條路都在繞同一道記憶體牆。出處：第一堂第 27–28 頁。",
        dig: "Groq 用幾百顆晶片裝下一個模型之後，原本「HBM 頻寬」的瓶頸會轉移到哪裡？"
      },
      {
        level: "邊界", type: "debug", concept: "tiling 的上限",
        q: "同事把 naive matmul 改寫成 tiled 版本，tile 邊長從 16 一路加大。一開始越大越快，但加到某個大小之後反而突然變慢。最可能是哪個前提破了？",
        options: [
          { label: "tile 越大，總 FLOPs 就越多", desc: "大 tile 會做更多重複的乘加運算，額外的計算把好處吃掉了。", hint: "如果切塊方式會讓運算重複，那確實會多算。但 tiling 只改變資料被讀幾次，矩陣乘法本身的 FLOPs 不變。再想一次？" },
          { label: "HBM 頻寬被大 tile 吃滿", desc: "tile 一次搬進來的資料量太大，把 HBM 頻寬塞爆了。", hint: "如果每次搬運的總量變多，頻寬確實可能吃緊。但 tile 越大，每份資料被重複使用越多次，從 HBM 讀的總量反而變少。再想一次？" },
          { label: "Tensor Core 只吃 16×16 的形狀", desc: "硬體矩陣單元有固定形狀，tile 大於 16 就無法走 Tensor Core。", hint: "Tensor Core 一次確實只處理小塊（例如 16×16）。但大 tile 可以在 kernel 裡再拆成多個小塊餵進去，形狀不會卡住。再想一次？" },
          { label: "tile 裝不進 shared memory", desc: "tile 要留在 SM 內反覆使用，那塊空間很小。", hint: null }
        ],
        answer: 3,
        explain: "tiling 把 T×T 的小塊放進 shared memory 重複用 T 次，HBM 讀取量從 O(N³) 降到 O(N³/T)，所以 AI ≈ O(T)。但這條路的上限是 shared memory 容量；tile 裝不下時就會溢出或降低駐留的 warp 數，效益反轉。出處：第一堂第 15 頁（tiling）。",
        dig: "tile 太大時會同時壓縮每個 SM 能駐留的 warp 數。這和「用 warp 切換藏住記憶體延遲」有什麼衝突？"
      },
      {
        level: "邊界", type: "debug", concept: "FLOPs ≠ 速度",
        q: "把一個標準卷積換成 depthwise separable conv，FLOPs 降到約 1/9，但在 GPU 上實測只快了 1.5–3 倍。最可能的原因是什麼？",
        options: [
          { label: "新架構的精度比較差", desc: "depthwise 的表達力較弱，要多跑幾層才補得回來。", hint: "如果換了架構之後還得加層數補品質，那確實會抵消速度。但題目比較的是同一個網路在 GPU 上的實測時間，沒有另外加層。再想一次？" },
          { label: "depthwise 那一段 AI 很低", desc: "逐通道運算的資料重複使用很少，每搬一個 byte 只做幾次運算。", hint: null },
          { label: "GPU 不支援 depthwise 運算", desc: "硬體缺少對應指令，只能用 CUDA core 慢慢模擬。", hint: "如果硬體真的缺這種運算，那確實會很慢。但 depthwise 用 CUDA core 就能算，問題不在「能不能算」，而在它能不能把算力用起來。再想一次？" },
          { label: "kernel launch 次數變成兩倍", desc: "一層拆成 depthwise 和 pointwise 兩個 kernel，多了一次 launch。", hint: "拆成兩個 kernel 確實多一次 launch，對很小的層有影響。但這個開銷是微秒級，解釋不了「省了 9 倍 FLOPs 卻只快 1.5–3 倍」這麼大的落差。再想一次？" }
        ],
        answer: 1,
        explain: "depthwise 段的算術強度只有個位數，落在 roofline 斜線段，是 memory-bound；它省下的是紙上的 FLOPs，但搬運的 bytes 沒有等比例減少，所以牆上時間只快 1.5–3 倍。這是「FLOPs ÷9 ≠ 速度 ÷9」的經典例子。出處：第一堂講稿「共同演化」推導速查（第 29–32 頁 case）。",
        dig: "同樣的「FLOPs 省了、速度沒跟上」在 MoE 和線性注意力上各是怎麼發生的？"
      },
      {
        level: "遷移", type: "scenario", concept: "decode 延遲下限",
        q: "一張卡的記憶體頻寬 1 TB/s、峰值 300 TFLOPS。在上面跑 13B 參數的 fp16 模型，batch=1 做 decode。每產生一個 token，延遲的物理下限大約是多少？",
        options: [
          { label: "約 0.09 ms", desc: "每 token 約 26 GFLOP，除以 300 TFLOPS 的峰值算力。", hint: "如果這一步是 compute-bound，用 FLOPs 除以算力確實就是下限。但 batch=1 的 decode 每讀 1 byte 只做約 1 次運算，被卡住的不是算力。再想一次？" },
          { label: "約 26 ms", desc: "每一步都要把全部權重從記憶體讀過一遍。", hint: null },
          { label: "約 13 ms", desc: "13B 個參數，每個參數讀一次，除以 1 TB/s。", hint: "「每步把權重讀一遍」這個方向是對的。但題目說的是 fp16，每個參數不是 1 byte。再想一次？" },
          { label: "約 2.6 ms", desc: "只有被用到的那部分權重要讀，大約一成。", hint: "如果是 MoE 模型，每 token 確實只讀一部分權重。但題目是一般的 13B 模型，每一層每個權重都會參與運算。再想一次？" }
        ],
        answer: 1,
        explain: "batch=1 decode 每步要讀整份權重：13B × 2 bytes ≈ 26 GB，÷ 1 TB/s ≈ 26 ms，也就是約 38 tok/s。算力那條只要 0.09 ms，差了將近 300 倍，所以它是 memory-bound。同樣的算法，7B 在 H100 上是 14e9 ÷ 3.35e12 ≈ 4.2 ms。出處：第一堂第 18 頁（decode 解謎）與講稿推導速查。",
        dig: "如果把這個 13B 模型量化成 int4，同一張卡的延遲下限會變成多少？batch 拉到 64 時又會怎樣？"
      },
      {
        level: "遷移", type: "scenario", concept: "prefetch / overlap 的上限",
        q: "一條訓練管線從 SSD 載入每批資料要 50 ms，GPU 算一批只要 10 ms。你加上完美的 prefetch／overlap（搬下一批的同時算這一批）。穩定之後，每一批大約要多久？",
        options: [
          { label: "約 10 ms，只剩運算", desc: "搬運被完全藏在運算後面，只剩下運算時間。", hint: "當搬運比運算快的時候，overlap 確實能把搬運完全藏起來。但題目裡搬運是運算的 5 倍長，藏不進去的是哪一段？再想一次？" },
          { label: "約 30 ms", desc: "兩者平行進行，時間約是兩段的平均。", hint: "平行進行確實會縮短總時間。但兩條管線同時跑時，節奏是由比較慢的那一條決定，不是平均。再想一次？" },
          { label: "約 60 ms，兩段相加", desc: "搬運加上運算，overlap 只是換個順序。", hint: "沒有 overlap 時，每批確實是兩段相加。但題目說加上了完美的 prefetch，兩段已經可以同時進行。再想一次？" },
          { label: "約 50 ms", desc: "兩段同時進行，整體節奏跟著較慢的那段走。", hint: null }
        ],
        answer: 3,
        explain: "完美 overlap 的總時間 ≈ C + N·max(C, K)，所以穩態每批 ≈ max(50, 10) = 50 ms，被搬運決定。overlap 在 C≈K 時最多約 2 倍；C≫K 時上限由 C 決定，這時該做的是讓搬運變快（換更快的路、pinned memory、減少位元組）。出處：第一堂第 21 頁（prefetch／overlap 壓軸）。",
        dig: "如果改成 C=10 ms、K=50 ms，overlap 還值得做嗎？這時瓶頸換成誰？"
      },
      {
        level: "取捨", type: "numeric", concept: "Amdahl 定律",
        q: "一段程式有 95% 可以平行、5% 必須序列執行。把它丟到 16,896 個核心上，理想加速大約是幾倍？",
        options: [
          { label: "約 20 倍", desc: "序列那 5% 無法被分攤，無論核心多少都跑不掉。", hint: null },
          { label: "約 200 倍", desc: "平行部分加速很多，但序列部分拖慢一個量級。", hint: "序列部分確實會把加速拖下來。但把數字代進去算：就算平行部分縮到接近 0，剩下的 5% 本身已經定出上限。再想一次？" },
          { label: "約 950 倍", desc: "16,896 個核心乘上 95% 可平行的比例，再扣掉一些開銷。", hint: "如果加速是「核心數 × 可平行比例」，那確實會很大。但 Amdahl 是用時間相加來算，序列部分的時間不會因為核心變多而縮短。再想一次？" },
          { label: "約 16,896 倍", desc: "每個核心分到一份工作，理想上是線性加速。", hint: "如果程式 100% 可平行，理想加速確實等於核心數。但題目有 5% 必須序列，那一段只能由一個核心做。再想一次？" }
        ],
        answer: 0,
        explain: "S(N) = 1 / ((1−p) + p/N) = 1 / (0.05 + 0.95/16896) ≈ 20，幾乎等於 N=∞ 時的 1/0.05。核心夠多之後，瓶頸只剩序列那段；在模型裡，序列相依就是 RNN 的 T 步遞迴、Transformer decode 的逐 token 生成。出處：第一堂第 6 頁（餵飽一張卡 + Amdahl）。",
        dig: "Transformer 訓練時把 RNN 的序列鏈拿掉了，為什麼 decode 時 (1−p) 又回來了？Mamba 又是怎麼處理的？"
      },
      {
        level: "取捨", type: "tradeoff", concept: "batch 換到什麼、換不到什麼",
        q: "decode 是 memory-bound，所以有人提議「把 batch 從 1 開到 64」。這樣做換到的和換不到的，最準確的描述是？",
        options: [
          { label: "延遲與吞吐會一起變好", desc: "權重讀一次服務更多請求，每個請求也跟著變快。", hint: "權重讀一次服務多個請求，這確實讓總產量上升。但每一步的時間幾乎沒變，單一請求每個 token 還是要等那麼久。再想一次？" },
          { label: "單請求延遲下降、吞吐持平", desc: "每個請求分到的權重讀取變少，所以每步更快。", hint: "如果每個請求真的少讀權重，那延遲確實會降。但 batch 裡所有請求共用同一次權重讀取，每一步仍要讀完整份權重。再想一次？" },
          { label: "吞吐上升，單請求延遲不降", desc: "每步時間幾乎不變，但一步產出 64 個 token。", hint: null },
          { label: "吞吐上升，KV 需求不會變", desc: "權重共用一份，所以記憶體用量跟 batch=1 一樣。", hint: "權重確實只放一份、共用。但每個請求都有自己的 KV cache，batch 越大 KV 佔的 HBM 也越多。再想一次？" }
        ],
        answer: 2,
        explain: "單步延遲 ≈ max(權重/頻寬, 2·params·batch/算力)；batch 小時由第一項主宰，所以開大 batch 讓 tokens/s 幾乎線性上升，但單請求延遲不變（throughput↑ ≠ latency↓）。而且 batch 會被 KV cache 容量卡住，超過 ridge（約數百）後還會轉成 compute-bound。出處：第一堂第 20 頁（batch）與 Q&A 速查。",
        dig: "如果目標是降低單一使用者的延遲而不是提高吞吐，講稿提到的哪兩種方法有效？它們各自是在打分子還是分母？"
      }
    ]
  },

  // ───────────────────────────── 第二堂 ─────────────────────────────
  {
    id: 2, title: "第二堂 · Transformer × GPU", concept: "Transformer 逐 block 上機與多卡平行",
    questions: [
      {
        level: "辨識", type: "concept", concept: "哪個 block 是 memory-bound",
        q: "在 prefill（一次處理整段 prompt）時，下面哪一個 block 仍然是 memory-bound、主要跑在 CUDA core 上？",
        options: [
          { label: "QKV 投影", desc: "輸入乘上 Wq、Wk、Wv 三個權重矩陣。", hint: "QKV 投影確實要讀三個權重矩陣。但在 prefill 時 T 很大，它是一個大 GEMM，每份資料會被重複使用很多次。再想一次？" },
          { label: "Add & LayerNorm", desc: "逐元素相加，再沿著特徵維度做平均與變異數的歸約。", hint: null },
          { label: "FFN 的兩個大矩陣乘", desc: "先放大到 4 倍寬、再縮回來，是一層裡最大的權重。", hint: "FFN 確實是一層裡權重最多的地方。但權重多不等於被頻寬卡住：prefill 時它是兩個大 GEMM，每個權重會被整段 token 重複使用。再想一次？" },
          { label: "Attention 裡的 Q·Kᵀ", desc: "每個 query 和所有 key 做內積，產生 T×T 的分數矩陣。", hint: "在 decode 時，attention 的矩陣乘確實會退化成 memory-bound。但題目說的是 prefill，整段 Q 一起乘 Kᵀ，是一個矩陣乘法。再想一次？" }
        ],
        answer: 1,
        explain: "Add & LayerNorm 是逐元素加法加上歸約，每個元素只做幾次運算，卻要把整份 activation 從 HBM 讀進來再寫回去，所以在 CUDA core 上是 memory-bound，常用 kernel fusion 減少進出次數。QKV、FFN 是大 GEMM，跑在 Tensor Core 上是 compute-bound。出處：第二堂第 9 頁與第 11 頁彙整表。",
        dig: "kernel fusion 把 LayerNorm 和前後的運算合在一起，省下的是 FLOPs 還是 bytes？用 roofline 說明它往哪個方向移。"
      },
      {
        level: "辨識", type: "concept", concept: "四種平行各切什麼",
        q: "有一種平行方式，每一層在 attention 輸出和 FFN 輸出各要做一次 all-reduce，一次 forward 就要數十到上百次，所以只能待在 NVLink 域裡。這是哪一種？",
        options: [
          { label: "DP 資料平行", desc: "每張卡放整份模型，各自吃一部分 batch。", hint: "DP 確實也要 all-reduce。但它是每一步（整個 batch 算完）才同步一次梯度，不是每一層都同步，所以可以跨節點。再想一次？" },
          { label: "PP 管線平行（層間）", desc: "按層切成幾段 stage，每段放在不同的卡上。", hint: "PP 確實把模型切到多張卡上。但它只在 stage 邊界用 p2p 傳 activation，通訊很稀疏，可以跨節點。再想一次？" },
          { label: "EP 專家平行（MoE）", desc: "MoE 的專家散到不同卡，token 用 all-to-all 路由過去。", hint: "EP 的通訊量確實很大，也偏好留在 scale-up。但它的通訊型態是 all-to-all 路由，不是每層兩次 all-reduce。再想一次？" },
          { label: "TP 張量平行", desc: "把每一層的矩陣切成幾片，各卡算一片。", hint: null }
        ],
        answer: 3,
        explain: "TP 把層內的權重矩陣切片，每層算完要把部分和加起來，attention 與 FFN 輸出各一次 all-reduce，通訊極密，走 PCIe 或跨節點 IB 就會被頻寬拖死，所以必須在 NVLink 域內。DP 每步同步梯度、PP 只在 stage 邊界 p2p、EP 是 all-to-all。出處：第二堂第 16–17 頁與講稿「四種平行」速查表。",
        dig: "為什麼 TP 在一層裡剛好是「兩次」all-reduce？Megatron 式的切法是怎麼把 FFN 的兩個矩陣配對，省掉中間那次同步的？"
      },
      {
        level: "邊界", type: "debug", concept: "DP 救不了裝不下的模型",
        q: "團隊有一個 70B 參數的 fp16 模型（權重約 140 GB），手上是 8 張 80 GB 的卡。他們決定用 8 路資料平行（DP）來跑推論。會發生什麼事？",
        options: [
          { label: "每張卡還是放不下整份模型", desc: "DP 讓每張卡都存一份完整的權重，模型本身沒有變小。", hint: null },
          { label: "每張卡放 1/8，順利跑起來", desc: "8 張卡總共 640 GB，平均分下去每張只要約 18 GB。", hint: "如果把模型本身切開，每張卡確實只要放一部分。但題目選的是資料平行，它切的是 batch，不是模型。再想一次？" },
          { label: "跑得起來，但梯度同步很慢", desc: "每步都要把梯度 all-reduce，通訊成為瓶頸。", hint: "在訓練時，梯度同步確實是 DP 的主要成本。但題目是推論，而且在同步之前，還有一個更早就會卡住的問題。再想一次？" },
          { label: "跑得起來，但只有吞吐變高", desc: "DP 只增加同時服務的請求數，單請求延遲不變。", hint: "「DP 只增吞吐、不降延遲」這個說法本身是對的。但前提是每張卡要先把模型放得下，題目的數字過不了這一關。再想一次？" }
        ],
        answer: 0,
        explain: "DP 的第一個問題就是「模型沒變小」：每張卡都要放整份權重，140 GB 的模型放不進 80 GB 的卡，DP 開幾路都沒用。要讓每張卡放得下，得切模型本身（TP／PP／EP）。這是第二堂所說 DP 四個問題的第 ①。出處：第二堂第 13–15 頁。",
        dig: "如果把 DP 換成 ZeRO／FSDP，70B 模型在同樣 8 張卡上能不能訓練？它是用什麼換來容量的？"
      },
      {
        level: "邊界", type: "debug", concept: "decode 讓 GEMM 退化成 GEMV",
        q: "prefill 時，QKV 投影和 attention 的矩陣乘都跑在 Tensor Core 上、是 compute-bound。換到 decode 階段，同樣的程式碼卻變成 memory-bound。是哪個前提改變了？",
        options: [
          { label: "decode 改用 CUDA core 計算", desc: "decode 的運算量小，框架會改派到比較簡單的單元。", hint: "運算單元的選擇確實會影響速度。但即使繼續用 Tensor Core，這一步的問題也不在運算單元，而在資料能被重複用幾次。再想一次？" },
          { label: "KV cache 長到裝不進 HBM", desc: "序列越長 KV 越大，超出容量就要從別處搬。", hint: "KV cache 確實會隨序列長度變大、最終撐爆容量。但題目的現象在短序列時就會出現，跟容量無關。再想一次？" },
          { label: "每步只剩 1 個新 token", desc: "矩陣乘的 T 維縮成 1，GEMM 退化成矩陣乘向量。", hint: null },
          { label: "權重在 decode 時要重新載入", desc: "每一步都把權重從 CPU 記憶體搬回 GPU。", hint: "每一步確實都要讀一遍權重。但權重一直住在 HBM，不是從 CPU 搬回來；真正的變化在於讀進來之後能被用幾次。再想一次？" }
        ],
        answer: 2,
        explain: "decode 每步只處理 1 個新 token，X·W 從 [T, d] × [d, d] 退化成 [1, d] × [d, d] 的 GEMV，每讀一個權重只用一次，AI ≈ 1。於是連原本吃算力的 QKV、FFN、attention 也全部變成 memory-bound，這就是第一堂「<5% 利用率之謎」在逐 block 層級的樣子。出處：第二堂第 8 頁與講稿 Part A 速查。",
        dig: "server 端用 continuous batching 把很多請求的 decode 併在一起，GEMV 會變回什麼？attention 那一段能不能一起被救回來？"
      },
      {
        level: "遷移", type: "scenario", concept: "PP 的 bubble",
        q: "你把一個模型用管線平行（PP）切成 8 段放到 8 張卡上，每個 step 只餵 2 個 micro-batch。profiler 顯示大部分卡有一大半時間在閒置。最先該調整什麼？",
        options: [
          { label: "把 stage 之間的線換成 NVLink", desc: "stage 間的傳輸變快，下一段就不用等那麼久。", hint: "如果 stage 間的傳輸真的很慢，換快的線確實有幫助。但 PP 只在邊界傳 activation，通訊量不大；卡在閒置的主因是排程結構。再想一次？" },
          { label: "再加一份 DP 副本分攤流量", desc: "多一份副本就能同時處理更多 batch。", hint: "加 DP 副本確實會增加總吞吐。但每一份 PP 管線內部的閒置比例並不會因此改變。再想一次？" },
          { label: "開啟 SHARP 在交換器內歸約", desc: "把集合通訊的加總搬進交換器 ASIC 處理。", hint: "SHARP 確實能加速 all-reduce。但 PP 的 stage 之間是點對點傳遞，沒有需要被歸約的集合通訊。再想一次？" },
          { label: "把 micro-batch 切得更多", desc: "讓管線裡同時有更多份工作在流動。", hint: null }
        ],
        answer: 3,
        explain: "PP 有 P 個 stage、M 個 micro-batch 時，理想利用率 ≈ M/(M+P−1)。P=8、M=2 時只有 2/9 ≈ 22%，bubble 佔了近八成。要把 bubble 壓小，就要增加 micro-batch 數（例如 M=32 時約 82%）。出處：第二堂講稿「關鍵推導速查：PP 的 bubble」。",
        dig: "micro-batch 切得越多，每一份就越小。切到太小時，會在第一堂的 roofline 上碰到什麼新問題？"
      },
      {
        level: "遷移", type: "scenario", concept: "平行策略落位",
        q: "你有 2 台 8 卡 H100 節點：機內 8 卡由 NVLink 相連，兩台之間走 InfiniBand。要跑 TP=8 × PP=2 的配置，該怎麼擺？",
        options: [
          { label: "TP 留機內，PP 跨機", desc: "每台一段 stage，段內 8 卡走 NVLink。", hint: null },
          { label: "TP 跨兩台，PP 放在機內", desc: "每台放 4 張 TP 卡，兩段 stage 都在同一台裡完成。", hint: "如果兩台之間的頻寬和機內一樣快，怎麼擺都行。但題目裡機間走 IB，比 NVLink 慢一個數量級，要想想哪種通訊最密。再想一次？" },
          { label: "兩種都跨機，平均分攤流量", desc: "讓 NVLink 和 IB 各承擔一半的通訊，避免單點擁塞。", hint: "分攤流量在頻寬相近的網路上確實有道理。但這裡兩條線差一個數量級，TP 的通訊只要有一部分走慢線，整層就會等它。再想一次？" },
          { label: "怎麼擺都一樣，看總頻寬", desc: "總頻寬固定，擺法只影響哪條線忙、不影響總時間。", hint: "如果每種平行的通訊量都差不多，那確實只看總頻寬。但 TP 每層兩次 all-reduce、PP 只在 stage 邊界傳一次，兩者差很多。再想一次？" }
        ],
        answer: 0,
        explain: "TP 每層兩次 all-reduce、通訊最密，必須留在 NVLink（scale-up）域；PP 只在 stage 邊界 p2p 傳 activation，可以容忍較慢的 IB（scale-out）。所以是「通訊密的留機架內、稀的才跨節點」，這是被頻寬逼出來的，不是選擇。出處：第二堂第 17 頁、第 21 頁。",
        dig: "如果換成 GB200 NVL72（72 卡同一個 NVLink 域），同樣的模型你會怎麼重新安排 TP、PP、EP？"
      },
      {
        level: "取捨", type: "numeric", concept: "DP 的 all-reduce 成本",
        q: "用 DP 訓練一個 7B 參數、fp16 梯度的模型，採用 ring all-reduce、卡數很多。每一步每張卡大約要搬多少梯度資料？如果 batch 加倍又會怎樣？",
        options: [
          { label: "約 14 GB，且隨 batch 加倍", desc: "梯度大小等於參數量，batch 越大要同步的梯度越多。", hint: "梯度大小確實和參數量一樣是 14 GB。但每張卡在同步前已經把自己 batch 的梯度加總成一份，batch 大小不影響那份的大小；ring all-reduce 本身還有一個倍數。再想一次？" },
          { label: "約 7 GB，與 batch 無關", desc: "ring all-reduce 把資料切段輪流傳，每張卡只負責一半。", hint: "「與 batch 無關」這一半說對了。但 ring all-reduce 要先做 reduce-scatter、再做 all-gather，每張卡送出的量不是變少。再想一次？" },
          { label: "約 28 GB，與 batch 無關", desc: "約 2 倍的參數位元組，只跟模型大小有關。", hint: null },
          { label: "約 28 GB，也隨 batch 加倍", desc: "2 倍參數量是基準，再乘上 batch 的倍數。", hint: "「約 2 倍參數位元組」這個數字是對的。但同步的是每張卡加總後的梯度，大小和 batch 無關。再想一次？" }
        ],
        answer: 2,
        explain: "ring all-reduce 每步每卡的通訊量 ≈ 2×(N−1)/N × 參數位元組；7B × 2 bytes = 14 GB，N 很大時約 28 GB。它與 batch 無關、與參數量成正比，所以模型越大、同步越貴，這就是 DP 第 ② 個問題「通訊隨規模長大」。出處：第二堂第 15 頁與講稿推導速查。",
        dig: "SHARP 把 all-reduce 的加總搬進交換器之後，這 28 GB 的通訊量會怎麼變？GPU 的 SM 又省下了什麼？"
      },
      {
        level: "取捨", type: "tradeoff", concept: "ZeRO / FSDP 的交換",
        q: "DP 的第 ③ 個問題是權重、梯度、optimizer 狀態在每張卡上各存一份。ZeRO／FSDP 把這些狀態分片到各卡。它換到了什麼、付出了什麼？",
        options: [
          { label: "省下通訊，但每卡容量變多", desc: "狀態分片後各卡只同步自己那份，但要預留更多空間。", hint: "分片之後，每張卡確實只管自己那一份。但要用到完整權重時得先把其他卡的分片收回來，通訊反而增加。再想一次？" },
          { label: "省下每卡容量，換更多通訊", desc: "每卡只存一片，用時再 all-gather。", hint: null },
          { label: "省下容量，也順便省下通訊", desc: "資料少了，同步的東西自然變少，一舉兩得。", hint: "每卡存的東西確實變少了。但每次 forward／backward 都要把分片 all-gather 回來，這些通訊是新增加的。再想一次？" },
          { label: "降低單請求 decode 延遲", desc: "每卡負擔變輕，推論時每一步跟著變快。", hint: "每張卡的負擔確實變輕了。但 decode 延遲取決於每步要搬的權重與 KV ÷ 頻寬，分片還得先把權重收回來，延遲不會因此下降。再想一次？" }
        ],
        answer: 1,
        explain: "ZeRO／FSDP 是「切狀態的 DP」：資料平行的骨架，但把權重、梯度、optimizer 狀態分片到各卡，用到時再 all-gather，所以省容量、換更多通訊，介於 DP 與模型平行之間。出處：第二堂第 15 頁與 Q&A「ZeRO / FSDP 算 DP 還是切模型？」。",
        dig: "訓練狀態約是參數量的 3–4 倍。把 70B 模型的狀態分到 64 張卡上，每張卡大約省下多少 GB？用什麼換？"
      }
    ]
  },

  // ───────────────────────────── 第三堂 ─────────────────────────────
  {
    id: 3, title: "第三堂 · 推論引擎單機篇", concept: "SGLang × vLLM 單機四個問題",
    questions: [
      {
        level: "辨識", type: "concept", concept: "分頁與 radix tree 是兩層",
        q: "網路文章常寫「SGLang 用 RadixAttention 取代了 vLLM 的 PagedAttention」。這個說法錯在哪裡？",
        options: [
          { label: "SGLang 其實沒有前綴快取", desc: "RadixAttention 只是行銷名稱，底下是一般的 KV cache。", hint: "如果 SGLang 沒做前綴快取，這個說法確實整個不成立。但 radix tree 是實際在跑的索引結構，而且會主動影響排程。再想一次？" },
          { label: "vLLM 已經改用 radix tree", desc: "vLLM V1 把 APC 換成前綴樹，兩家現在做法一樣。", hint: "兩家功能確實在趨同。但 vLLM V1 的前綴快取仍是鏈式雜湊表，命中要對齊 16-token 的 block 邊界。再想一次？" },
          { label: "PagedAttention 比 Radix 新", desc: "分頁是較晚才出現的技術，是 radix 的升級版本。", hint: "如果兩者是同一層的新舊版本，比較誰新確實有意義。但問題不在時間先後，而在它們各自回答的是不同的問題。再想一次？" },
          { label: "兩者分屬配置層與索引層", desc: "一個管 KV 怎麼放，一個管放好的 block 怎麼被找到。", hint: null }
        ],
        answer: 3,
        explain: "分頁（固定 16-token block、block table）是「KV 怎麼放」的記憶體配置層，兩家一致；radix tree 與鏈式雜湊表是「放好的 block 怎麼被找到」的索引層，兩家不同。把它們寫成競品，是把兩層混為一談。出處：第三堂第 11–12 頁（共同地基：分頁；對比②a 索引層）。",
        dig: "radix tree 的分叉點可以落在任意 token，雜湊表必須對齊 16-token block。在什麼樣的流量下，這個差異會造成明顯的命中率落差？"
      },
      {
        level: "辨識", type: "concept", concept: "jump-forward decoding",
        q: "在前綴共用重的結構化輸出場景，SGLang 的吞吐約是 vLLM 的 3 倍。這個差距主要來自哪裡？",
        options: [
          { label: "SGLang 自研的語法引擎更快", desc: "SGLang 有自己的 FSM 編譯器，vLLM 用的是外掛後端。", hint: "如果兩家用的語法引擎不同，這確實可能是原因。但兩家的預設後端都是 XGrammar，差異不在引擎本身。再想一次？" },
          { label: "確定的 token 直接吐出", desc: "schema 規定好、只有一種可能的片段不必問模型。", hint: null },
          { label: "FSM 改在背景非同步編譯", desc: "請求等編譯完成時不會卡住其他請求。", hint: "非同步編譯確實能避免阻塞，但這是 vLLM 那一側的做法，讓兩家追平而不是拉開差距。再想一次？" },
          { label: "FSM 狀態存進 radix tree 共用", desc: "同前綴的請求直接重用別人算好的語法狀態。", hint: "前綴 KV 確實可以跨請求共用。但語法狀態屬於每個請求自己，必須各自推進，不能共用。再想一次？" }
        ],
        answer: 1,
        explain: "jump-forward decoding 在同一個 FSM 上多走幾步：像 {\"name\": \" 這種在 schema 下唯一確定的片段，直接吐出、0 次推理，省下的時間拿去服務別的請求。兩家共同地基都是 XGrammar 的 FSM + 位元遮罩，差異就在這一步。出處：第三堂第 18–19 頁。",
        dig: "為什麼說「約束不是成本，是資訊」？加了語法約束的生成有可能比自由生成更快，前提是什麼？"
      },
      {
        level: "邊界", type: "debug", concept: "前綴匹配只認共同開頭",
        q: "某客服系統為了「讓問題更顯眼」，把使用者問題放在最前面、2,000 token 的 system prompt 接在後面。上線後前綴快取的命中率掉到接近 0。為什麼？",
        options: [
          { label: "system prompt 太長，被 LRU 淘汰", desc: "長段落佔用太多 block，會最先被趕出快取。", hint: "快取壓力大時，長段落確實可能被淘汰。但這裡就算快取空間很充足，命中率還是會接近 0。再想一次？" },
          { label: "雜湊表只能對齊 16-token 邊界", desc: "不同問題的長度不一，system prompt 對不齊 block。", hint: "vLLM 的命中確實必須對齊 block 邊界。但換成可以在任意 token 分叉的 radix tree，這個系統的命中率一樣接近 0。再想一次？" },
          { label: "匹配只認從頭開始的共同段", desc: "每個 token 的 K/V 都依賴它前面所有的 token。", hint: null },
          { label: "每個請求要各自推進自己的 FSM", desc: "請求狀態不可共享，所以快取也無法共用。", hint: "FSM 狀態確實不可共享，這是另一條規則。但它管的是語法狀態，不影響 KV 能不能被複用。再想一次？" }
        ],
        answer: 2,
        explain: "前綴匹配是嚴格按 token 順序從頭比對；第 n 個 token 的 K/V 依賴前面所有 token，所以只有「位置一樣、前文一樣」才能共用。問題放前面，每個請求從第一個 token 就不同，後面相同的 system prompt 也無法命中。出處：第三堂第 12 頁「前綴匹配的常見誤解」。",
        dig: "RAG 系統把檢索到的文件和使用者問題串接成 prompt。為了讓前綴快取命中率最高，這些片段應該怎麼排序？"
      },
      {
        level: "邊界", type: "debug", concept: "可快取的是計算結果，不是請求狀態",
        q: "兩個請求共享同一段前綴 KV，也都要求輸出同一種 JSON schema。工程師想讓它們「順便共用 FSM 狀態」以省下語法檢查。這樣做的問題是什麼？",
        options: [
          { label: "FSM 狀態跟著各自輸出走", desc: "兩個請求吐的 token 不同，語法位置就不同。", hint: null },
          { label: "FSM 太大，放不進 HBM", desc: "預編譯的狀態機會佔掉原本要給 KV 的記憶體。", hint: "如果 FSM 真的很大，容量確實會有壓力。但就算容量完全不是問題，兩個請求共用同一份 FSM 狀態仍然會出錯。再想一次？" },
          { label: "XGrammar 不支援多請求", desc: "語法後端一次只能處理一個請求的遮罩。", hint: "如果後端有這種限制，那確實共用不了。但 XGrammar 是三家引擎的預設後端，本來就同時服務大量請求。再想一次？" },
          { label: "會讓前綴快取的命中率下降", desc: "語法狀態一綁上去，KV 就只能給同一種 schema 用。", hint: "如果共用語法狀態會污染 KV 的鍵，那確實會影響命中。但真正的問題更早出現：這兩個請求從前綴結束後就各自生成不同的內容。再想一次？" }
        ],
        answer: 0,
        explain: "前綴複用省的是算力（可共享的計算結果）；約束生成裁剪的是候選空間，FSM 狀態屬於請求本身，每個請求依自己吐出的 token 各自推進。紅線：就算共享前綴快取，FSM 狀態仍必須獨立。通用判準是「可以快取計算結果，不可以快取請求狀態」。出處：第三堂第 20 頁。",
        dig: "第四堂做跨機 KV 複製時，這條判準怎麼用？哪些東西可以跟著 KV 複製過去，哪些不行？"
      },
      {
        level: "遷移", type: "scenario", concept: "選型看流量長相",
        q: "你的公司用 AMD MI300X 叢集，主要工作是對上萬篇互不相干的文件做一次性摘要，每篇的 prompt 幾乎沒有共同前綴。SGLang 和 vLLM 該怎麼選？",
        options: [
          { label: "vLLM：硬體支援面比較廣", desc: "前綴各自獨立時兩家吞吐差距很小，選生態成熟的那家。", hint: null },
          { label: "SGLang：radix tree 命中率高", desc: "前綴樹能找到任意長度的共同開頭，快取效果最好。", hint: "在前綴共用很重的流量上，radix tree 確實優勢明顯。但題目說文件互不相干、幾乎沒有共同前綴，這個優勢發揮不出來。再想一次？" },
          { label: "SGLang：jump-forward 快 3 倍", desc: "跳過確定的 token，整體輸出吞吐大幅提升。", hint: "jump-forward 在結構化輸出上確實很強。但題目是自由文字的摘要，沒有 schema 規定哪些 token 是唯一確定的。再想一次？" },
          { label: "看最新 benchmark 排名決定", desc: "兩家差異會隨版本改變，以最新數字為準最客觀。", hint: "兩家功能確實一直在趨同，數字會變。但講稿的結論正是不要拿某一次 benchmark 當永久答案，而要看自己的流量長相和部署條件。再想一次？" }
        ],
        answer: 0,
        explain: "前綴各自獨立時，兩家吞吐差 <5%，這時看生態：非 NVIDIA 硬體（AMD／TPU／Gaudi／CPU）vLLM 覆蓋明顯更廣。SGLang 的優勢（radix、cache-aware 排程、jump-forward）要在前綴共用 >60% 或結構化輸出時才會拉開。出處：第三堂第 15 頁與第 27 頁帶走三句話。",
        dig: "如果同一家公司下個月改做多輪對話的 agent 服務，硬體不變，這個選擇會不會翻轉？要先量哪個數字？"
      },
      {
        level: "遷移", type: "scenario", concept: "本機推論的天花板",
        q: "你在自己的 RTX 4090 上用 8B 模型聊天，只有你一個人用。想讓回應變快，下面哪一個方向最有效？",
        options: [
          { label: "開啟 continuous batching", desc: "每次 forward 都把新到的請求併進來一起算。", hint: "在多人使用的伺服器上，continuous batching 確實是撐大 batch 的關鍵。但題目只有你一個人用，沒有別的請求可以併進來。再想一次？" },
          { label: "換成 cache-aware 排程", desc: "把同前綴的請求排在一起，提高快取命中率。", hint: "cache-aware 排程在大量請求共用前綴時很有效。但單人使用時只有一條請求流，沒有東西可以重新排序。再想一次？" },
          { label: "量化或投機解碼", desc: "一個減少要搬的位元組，一個用閒置算力一次驗多個 token。", hint: null },
          { label: "換成 zero-overhead 排程器", desc: "把 CPU 排程藏進上一步 GPU 的執行時間裡。", hint: "當 GPU 每步只要 5–10 ms 時，CPU 開銷確實會浮出來。但單人 batch=1 時，每一步主要是在等權重從記憶體讀進來，那才是大頭。再想一次？" }
        ],
        answer: 2,
        explain: "本機 batch 幾乎恆等於 1，利用率約 0.3%，買的算力 99% 在閒置。這時能加速的只有天花板那兩條路：量化（砍掉要搬的位元組）和投機解碼（權重讀一次、驗 k 個 token，AI 從 1 變成 k）。這也是 llama.cpp／MLX 的賣點永遠是量化格式的原因。出處：第三堂第 24–25 頁與 Q&A「這些跟我只有一張 4090 有關嗎？」。",
        dig: "投機解碼的收益跟 draft 的接受率有關。在本機上，n-gram draft 和小模型 draft 各適合什麼樣的對話內容？"
      },
      {
        level: "取捨", type: "numeric", concept: "投機解碼的期望接受數",
        q: "投機解碼的草稿長度 k=4、每個草稿 token 的接受率 α=0.8。平均每一步 forward 大約能產出幾個 token？",
        options: [
          { label: "約 1.8 個", desc: "第一個 token 必定產出，後面約 0.8 個被接受。", hint: "如果只驗 1 個草稿 token，這確實接近答案。但題目一次驗 4 個草稿，每多一個被接受就多一個 token。再想一次？" },
          { label: "約 3.4 個", desc: "連續被接受的機率一路相乘，再加上驗證那一步自己的 token。", hint: null },
          { label: "約 4.0 個", desc: "4 個草稿 × 0.8 接受率，加上一點點修正。", hint: "直接相乘的想法很直覺。但草稿是一個接一個驗的，第 2 個要先接受第 1 個才算，機率是連乘，不是相加。再想一次？" },
          { label: "約 5.0 個（全部接受）", desc: "4 個草稿全過，再加上驗證時多出的那個 token。", hint: "5 個確實是這組參數的上限。但接受率 0.8 表示每個草稿都有機會被拒，平均不會每次都跑滿。再想一次？" }
        ],
        answer: 1,
        explain: "期望接受數 = (1 − α^(k+1)) / (1 − α) = (1 − 0.8⁵) / 0.2 ≈ 3.4。權重只讀一次、FLOPs 變 k 倍，AI 從 1 變 k，是唯一能在不增加 batch 的前提下改善單請求延遲的招式，而且 rejection sampling 保證輸出分佈不變。出處：第三堂第 24 頁（天花板① 投機解碼／MTP）。",
        dig: "DeepSeek-V3 的 MTP head 第二 token 接受率約 85–90%。把 α 從 0.8 換成 0.9，k=4 時期望接受數變成多少？為什麼 α 對結果的影響是非線性的？"
      },
      {
        level: "取捨", type: "tradeoff", concept: "batch 不能無限開大",
        q: "既然 decode 的 AI ≈ B（batch），為什麼生產環境不乾脆把 batch 無限開大？",
        options: [
          { label: "batch 越大，AI 反而會下降", desc: "每個請求要讀自己的 KV，batch 大時讀取量增加得比運算快。", hint: "每個請求的 KV 確實會增加讀取量，長序列時這點很重要。但講稿給的主要理由是另外三個，而且其中一個跟 ridge point 直接相關。再想一次？" },
          { label: "權重要被讀 B 次", desc: "每個請求都要讀一遍自己的權重，batch 越大搬越多。", hint: "如果每個請求各讀一次權重，batch 確實毫無幫助。但 batch 的全部意義就在於權重讀一次、B 個請求共用。再想一次？" },
          { label: "continuous batching 不支援", desc: "排程器只能在固定大小的 batch 上運作。", hint: "早期的靜態 batching 確實有固定大小的限制。但 continuous batching 正是以一次 forward 為單位動態調整，不是它擋住了 batch。再想一次？" },
          { label: "過了 ridge 之後 ITL 變差", desc: "轉為 compute-bound，步時隨 batch 變長。", hint: null }
        ],
        answer: 3,
        explain: "三個限制：① KV cache 吃光 HBM；② 超過 ridge point（H100 約 296）後變 compute-bound，步時隨 batch 線性成長，ITL 變差；③ 尾延遲與公平性。所以生產上是「在 SLO 之下把 batch 開到最大」的約束最佳化。出處：第三堂第 5 頁與 Q&A「既然 batch 越大越好，為什麼不無限加大？」。",
        dig: "分頁 KV、continuous batching、前綴複用被稱為「三隻腳」。它們各自撐起的是 B 的上限、實際值，還是省掉不必算的部分？"
      }
    ]
  },

  // ───────────────────────────── 第四堂 ─────────────────────────────
  {
    id: 4, title: "第四堂 · SGLang 多機篇", concept: "分散式推論的四個問題（⑤–⑧）",
    questions: [
      {
        level: "辨識", type: "concept", concept: "八類經典問題：規避、變形、放大",
        q: "把經典分散式系統的八類問題對照到 GPU 推論叢集，哪一類不但沒有被規避，反而被放大成核心工程難題？",
        options: [
          { label: "可擴展性與負載均衡", desc: "通訊量隨規模成長、局部性和均衡此消彼長、熱點難避免。", hint: null },
          { label: "共識（Paxos／Raft）", desc: "多個節點要對同一件事達成一致，才能繼續往下算。", hint: "在一般分散式資料庫裡，共識確實是核心難題。但 GPU 叢集的集合通訊是 SPMD 事先規劃好的，不需要執行期協商誰對。再想一次？" },
          { label: "拜占庭容錯", desc: "有節點可能送出錯誤或惡意的訊息，要能辨識出來。", hint: "在互不信任的開放網路裡，拜占庭容錯確實不可少。但 GPU 叢集是封閉、受信任、同構的環境，只會 fail-stop。再想一次？" },
          { label: "訊息丟失、重複與亂序", desc: "網路不可靠，應用層要自己處理重送與排序。", hint: "在一般網路程式裡這確實是大問題。但 RDMA／NCCL 在底層就保證了可靠性，這一類問題是換了軸，變成頻寬和拓撲。再想一次？" }
        ],
        answer: 0,
        explain: "GPU 叢集用放棄互不信任、放棄執行期協商、（訓練時）放棄持續可用，換到不必做拜占庭、共識與 CAP 取捨；代價是把全部複雜度壓到效率這一根軸上。可擴展性被放大：all-to-all 隨 EP 規模成長、快取局部性 vs 負載均衡、專家負載不均（資料傾斜）。出處：第四堂第 3–5 頁、講稿 §1.1–1.2。",
        dig: "推論服務比訓練更像傳統分散式系統。第三個放棄（持續可用）收不回來之後，哪幾類經典問題又回來了？"
      },
      {
        level: "辨識", type: "concept", concept: "KV 複製的動機",
        q: "在多副本推論叢集裡，router 有時會把某台機器上的熱門前綴 KV 複製到另一台。這麼做的主要動機是什麼？",
        options: [
          { label: "保證 KV 不會在故障時遺失", desc: "多存一份備份，副本掛掉時可以直接從另一台接手。", hint: "在一般的資料庫裡，複製確實主要為了可靠性。但 KV cache 完全由 prompt 決定，掉了可以重算，只是慢、不會錯。再想一次？" },
          { label: "維持各副本快取的一致性", desc: "讓每台機器看到的快取內容相同，避免讀到過時資料。", hint: "多副本一致性在經典分散式系統裡確實是大題目。但 KV 是由 token 序列決定的計算結果，不會被改寫，也不存在「過時版本」。再想一次？" },
          { label: "分流熱點，免去重算", desc: "新請求可分到空閒機器，又不必重新 prefill。", hint: null },
          { label: "滿足 CAP 裡的可用性需求", desc: "在網路分區時讓服務仍然能回應請求。", hint: "推論服務確實比訓練更在意可用性。但 KV 複製發生在正常負載調度時，不是為了應付網路分區。再想一次？" }
        ],
        answer: 2,
        explain: "KV cache 是「可重算的快取」，不是「不可回復的狀態」。所以 KV 複製的動機不是可靠性，而是省算力：在熱門機器負載過高時，把高價值快取複製到空閒機器，讓請求可以分流又不必重新 prefill。問題⑦與⑧用同一套機制，但動機完全不同。出處：第四堂第 13 頁、第 15 頁，講稿 §5.2、§6.1。",
        dig: "router 對各機快取內容只有近似且過時的視圖。這會讓複製決策出錯在哪個方向？是複製太多還是太少？"
      },
      {
        level: "邊界", type: "debug", concept: "MoE 在單卡上不省容量",
        q: "有人主張：「DeepSeek-V3 每個 token 只啟用 37B / 671B ≈ 5.5% 的參數，所以用一台 8 卡機的 HBM 就放得下。」這個推論錯在哪？",
        options: [
          { label: "活躍比例其實遠高於 5.5%", desc: "加上共享專家和 attention，實際啟用接近一半。", hint: "共享專家和 attention 確實每個 token 都會用到。但 37B 已經把它們算進去了，比例本身沒有算錯。再想一次？" },
          { label: "全部專家仍要常駐 HBM", desc: "每個 token 選的專家都不一樣，隨時可能用到任何一個。", hint: null },
          { label: "KV cache 比權重還大", desc: "長序列下 KV 會超過權重，擠掉專家的空間。", hint: "KV 確實會和權重搶 HBM。但就算完全不放 KV，這個推論在權重這一關就已經不成立了。再想一次？" },
          { label: "all-to-all 需要額外緩衝區", desc: "專家路由時要預留大量通訊 buffer 才能運作。", hint: "EP 的通訊 buffer 確實會佔用一些 HBM。但單機不一定要做 all-to-all，問題出在更基本的容量計算。再想一次？" }
        ],
        answer: 1,
        explain: "MoE 的「每 token 只活躍一小部分」省的是每 token 的 FLOPs 與權重讀取，不省容量：所有專家都得在 HBM 裡待命。要連容量也省，得用 EP 把 256 個專家散到幾十上百張卡，每張只放幾個，代價是每層兩次 all-to-all。出處：第四堂第 7 頁、講稿 §3。",
        dig: "DeepSeek-V3 的 FP8 權重約 688.6 GB。一台 8×80 GB 的 H100 放不下；最少要幾台才能放下權重並留一半 HBM 給 KV？"
      },
      {
        level: "邊界", type: "debug", concept: "局部性 vs 負載均衡",
        q: "團隊把 router 從 cache-aware 改成 Round Robin。各機負載變得很均勻，但平均 TTFT 反而明顯變差。最可能的原因？",
        options: [
          { label: "Round Robin 會造成熱點", desc: "輪詢讓少數機器接到太多長請求，排隊變長。", hint: "把同類請求固定送同一台的內容感知路由，確實會造成熱點。但題目說改成輪詢後各機負載很均勻，熱點不是這裡的問題。再想一次？" },
          { label: "router 的視圖太舊，送錯機器", desc: "router 不知道各機最新狀態，把請求送到忙的機器。", hint: "router 視圖過時確實是 cache-aware 路由要面對的問題。但 Round Robin 根本不看任何狀態，視圖新舊對它沒有影響。再想一次？" },
          { label: "各機的 FSM 狀態被打亂", desc: "同一個 schema 的請求分散後，語法狀態要重建。", hint: "結構化輸出確實有狀態。但 FSM 狀態本來就屬於每個請求自己，不會因為換機器而變差。再想一次？" },
          { label: "共享前綴被打散到各台", desc: "每台都只看到一部分同前綴請求，快取難以命中。", hint: null }
        ],
        answer: 3,
        explain: "多台機器各自維護獨立的 radix tree、彼此不共享 KV。Round Robin 完全不看內容，本該集中在同一台的共享前綴請求被打散，命中率崩潰，每台都得重新 prefill。反過來，內容感知路由會製造熱點。這就是快取局部性與負載均衡此消彼長。出處：第四堂第 12 頁、講稿 §5.1。",
        dig: "這個矛盾跟 CDN 邊緣快取、一致性雜湊是同一個結構。一致性雜湊加上「熱點複製」的做法，對應到 SGLang router 的哪個機制？"
      },
      {
        level: "遷移", type: "scenario", concept: "該搬還是該重算",
        q: "兩台推論機只靠 InfiniBand（約 50 GB/s）相連。某個熱門前綴有 32,000 token，用 Llama-3-8B（每 token KV 128 KB）計算。要讓空閒那台也能服務這個前綴，該複製 KV 還是讓它自己重算？",
        options: [
          { label: "重算：跨機頻寬太慢了", desc: "IB 比 NVLink 慢一個數量級，跨機搬 GB 級資料不划算。", hint: "如果是很短的前綴，重算確實可能比建立傳輸還划算。但題目是 32K token 的長前綴，要先算清楚兩邊各要多久。再想一次？" },
          { label: "重算：KV 屬於請求狀態", desc: "KV 綁定原本的請求，不能交給另一台機器使用。", hint: "請求狀態（例如 FSM、取樣種子）確實不能跨機共享。但 KV 完全由 token 序列決定，是計算結果，可以複製。再想一次？" },
          { label: "搬：約 80 ms 就到", desc: "4 GB ÷ 50 GB/s，比重跑 prefill 快。", hint: null },
          { label: "都不必，改用 Round Robin", desc: "把請求平均分出去，就不必處理快取放在哪裡。", hint: "Round Robin 確實能讓負載均勻。但它會把共享前綴打散，每台都得自己重算，等於選了最貴的那條路。再想一次？" }
        ],
        answer: 2,
        explain: "32,000 × 128 KB ≈ 4 GB，走 IB（50 GB/s）約 80 ms；重新 prefill 32K token 要數百 ms 到秒級。重算是 O(n²)、搬是 O(n)，所以前綴越長越該搬；在 NVLink 域內幾乎永遠該搬，跨節點走乙太才要仔細算。出處：第四堂第 14 頁（壓軸推導）、講稿 §5.3。",
        dig: "如果把模型換成 MLA 的 DeepSeek-V3（每 token 約 70 KB），這筆帳會怎麼變？在多短的前綴時「重算」才會開始划算？"
      },
      {
        level: "遷移", type: "scenario", concept: "PD 分離的好處",
        q: "某服務的流量從「短問題、長回答」的聊天，轉成「長文件、短摘要」的工作。在 PD 分離架構下，這個轉變最能被哪個特性吸收？",
        options: [
          { label: "chunked prefill 自動調整預算", desc: "每步的 token 預算會依 prompt 長度自動重新分配。", hint: "chunked prefill 確實能緩解 prefill 打斷 decode。但它是單機內的混合批次技巧，而題目問的是 PD 分離架構本身的特性。再想一次？" },
          { label: "兩個池子能分開擴縮容", desc: "prefill 和 decode 各自加減機器，比例跟著流量走。", hint: null },
          { label: "KV 走 NVLink，傳輸成本很低", desc: "prefill 算完的 KV 很快送到 decode，不會成為瓶頸。", hint: "KV 傳輸快確實是 PD 分離能成立的前提。但它不會因為流量型態改變而變成解方，題目問的是如何吸收負載比例的變化。再想一次？" },
          { label: "decode 端改開大 TP、追 FLOPs", desc: "長文件讓 decode 變成 compute-bound，要多給算力。", hint: "大 TP、追 FLOPs 確實是某一個池子的最佳配置。但那是 prefill 池的配置，而且流量轉成短摘要時，decode 並不會變得吃算力。再想一次？" }
        ],
        answer: 1,
        explain: "PD 分離讓 prefill 群（大 TP、小批、追 FLOPs 與 TTFT）與 decode 群（大 EP、大批、追頻寬與 KV 容量）各自開到最佳點，而且可以分開擴縮容；prefill 與 decode 的負載比例會隨流量型態變動，這點很重要。代價是多一次 KV 傳輸和多一層調度。出處：第四堂第 10–11 頁、講稿 §4。",
        dig: "流量轉成長文件摘要後，prefill 池和 decode 池的機器比例大概往哪個方向調？這時 router 的 cache-aware 還重要嗎？"
      },
      {
        level: "取捨", type: "tradeoff", concept: "大規模 EP 的代價",
        q: "把 DeepSeek 的 256 個專家從 8 張卡攤到 72 張卡上（EP 變大），每張卡的權重變小、decode 變快。付出的代價主要是什麼？",
        options: [
          { label: "每張卡的 KV 容量變小", desc: "專家變多占掉 HBM，留給 KV 的空間減少。", hint: "如果每張卡放的專家變多，KV 空間確實會被擠壓。但 EP 變大時每張卡放的專家是變少，省下的 HBM 反而能給 KV。再想一次？" },
          { label: "attention 要改成 TP 才算得動", desc: "專家散開後 attention 無法再各算各的。", hint: "attention 的切法確實要跟著調整。但講稿的配置是 attention 走資料平行、各卡算自己那批請求，不需要改成 TP。再想一次？" },
          { label: "單請求要讀的權重變多", desc: "專家分散後，每個 token 要去多張卡讀權重。", hint: "token 確實會被送到多張卡上的專家。但每張卡讀的是自己那幾個專家，整體每 token 讀的權重量沒有變多。再想一次？" },
          { label: "all-to-all 與專家熱點", desc: "每層兩次全員交換，熱門專家那張卡會被塞爆。", hint: null }
        ],
        answer: 3,
        explain: "EP 的代價：每層兩次 all-to-all（token 送去專家、算完送回），通訊成為主角；專家負載不均等於資料傾斜，熱門專家的卡塞爆、其他卡閒著，推論期靠 EPLB 做副本與重排。配套 kernel 是 DeepEP（all-to-all）、DeepGEMM、FlashMLA。出處：第四堂第 7–8 頁、講稿 §3。",
        dig: "SGLang 在 96×H100 上用大規模 EP 相對純 TP 輸出吞吐最高 5 倍。既然 all-to-all 這麼貴，這 5 倍是從哪裡賺回來的？"
      },
      {
        level: "取捨", type: "tradeoff", concept: "最貴的失敗模式",
        q: "在 PD 分離的架構下，下面哪一種失敗最「貴」（浪費最多已花掉的資源）？",
        options: [
          { label: "KV 送達後 decode 機才掛", desc: "prefill 的算力和 KV 傳輸都已花掉，卻沒人接手生成。", hint: null },
          { label: "請求剛進 router 就被丟棄", desc: "還沒開始排隊，使用者必須重新送出請求。", hint: "使用者體驗上這確實很糟。但這時系統還沒花任何算力或頻寬，要重來的成本很低。再想一次？" },
          { label: "prefill 算到一半時 prefill 機掛", desc: "算到一半的 forward 全部作廢，需要換一台重算。", hint: "算到一半確實浪費了一部分算力。但和其他選項比，這裡還沒付出 KV 傳輸，損失的只是半段 prefill。再想一次？" },
          { label: "某台機器的 radix tree 被清空", desc: "快取內容全部消失，之後的請求命中率掉到 0。", hint: "快取清空確實會讓之後的請求變慢。但 KV 是可重算的快取，丟掉只是慢、不會錯，也不影響正在跑的請求。再想一次？" }
        ],
        answer: 0,
        explain: "PD 分離下 prefill 完成、KV 也已傳過去，decode 機卻掛了：算力已花、頻寬已花、所有權也還在交接中，是最貴的失敗模式。這是「KV 所有權轉移」這個舊問題的影子：中途誰負責釋放、接手前掛了怎麼辦。出處：第四堂第 11 頁、第 16 頁，講稿 §4 與 §6.2。",
        dig: "SGLang「先預留再推」和 vLLM「先算完再拉」這兩種交接寫法，哪一種在 decode 機掛掉時比較容易回收 prefill 端的資源？"
      }
    ]
  },

  // ───────────────────────────── 第五堂 ─────────────────────────────
  {
    id: 5, title: "第五堂 · 中國開源模型的五個旋鈕", concept: "把推論成本寫進模型架構的五個旋鈕",
    questions: [
      {
        level: "辨識", type: "concept", concept: "MoE 稀疏省的是什麼",
        q: "旋鈕②「少算」把 MoE 的活躍比例一路壓低（例如 Kimi K2 只有 3.2%）。在單張卡上，這個旋鈕直接省下的是什麼？",
        options: [
          { label: "HBM 裡要放的權重容量", desc: "只有被選到的專家才需要放在卡上。", hint: "如果專家可以按需載入，容量確實會省。但每個 token 選的專家都不同，所有專家都要在 HBM 待命。再想一次？" },
          { label: "每 token 的 KV cache 大小", desc: "活躍參數少，每層要存的 K、V 也跟著變少。", hint: "KV 確實是 decode 的另一大負擔。但 KV 大小由注意力的設計決定（head 數、latent 維度），MoE 改的是 FFN 那一塊。再想一次？" },
          { label: "每 token 的權重讀取與 FLOPs", desc: "只走少數專家，讀與算都變少。", hint: null },
          { label: "卡與卡之間的通訊量", desc: "每個 token 用到的專家少，要交換的資料自然少。", hint: "直覺上用得少、傳得也少。但在多卡上 MoE 反而多了 all-to-all 路由，單卡上則根本沒有這段通訊可省。再想一次？" }
        ],
        answer: 2,
        explain: "決定 decode 速度的是活躍參數 + KV，不是總參數；MoE 讓每 token 只讀、只算少數專家，所以省權重讀取與 FLOPs。但它在單卡上不省容量（專家都要在 HBM），要省容量得靠第四堂的大規模 EP。出處：第五堂第 6 頁。",
        dig: "Kimi K3 是 2.8T 總參數、104B 活躍。在本機跑它時，哪個數字決定「裝不裝得下」，哪個決定「跑多快」？"
      },
      {
        level: "辨識", type: "concept", concept: "MTP 在推論時的角色",
        q: "DeepSeek-V3 和 Qwen3-Next 都在訓練時加了 MTP（Multi-Token Prediction）head。到了推論時，這些 head 主要被拿來做什麼？",
        options: [
          { label: "取代部分 KV cache", desc: "多預測的 token 先存起來，後面的層就不用再存 K、V。", hint: "減少 KV 確實是這堂課的一大主題。但那是旋鈕①的事；MTP 屬於另一個旋鈕，處理的是單請求延遲。再想一次？" },
          { label: "降低訓練與推論的精度", desc: "多預測幾步可以抵銷低精度帶來的誤差。", hint: "精度確實是五個旋鈕之一。但 MTP 在訓練時的作用是提供更密的監督訊號，和數值精度無關。再想一次？" },
          { label: "引導 router 平衡專家負載", desc: "提前知道後面幾個 token，可以先把專家分配好。", hint: "專家負載均衡確實是 MoE 的重要問題。但 DeepSeek 用的是無輔助損失的路由 bias 調整，不是靠 MTP。再想一次？" },
          { label: "當投機解碼的 draft", desc: "由模型自己提出幾個候選 token，再一次驗證。", hint: null }
        ],
        answer: 3,
        explain: "MTP 訓練時多預測幾步當額外監督，推論時那些 head 直接當投機解碼的 draft：權重讀一次、驗 k 個 token，AI 從 1 變 k。這是模型端主動配合投機解碼，把 draft 模型長在自己身上；DeepSeek-V3 第二 token 接受率約 85–90%。出處：第五堂第 10 頁（旋鈕④）。",
        dig: "為什麼把 draft 長在模型自己身上，比另外找一個小模型當 draft 的接受率更高？部署上又省了什麼？"
      },
      {
        level: "邊界", type: "debug", concept: "線性注意力打壞 prefix caching",
        q: "某團隊把模型骨幹從 full attention 換成線性注意力，結果在 SGLang 上，RadixAttention 帶來的收益幾乎歸零。是哪個前提被破壞了？",
        options: [
          { label: "狀態無法按前綴切片複用", desc: "固定大小的遞迴狀態不像 KV 可以一段一段切下來共用。", hint: null },
          { label: "線性注意力沒有前綴概念", desc: "它不按 token 順序處理，前後文的順序被打亂了。", hint: "如果模型真的不看順序，前綴確實沒有意義。但線性注意力仍是按順序遞迴更新狀態，前綴是存在的。再想一次？" },
          { label: "KV 變得太大，放不進 radix tree", desc: "線性注意力的每 token 狀態比 full attention 更大。", hint: "如果 KV 變大，快取確實會更吃緊。但線性注意力是「不存 KV」，換成固定大小的狀態，問題不在大小。再想一次？" },
          { label: "排程器不再把同前綴排在一起", desc: "cache-aware 排程只認得 full attention 的請求。", hint: "cache-aware 排程確實影響命中率。但排程只是決定順序；就算排在一起，這種模型也拿不出可以複用的東西。再想一次？" }
        ],
        answer: 0,
        explain: "稀疏注意力保留完整 KV、只是每個 query 看 top-k；線性注意力不存 KV，改成固定大小的遞迴狀態。這種狀態不像 KV 能直接切片複用，所以 RadixAttention 的整套價值歸零；同時線性狀態對精度敏感（旋鈕⑤失效），投機解碼在線性骨幹上也仍是未解問題。出處：第五堂第 7 頁、第 9 頁。",
        dig: "Qwen 和 Kimi K3 都用約 3:1 的混合比例保留了一部分 full attention 層。這些 full 層對 prefix caching 能救回多少？"
      },
      {
        level: "邊界", type: "debug", concept: "理論 FLOPs ≠ wall-clock",
        q: "MiniMax M1 的線性注意力在紙上把長生成的 FLOPs 大幅壓低，但實測速度沒有等比例提升，甚至訓練時也吃不滿算力。第一個該懷疑的是什麼？",
        options: [
          { label: "評測分數在多跳推理上退化", desc: "放大後才發現，長文中串線索的能力壞掉了。", hint: "多跳推理退化確實是 MiniMax 的另一個教訓。但那是品質問題，題目問的是為什麼「速度」沒有跟上 FLOPs 的減少。再想一次？" },
          { label: "實作受頻寬限制", desc: "每搬一個 byte 只做很少運算。", hint: null },
          { label: "MTP 的接受率太低", desc: "投機解碼一直被拒絕，一次只能產生一個 token。", hint: "投機解碼在線性骨幹上確實有困難。但題目說連訓練時都吃不滿算力，訓練並不經過投機解碼。再想一次？" },
          { label: "MoE 專家負載嚴重不均", desc: "熱門專家所在的卡拖慢了整體速度。", hint: "專家熱點確實會拖慢 MoE。但題目描述的現象出在注意力那一段，而且單卡訓練時也會出現。再想一次？" }
        ],
        answer: 1,
        explain: "線性注意力的實作本身就是 memory-bound，即使訓練時也吃不滿算力；省下的是紙上的 FLOPs，不是牆上的時間。這完全是第一堂 roofline 的教訓：「理論複雜度更低」離「生產環境更快」隔著 kernel 效率、評測有效性、生態相容性三層。出處：第五堂第 8 頁（No Free Lunch 第 2 個理由）。",
        dig: "用 roofline 的語言說：線性注意力的 AI 大約落在哪裡？要怎麼改寫 kernel 才可能把它推向平頂？"
      },
      {
        level: "遷移", type: "scenario", concept: "總參數 vs 活躍參數",
        q: "你有一台 128 GB 統一記憶體的 Mac（容量大、頻寬比 GPU 低），想在本機跑一個「裝得下、又不會太慢」的開源模型。哪一類設計最適合？",
        options: [
          { label: "同樣大小的 dense 模型", desc: "沒有路由開銷，每一層都完整使用，品質最穩定。", hint: "dense 模型確實沒有路由問題。但 dense 每 token 要讀全部權重，在頻寬低的機器上，每一步都會很慢。再想一次？" },
          { label: "大總參數、小活躍的 MoE", desc: "總量放得下，每 token 只讀一小部分。", hint: null },
          { label: "2.8T 參數的 MXFP4 模型", desc: "4-bit 讓權重縮到 1/4，大模型也能塞進本機。", hint: "MXFP4 確實讓權重大幅縮小。但 2.8T 就算是 4-bit 也要約 1.4 TB，遠超過 128 GB。再想一次？" },
          { label: "純線性注意力的小模型", desc: "不存 KV，長對話也不會吃掉記憶體。", hint: "不存 KV 確實省記憶體。但 MiniMax 的教訓是純線性骨幹在多跳推理與生態相容上都有實證問題，也不是講稿推薦給本機的答案。再想一次？" }
        ],
        answer: 1,
        explain: "總參數是容量門檻、活躍參數是速度門檻。Qwen3-Next 80B-A3B 這種大總參數、小活躍的設計，放得進大容量統一記憶體，每 token 只讀約 3B 權重，在低頻寬機器上仍可接受，呼應第一堂「容量夠、頻寬低 → 慢但跑得動」。出處：第五堂 Q&A「這些模型我在本機跑得動嗎？」。",
        dig: "80B-A3B 用 4-bit 量化後大約佔多少 GB？如果這台 Mac 的頻寬是 400 GB/s，batch=1 decode 的速度下限大約多少 tok/s？"
      },
      {
        level: "遷移", type: "scenario", concept: "稀疏 vs 線性的生態代價",
        q: "一家新實驗室想降低 1M context 的推論成本，但他們的服務重度依賴 prefix caching 與投機解碼，不想重做這些系統。旋鈕③該往哪個方向轉？",
        options: [
          { label: "稀疏注意力", desc: "保留完整的 KV，每個 query 只挑 top-k 個位置來看。", hint: null },
          { label: "線性注意力為主的混合", desc: "大部分層改成固定大小的遞迴狀態，只留少數 full 層。", hint: "混合線性確實能大幅降低長 context 的成本，Qwen 和 Kimi 都這樣做。但線性層不存 KV，題目要保住的那兩個系統在這些層上會失效。再想一次？" },
          { label: "滑動窗口混合", desc: "每層只看最近一段固定長度的 token。", hint: "滑動窗口確實能控制成本。但 MiniMax 試過各種比例與設定，在 agent 任務與複雜長文評測上一致地很差。再想一次？" },
          { label: "改用 GQA 壓縮 KV head", desc: "多個 query head 共用一組 K、V。", hint: "GQA 確實能減少 KV，但它屬於旋鈕①（壓 KV），不會改變每個 query 要看的範圍，長 context 的 O(n²) 還在。再想一次？" }
        ],
        answer: 0,
        explain: "稀疏注意力保留完整 KV、每個 query 只看 top-k，所以「KV 還在」，prefix caching 與投機解碼都還能用；線性注意力不存 KV，三個生產系統都得重做。這就是 2026 的共識是「混合 + 稀疏」而不是線性取代 full attention 的原因。例：DeepSeek V4 在 1M ctx 下 FLOPs 只需 V3.2 的 27%、KV 只需 10%。出處：第五堂第 7 頁、第 11 頁。",
        dig: "稀疏注意力雖然保留了 KV，但每個 query 只看 top-k。這對 KV 的容量有沒有幫助？還是只省頻寬和運算？"
      },
      {
        level: "取捨", type: "numeric", concept: "MLA 的 KV 量級",
        q: "DeepSeek-V3 用 MLA 把 K/V 投影成 576 維的 latent（61 層、BF16）。每個 token 的 KV 大約多大？",
        options: [
          { label: "約 512 KB", desc: "和 Llama 式 32 頭 MHA 同一個量級。", hint: "512 KB 是 Llama 式 32 頭 MHA 的數字。但 MLA 每層只存一個 576 維的向量，不是每個 head 各存一份 K、V。再想一次？" },
          { label: "約 128 KB", desc: "和 Llama-3-8B 的 GQA 同一個量級。", hint: "128 KB 是 Llama-3-8B 用 GQA（8 個 KV head）的數字。MLA 每層只存 576 個值，用 576 × 61 × 2 bytes 算算看。再想一次？" },
          { label: "約 3.8 MB（同規模 MHA）", desc: "DeepSeek-V3 骨架若用 MHA，每 token 就要這麼多。", hint: "3.8 MB 是同規模 MHA 推算出來的對照值。題目問的是用了 MLA 之後，只存 576 維 latent 的情況。再想一次？" },
          { label: "約 70 KB", desc: "576 個值 × 61 層 × 2 bytes。", hint: null }
        ],
        answer: 3,
        explain: "576 × 61 × 2 B ≈ 70 KB／token；同規模 MHA 推算約 3.8 MB，Llama-3-8B 的 GQA 是 128 KB、32 頭 MHA 是 512 KB。DeepSeek-V2 論文自陳 MLA 讓 KV 比 MHA 減少 93.3%。MLA 是為了 decode 的 HBM 頻寬而發明的，架構決策就是硬體帳單。出處：第五堂第 5 頁（旋鈕① 壓 KV）。",
        dig: "同樣 5,000 token 的請求，MLA 和 MHA 的 KV 各有多大？第六堂為什麼說 MLA「順手付了另一張帳」？"
      },
      {
        level: "取捨", type: "tradeoff", concept: "效率研究的結構性障礙",
        q: "MiniMax 從 M1 的線性混合退回 M2 的 full attention。講稿認為，這個故事揭露的效率研究最大的結構性障礙是什麼？",
        options: [
          { label: "線性注意力的論文太少", desc: "缺乏公開的研究基礎，各家只能自己摸索。", hint: "如果研究真的很少，摸索確實會比較慢。但 Lightning、GDN、KDA 等線性注意力已經有多家在做，問題不在數量。再想一次？" },
          { label: "出口管制讓算力不夠", desc: "沒有足夠的 GPU，連訓練都跑不完。", hint: "算力受限確實是這批實驗室追求效率的背景。但題目問的是「效率研究本身」的結構性障礙，就算算力無限，這個障礙依然存在。再想一次？" },
          { label: "驗證本身就極耗算力", desc: "問題放大後才會浮現，要得到顯著訊號得先花大錢。", hint: null },
          { label: "開源社群不願意支援新架構", desc: "vLLM 和 SGLang 拒絕合併非標準的注意力 kernel。", hint: "新架構要被框架支援確實是一道門檻。但講稿的說法是實驗室自己開源 kernel 讓框架接得住，而不是框架拒絕。再想一次？" }
        ],
        answer: 2,
        explain: "混合注意力在 MMLU、BBH、LongBench 上看起來沒問題，放大後才發現多跳推理明顯退化；要在困難任務上得到統計顯著訊號，所需算力是天文數字。弔詭的是：「省算力的方法」需要巨量算力才驗證得了。出處：第五堂第 8 頁與第 16 頁帶走三句話。",
        dig: "如果你只有有限算力、又想評估一個新的高效注意力，該設計哪種評測才能盡早抓到多跳推理的退化？"
      }
    ]
  },

  // ───────────────────────────── 第六堂 ─────────────────────────────
  {
    id: 6, title: "第六堂 · 一個字穿過一整排機櫃", concept: "機櫃群推論的四種資料與 scale-up 域",
    questions: [
      {
        level: "辨識", type: "concept", concept: "四種資料 × 四條路",
        q: "一個請求穿過 DeepSeek 推論叢集時，會搬四種資料。依「多常搬 × 一次多大」來看，哪一種付不起跨機的慢車道？",
        options: [
          { label: "prefill → decode 的 KV 交接", desc: "每個請求一次，MLA BF16 約 351 MB。", hint: "351 MB 看起來很大，跨機確實要花時間。但它每個請求只搬一次，走 400G 網卡約 7 ms，放在 2–5 秒的 TTFT 裡是雜訊。再想一次？" },
          { label: "MoE 的 dispatch／combine", desc: "每層每步、全體卡同步，每產一個字要 116 次。", hint: null },
          { label: "整份模型權重", desc: "全部約 688.6 GB，是四種裡最大的一包。", hint: "權重確實是最大的一包。但它幾乎不搬，只在啟動和 EPLB 重排時動一下，頻率極低。再想一次？" },
          { label: "token ids 與串流回傳", desc: "每請求一次、每字一次，每次只有幾 KB。", hint: "這種資料確實搬得很頻繁。但每次只有幾 KB，走 ms 級的前端乙太就夠了。再想一次？" }
        ],
        answer: 1,
        explain: "MoE hidden states 是每層 × 每步 × 72 張卡同步交換，DeepSeek-V3 有 58 層 MoE、每層 dispatch + combine 各一次，所以每個字 116 次，只有 scale-up 域或特化 RDMA 付得起。請求本身走前端乙太、KV 走跨機 RDMA（MLA 讓它只要 7 ms）、權重幾乎不搬。出處：第六堂第 4 頁、第 14 頁。",
        dig: "116 次是怎麼算出來的？如果模型改成每 2 層才放一個 MoE 層，這個數字和對 scale-up 域的依賴會怎麼變？"
      },
      {
        level: "辨識", type: "concept", concept: "兩家 KV 交接流程",
        q: "PD 分離時，prefill 算完的 KV 要交給 decode。下面哪一個描述的是 vLLM + NIXL 的寫法？",
        options: [
          { label: "decode 用 RDMA 拉走 KV", desc: "prefill 留著 KV、回傳位址，decode 自己讀。", hint: null },
          { label: "decode 先預留槽位，再收推送", desc: "router 同時送給兩邊，decode 握手後先配好 KV 空間。", hint: "「先預留、再推」確實是一種真實的交接流程。但它是另一家引擎的寫法，差別在由誰主動、以及請求是同時送還是串行送。再想一次？" },
          { label: "prefill 經 CPU 記憶體轉交 decode", desc: "KV 先落到主機記憶體當中繼，再搬到另一台機器。", hint: "經 CPU 中轉確實是比較傳統的做法。但這兩家的交接都走 GPU 對 GPU 的 RDMA，刻意避開 CPU 這一跳。再想一次？" },
          { label: "router 先暫存 KV 再轉發", desc: "router 收下整包 KV，選好 decode 機後再送出去。", hint: "router 確實站在兩個池子中間。但它只搬幾 KB 的請求與 metadata，不會經手幾百 MB 的 KV。再想一次？" }
        ],
        answer: 0,
        explain: "vLLM + NIXL 是「串行送、先算完、再拉」：proxy 先送 prefill，prefill 把 KV 留在自己 GPU 上並回傳位址，decode 用單邊 RDMA read 拉走，之後 prefill 才釋放 block。SGLang 是「並行送、先預留、再推」：router 同時送兩邊，decode 先預留 KV 槽位，prefill 算完用 RDMA 推過去。出處：第六堂第 7 頁。",
        dig: "在「拉」的寫法裡，prefill 端的 KV block 要等 decode 讀走才釋放。如果 decode 遲遲不來讀，會壓到 prefill 池的什麼資源？"
      },
      {
        level: "邊界", type: "debug", concept: "頻寬 vs 域大小",
        q: "把 9 台 H100 8 卡機換成 9 台 B200 8 卡機，NVLink 每卡頻寬快了一倍，EP72 的配置不變。結果每一步的 MoE 通訊時間幾乎沒省（仍約 50 ms）。為什麼？",
        options: [
          { label: "B200 的 NVLink 其實沒有比較快", desc: "規格頁寫的是雙向合計，單向跟 H100 差不多。", hint: "雙向與單向的換算確實常被搞混。但即使換算成每方向，NVLink 5 的 900 GB/s 仍是 NVLink 4 的 450 GB/s 的兩倍。再想一次？" },
          { label: "DeepEP 在 B200 上沒有最佳化", desc: "通訊 kernel 還沒為新硬體調校，吃不滿頻寬。", hint: "軟體成熟度確實常常落後於新硬體。但這裡用同樣的計算模型推算也得到一樣的結果，不是 kernel 的問題。再想一次？" },
          { label: "計算時間蓋過了通訊", desc: "通訊已經被 TBO 完全藏進計算裡，看不出差別。", hint: "TBO 確實能藏住一部分通訊。但題目比的是通訊本身的時間，而且就算有 TBO，也只藏得住約 65%。再想一次？" },
          { label: "約九成的流量走跨機網卡", desc: "72 張卡裡只有 7 張跟自己同機，大部分交換要出機器。", hint: null }
        ],
        answer: 3,
        explain: "一個 token 送往的卡裡，落在同一個 scale-up 域的比例 ≈ (8−1)/(72−1) ≈ 10%，約 90% 的流量走 400G 網卡（約 50 GB/s）。瓶頸在網卡，NVLink 快一倍只加速那一成。換成 GB200 NVL72（域 = 72），跨機比例變 0%，每步通訊約 9 ms。出處：第六堂第 13–14 頁。",
        dig: "如果改成 EP16（兩台 8 卡機），跨機流量比例會變成多少？這時 NVLink 變快一倍的效果會比 EP72 時明顯嗎？"
      },
      {
        level: "邊界", type: "debug", concept: "MLA 讓 KV 可以跨機",
        q: "假設 DeepSeek-V3 的骨架改用 MHA（128 頭 × 128 維），其他不變。平均 4,989 token 的 KV 經 400G 網卡從 prefill 交給 decode，會發生什麼？",
        options: [
          { label: "約 7 ms，依然只是雜訊", desc: "交接時間和 MLA 差不多，TTFT 幾乎不受影響。", hint: "7 ms 是用 MLA（每 token 70 KB）算出來的數字。換成 MHA 後每 token 的 KV 大很多，要重新代數字。再想一次？" },
          { label: "約 33 ms，開始有點明顯", desc: "和 Llama-3-70B 的 GQA 同一個量級。", hint: "33 ms 是 GQA（每 token 320 KB）的數字。題目是每層 128 個 head 各存 K、V 的 MHA，每 token 比 GQA 大一個量級。再想一次？" },
          { label: "約 400 ms，變成負擔", desc: "每 token 約 4 MB、共約 20 GB。", hint: null },
          { label: "約 4 秒，完全無法跨機", desc: "KV 大到要切好幾批，每批都要重新握手。", hint: "KV 變大確實會讓交接變慢。但用每 token 約 4 MB × 4,989 token ÷ 49.5 GB/s 算一次，量級沒有到秒。再想一次？" }
        ],
        answer: 2,
        explain: "MHA 每 token 約 4.0 MB，一個請求約 20 GB，÷ 49.5 GB/s ≈ 400 ms；MLA BF16 每 token 70 KB、一個請求 351 MB，只要約 7 ms。MLA 在第五堂是為了 decode 的 HBM 頻寬而發明，還順手讓 prefill 和 decode 可以放在不同機櫃。出處：第六堂第 8 頁。",
        dig: "附錄提到 1M context 時 V4 每條序列 KV 9.62 GiB。跨機交接要多久？這時第 4 頁 KV 那一列還「搬得起」嗎？"
      },
      {
        level: "遷移", type: "scenario", concept: "最小單位是一台",
        q: "一家中型公司想用「一台 8 卡機」部署 DeepSeek-V3 FP8（權重 688.6 GB），QPS 不高，想避開跨機複雜度，並把剩下最多的 HBM 留給 KV。哪一台最適合？",
        options: [
          { label: "AMD MI355X 8 卡機", desc: "每卡 288 GB，一台約 2.3 TB。", hint: null },
          { label: "NVIDIA H100 8 卡機", desc: "生態最成熟，SGLang 主角就是用它。", hint: "H100 叢集確實是第六堂的主角。但一台 H100 只有 8×80 GB = 640 GB，連 688.6 GB 的權重都放不下。再想一次？" },
          { label: "NVIDIA HGX B200 8 卡機", desc: "每卡 180 GB，NVLink 5、NVSwitch 交換。", hint: "B200 一台 1.44 TB，確實放得下權重。但題目要剩下最多的 HBM 給 KV，比一比各台放完權重之後還剩多少。再想一次？" },
          { label: "GB200 NVL72 機櫃", desc: "72 卡同一個 NVLink 域，總 HBM 13.4 TB。", hint: "NVL72 在大規模 EP 上確實領先。但它是一整個機櫃，不是題目要的「一台 8 卡機」這個最小單位。再想一次？" }
        ],
        answer: 0,
        explain: "單位是一台時，決定性數字是單台總 HBM：MI355X 約 2.3 TB，放完 688.6 GB 權重還剩約 1.6 TB 給 KV；B200 1.44 TB 剩約 0.75 TB；H100 640 GB 連權重都放不下。這一格 AMD 有結構優勢；單位換成機櫃時，就要看 scale-up 域大小。出處：第六堂第 17–18 頁。",
        dig: "如果這家公司一年後 QPS 成長十倍、要上大規模 EP，MI355X 的哪個特性會變成弱點？Helios 能補上嗎？"
      },
      {
        level: "遷移", type: "scenario", concept: "節拍器與 straggler",
        q: "decode 池的 72 張卡組成一個 DP-attention + EP72 單元。其中一張卡因為散熱問題被降頻，算得比別人慢 30%。會發生什麼事？",
        options: [
          { label: "只有那張卡的請求變慢", desc: "attention 各算各的，其他卡不受它影響。", hint: "attention 那一段確實是各卡各算。但每一層的 MoE 交換都要全員同步進入，其他卡會在那裡等它。再想一次？" },
          { label: "router 把流量導走就能無感", desc: "不送新請求給那張卡，其他卡照常服務。", hint: "在各副本彼此獨立時，router 導流確實有效。但這 72 張卡是同一個單元，沒請求的卡也要跑空批次陪大家進 all-to-all。再想一次？" },
          { label: "EPLB 把專家搬走，零成本解決", desc: "把那張卡上的專家移到別的卡上，問題就消失了。", hint: "EPLB 確實會重排專家來處理熱點。但重排要搬動權重，並不是零成本；而且那張卡本身在每層同步時仍然會拖住大家。再想一次？" },
          { label: "整個單元的 ITL 被拖慢", desc: "每層都要全員同步，最慢那張卡決定整體節奏。", hint: null }
        ],
        answer: 3,
        explain: "DP-attention + EP 下，同一個 decode 單元的所有 rank 必須同步進入每一層的 all-to-all，沒請求的卡也得跑空批次陪跑。所以最慢那張卡決定整體 ITL（barrier／straggler），掉一張卡整個 72 卡單元停擺；拆散換到大 batch，代價就是被這個節拍器綁住。出處：第六堂第 12 頁。",
        dig: "EP 越大、爆炸半徑越大。在第四堂的「掛了 vs 慢了分不清」這個問題上，72 卡單元要怎麼判斷該等它還是把它踢掉？"
      },
      {
        level: "取捨", type: "numeric", concept: "scale-up 域大小的效益",
        q: "SemiAnalysis 比較同一顆 Blackwell、同樣每卡 NVLink 頻寬的 B200 8 卡機（域 = 8）與 GB200 NVL72（域 = 72），在 125 tok/s/user 條件下跑 DeepSeek-R1。每張 GPU 的吞吐差大約幾倍？",
        options: [
          { label: "約 1.2 倍而已", desc: "只有通訊那一段變快，計算時間不變。", hint: "只看通訊那一段，教具模型算出的差確實只有 1.2–1.5 倍。但實測數字還包含了域變大之後 EP 開更大、每卡權重更少、batch 更大的效果。再想一次？" },
          { label: "約 18 倍", desc: "NVLink 5 每方向頻寬是 400G 網卡的 18 倍。", hint: "18 倍是兩條線的頻寬比。但每卡吞吐還受計算、batch 等其他因素限制，不會直接等於頻寬比。再想一次？" },
          { label: "約 4.4 倍", desc: "941 對 4,130 tok/s/GPU。", hint: null },
          { label: "約 2 倍", desc: "域變大後，跨機流量少了一半左右。", hint: "跨機流量減少是對的方向。但從域 8 到域 72，跨機比例是從約九成降到零，不只是少一半；而且吞吐的差距還包含其他效果。再想一次？" }
        ],
        answer: 2,
        explain: "B200 941 vs GB200 NVL72 4,130 tok/s/GPU，約 4.4 倍、每 1M token 成本 $0.576 → $0.149（÷3.9）。晶片一樣、每卡頻寬一樣，差別幾乎只剩域的大小。教具模型只解釋了通訊那段的 1.2–1.5 倍，其餘來自 EP 開更大、每卡權重更少、batch 更大。出處：第六堂第 15 頁。",
        dig: "4.4 倍裡，大約多少可以歸給「通訊變快」、多少歸給「batch 變大」？用第 10–11 頁的時間帳和 HBM 帳拆一次。"
      },
      {
        level: "取捨", type: "tradeoff", concept: "拆散為何反而便宜",
        q: "SGLang 用 DP-attention + EP72 跑 decode，每張卡的產出比 TP16 高約 5.2 倍，儘管通訊變多了。這筆交換的核心是什麼？",
        options: [
          { label: "通訊量其實變少了", desc: "EP 只傳 hidden states，比 TP 的 all-reduce 輕很多。", hint: "單次交換的量確實可以比較。但 EP72 每產一個字要 116 次全員交換，不做重疊時通訊約佔一步的四成，並沒有變少。再想一次？" },
          { label: "省 HBM 開大 batch", desc: "每張卡只放幾個專家，騰出空間給更多序列的 KV。", hint: null },
          { label: "單請求延遲大幅降低", desc: "每張卡工作變少，每一步跑得更快。", hint: "每卡權重變少確實讓讀取變輕。但主角每步約 92 ms、ITL 約 100 ms，EP 換到的主要不是單請求更快，而是別的東西。再想一次？" },
          { label: "KV 在各卡重複存，命中率提高", desc: "每張卡都有一份完整的 KV，任何請求都能命中。", hint: "重複存一份 KV 在某些場景確實能提高命中。但這正是 TP16 的缺點：MLA latent 不能按 head 切，16 張卡各存同一份，浪費了空間。再想一次？" }
        ],
        answer: 1,
        explain: "EP72 每卡只放 4 個 routed + 1 個 shared 專家，權重約 29 GB，剩約 51 GB 只存自己那批請求的 KV；TP16 每卡約 43 GB 權重，而且 MLA latent 在 16 張卡上各存一份。省下的 HBM 讓每張卡跑 256 條序列，AI ≈ B 被推到幾百。代價則是 72 張卡綁成一個節拍器。出處：第六堂第 11–12 頁。",
        dig: "72 張卡能放的「不重複」KV，EP72 約 3.6 TB、TP16 約 170 GB。這 20 倍的差距裡，有多少來自「權重變少」、多少來自「KV 不重複」？"
      }
    ]
  }
];
