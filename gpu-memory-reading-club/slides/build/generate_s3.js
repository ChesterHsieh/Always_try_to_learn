// S3 — Training vs Inference 的瓶頸差異（以 ASR 為例）
// 產生 ../s3_train_vs_infer_asr.pptx。沿用 S1 的深色「矽晶」主題。
// 執行：node generate_s3.js
const pptxgen = require("pptxgenjs");

// ---- palette（與 S1 一致）----
const BG = "0E1726", BG2 = "16233A", BG3 = "1C2E4A";
const INK = "EAF1FB", MUTE = "8FA6C4", LINE = "2A3D5C", FOOTC = "5C7299";
const MEM = "38BDF8", COMP = "F59E0B", WARN = "FB7185";
const MEMTINT = "10455F", WARNTINT = "4A2433", COMPTINT = "4A3410";
const HEAD = "PingFang TC", BODY = "PingFang TC", MONO = "Menlo";

const W = 13.33, H = 7.5, MX = 0.7, TITLE_Y = 0.62, FOOT_Y = 7.05, TOTAL = 13;
const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 });

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "GPU 記憶體與資料搬遷讀書會";
pres.title = "S3 — Training vs Inference 的瓶頸差異（以 ASR 為例）";

// ---- helpers ----
const base = (s) => { s.background = { color: BG }; };
function runningHeader(s) {
  s.addText("GPU 記憶體與資料搬遷讀書會 · S3", {
    x: W - 5.2, y: 0.3, w: 4.5, h: 0.3, align: "right", fontFace: BODY, fontSize: 10, color: MUTE, margin: 0,
  });
}
function footer(s, n) {
  s.addText("Training vs Inference（以 ASR 為例）", { x: MX, y: FOOT_Y, w: 8, h: 0.3, fontFace: BODY, fontSize: 9, color: FOOTC, margin: 0 });
  s.addText(`${n} / ${TOTAL}`, { x: W - 1.6, y: FOOT_Y, w: 0.9, h: 0.3, align: "right", fontFace: MONO, fontSize: 9, color: FOOTC, margin: 0 });
}
function header(s, num, title, accent) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: TITLE_Y, w: 0.62, h: 0.62, rectRadius: 0.08, fill: { color: accent }, line: { type: "none" }, shadow: shadow() });
  s.addText(num, { x: MX, y: TITLE_Y, w: 0.62, h: 0.62, align: "center", valign: "middle", fontFace: MONO, fontSize: 22, bold: true, color: BG, margin: 0 });
  s.addText(title, { x: MX + 0.85, y: TITLE_Y, w: W - MX - 0.85 - 0.5, h: 0.62, valign: "middle", fontFace: HEAD, fontSize: 29, bold: true, color: INK, margin: 0 });
}
function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: LINE, width: 1 }, shadow: shadow() });
}
// 平行刻度（encoder：一次平行）
function parallelTicks(s, x, y, count, tw, th, gap, color, heights) {
  for (let i = 0; i < count; i++) {
    const hh = heights ? heights[i % heights.length] * th : th;
    s.addShape(pres.shapes.RECTANGLE, { x: x + i * (tw + gap), y: y + (th - hh), w: tw, h: hh, fill: { color }, line: { type: "none" } });
  }
}
// 序列鏈（decoder：逐 token，dots + 箭頭）
function chain(s, x, y, count, r, gap, color) {
  for (let i = 0; i < count; i++) {
    const cx = x + i * (r + gap);
    s.addShape(pres.shapes.OVAL, { x: cx, y, w: r, h: r, fill: { color }, line: { type: "none" } });
    if (i < count - 1) s.addShape(pres.shapes.LINE, { x: cx + r, y: y + r / 2, w: gap, h: 0, line: { color, width: 1.5, endArrowType: "triangle" } });
  }
}
// 小型 roofline
function miniRoofline(s, x0, y0, pw, ph) {
  const topY = y0 - ph, xr = x0 + 0.55 * pw;
  s.addShape(pres.shapes.LINE, { x: x0, y: topY, w: 0, h: ph, line: { color: MUTE, width: 1 } });
  s.addShape(pres.shapes.LINE, { x: x0, y: y0, w: pw, h: 0, line: { color: MUTE, width: 1 } });
  s.addShape(pres.shapes.LINE, { x: x0, y: topY, w: xr - x0, h: ph, flipV: true, line: { color: MEM, width: 3 } });
  s.addShape(pres.shapes.LINE, { x: xr, y: topY, w: x0 + pw - xr, h: 0, line: { color: COMP, width: 3 } });
}

// =============================================================================
// Slide 1 — 標題
// =============================================================================
(() => {
  const s = pres.addSlide(); base(s);
  // 主題字符：上排平行刻度（平行）、下排序列鏈（自迴歸）
  parallelTicks(s, 9.0, 1.95, 9, 0.16, 0.7, 0.12, MEM);
  chain(s, 9.05, 3.25, 6, 0.2, 0.32, COMP);

  s.addText("GPU 記憶體與資料搬遷讀書會  ·  S3 / 共 4 場", { x: MX, y: 1.7, w: 9, h: 0.4, fontFace: BODY, fontSize: 15, color: MEM, bold: true, charSpacing: 1, margin: 0 });
  s.addText([
    { text: "Training vs Inference", options: { breakLine: true } },
    { text: "的瓶頸差異（以 ASR 為例）", options: {} },
  ], { x: MX, y: 2.45, w: 8.3, h: 2.0, fontFace: HEAD, fontSize: 42, bold: true, color: INK, lineSpacingMultiple: 1.06, margin: 0 });
  s.addText("同一個模型，訓練吃算力、推論吃頻寬——為什麼？用 S1 那把尺量給你看。", { x: MX, y: 4.85, w: 11.0, h: 0.6, fontFace: BODY, fontSize: 18, color: MUTE, margin: 0 });
  s.addText("承接 S1：compute-bound vs memory-bound · 算術強度（AI = FLOPs / Byte）", { x: MX, y: 5.6, w: 11.0, h: 0.4, fontFace: BODY, fontSize: 13, color: FOOTC, margin: 0 });
})();

// =============================================================================
// Slide 2 — 回顧 S1 的尺
// =============================================================================
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "01", "先把 S1 那把尺帶回來", MEM);

  miniRoofline(s, 1.6, 4.7, 3.6, 2.0);
  s.addText("算術強度 AI →", { x: 1.6, y: 4.85, w: 3.6, h: 0.3, align: "center", fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });
  s.addText("頻寬上限", { x: 1.7, y: 3.7, w: 1.6, h: 0.3, fontFace: BODY, fontSize: 11, bold: true, color: MEM, margin: 0 });
  s.addText("算力上限", { x: 3.5, y: 2.6, w: 1.6, h: 0.3, fontFace: BODY, fontSize: 11, bold: true, color: COMP, margin: 0 });

  const x2 = 6.2, cw = 6.1;
  card(s, x2, 2.0, cw, 1.5, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: 2.0, w: 0.12, h: 1.5, fill: { color: COMP }, line: { type: "none" } });
  s.addText("AI 高 → compute-bound", { x: x2 + 0.4, y: 2.2, w: cw - 0.7, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: COMP, margin: 0 });
  s.addText("瓶頸是峰值算力。加大 batch 也快不了。", { x: x2 + 0.4, y: 2.7, w: cw - 0.7, h: 0.6, fontFace: BODY, fontSize: 14, color: INK, margin: 0 });

  card(s, x2, 3.7, cw, 1.5, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: 3.7, w: 0.12, h: 1.5, fill: { color: MEM }, line: { type: "none" } });
  s.addText("AI 低 → memory-bound", { x: x2 + 0.4, y: 3.9, w: cw - 0.7, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: MEM, margin: 0 });
  s.addText("瓶頸是記憶體頻寬。加大 batch 吞吐會上升。", { x: x2 + 0.4, y: 4.4, w: cw - 0.7, h: 0.6, fontFace: BODY, fontSize: 14, color: INK, margin: 0 });

  s.addText("本場就用這把尺，量「訓練」與「推論」分別落在 roofline 哪一區。", { x: MX, y: 6.05, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 16, bold: true, color: MEM, margin: 0 });
  footer(s, 2);
})();

// =============================================================================
// Slide 3 — Training
// =============================================================================
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "02", "Training：吃算力，但更怕「裝不下」", COMP);

  const cw = 5.7, ch = 3.7;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 1.95, w: cw, h: 0.12, fill: { color: COMP }, line: { type: "none" } });
  s.addText("為什麼通常 compute-bound", { x: MX + 0.35, y: 2.25, w: cw - 0.7, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: COMP, margin: 0 });
  s.addText([
    { text: "大 batch → 大 GEMM → AI 高", options: { bullet: true, breakLine: true } },
    { text: "forward / backward 都是大矩陣乘法", options: { bullet: true, breakLine: true } },
    { text: "算力利用率高、餵得飽 tensor core", options: { bullet: true } },
  ], { x: MX + 0.35, y: 2.85, w: cw - 0.7, h: 2.5, fontFace: BODY, fontSize: 15.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 8, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: 1.95, w: cw, h: 0.12, fill: { color: MEM }, line: { type: "none" } });
  s.addText("但記憶體同時要放下", { x: x2 + 0.35, y: 2.25, w: cw - 0.7, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: MEM, margin: 0 });
  const items = [
    ["權重 weights", COMP],
    ["啟動值 activations（留著等 backward）", MEM],
    ["梯度 gradients", COMP],
    ["優化器狀態（Adam ≈ 2× 參數，fp32）", WARN],
  ];
  items.forEach((it, i) => {
    const y = 2.9 + i * 0.6;
    s.addShape(pres.shapes.RECTANGLE, { x: x2 + 0.35, y: y + 0.04, w: 0.22, h: 0.22, fill: { color: it[1] }, line: { type: "none" } });
    s.addText(it[0], { x: x2 + 0.72, y: y - 0.04, w: cw - 1.1, h: 0.4, fontFace: BODY, fontSize: 14, color: INK, valign: "middle", margin: 0 });
  });

  s.addText("訓練的痛點常是「裝不下」(容量)，不是「餵不飽」(頻寬)——所以才有梯度檢查點、ZeRO、offload。", { x: MX, y: 5.95, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: COMP, margin: 0 });
  footer(s, 3);
})();

// =============================================================================
// Slide 4 — Inference 兩階段
// =============================================================================
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "03", "Inference 兩階段：prefill vs decode", MEM);

  // 時間軸
  s.addText("時間軸 →", { x: MX, y: 1.95, w: 2, h: 0.3, fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 2.35, w: 3.2, h: 0.7, fill: { color: COMP }, line: { type: "none" } });
  s.addText("prefill（一次吃整段 prompt）", { x: MX, y: 2.35, w: 3.2, h: 0.7, align: "center", valign: "middle", fontFace: BODY, fontSize: 13, bold: true, color: BG, margin: 0 });
  for (let i = 0; i < 9; i++) {
    s.addShape(pres.shapes.RECTANGLE, { x: MX + 3.4 + i * 0.85, y: 2.35, w: 0.62, h: 0.7, fill: { color: MEM }, line: { color: BG, width: 1 } });
  }
  s.addText("decode（一次一 token，逐步生成）", { x: MX + 3.4, y: 3.15, w: 7.6, h: 0.3, fontFace: BODY, fontSize: 13, bold: true, color: MEM, margin: 0 });

  const cw = 5.7, cy = 3.95, ch = 1.9;
  card(s, MX, cy, cw, ch, BG2);
  s.addText("Prefill — compute-bound", { x: MX + 0.35, y: cy + 0.25, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: COMP, margin: 0 });
  s.addText("整段 prompt 一起算 → 大 GEMM、AI 高，像訓練的 forward。", { x: MX + 0.35, y: cy + 0.75, w: cw - 0.7, h: 1.0, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, cy, cw, ch, BG2);
  s.addText("Decode — memory-bound", { x: x2 + 0.35, y: cy + 0.25, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: MEM, margin: 0 });
  s.addText("一次一 token、batch 小 → GEMV、AI≈1。今天聚焦這裡。", { x: x2 + 0.35, y: cy + 0.75, w: cw - 0.7, h: 1.0, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, margin: 0 });

  footer(s, 4);
})();

// =============================================================================
// Slide 5 — 為什麼 decode 是 memory-bound
// =============================================================================
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "04", "為什麼 decode 是 memory-bound", MEM);

  card(s, MX, 1.95, 11.9, 1.2, BG3);
  s.addText("每產生 1 個 token，就要把「整份權重」從 HBM 讀一遍。", { x: MX, y: 1.95, w: 11.9, h: 1.2, align: "center", valign: "middle", fontFace: HEAD, fontSize: 22, bold: true, color: INK, margin: 0 });

  const cy = 3.5, ch = 1.9, cw = 3.75, gap = 0.32;
  const stats = [
    { big: "14", unit: "GB", label: "7B 模型 fp16 權重", color: MEM },
    { big: "4.2", unit: "ms / token", label: "14 GB ÷ 3.35 TB/s", color: COMP },
    { big: "~240", unit: "tokens/s", label: "batch=1 的延遲上限", color: INK },
  ];
  stats.forEach((st, i) => {
    const x = MX + i * (cw + gap);
    card(s, x, cy, cw, ch, BG2);
    s.addText(st.big, { x, y: cy + 0.28, w: cw, h: 0.85, align: "center", fontFace: MONO, fontSize: 46, bold: true, color: st.color, margin: 0 });
    s.addText(st.unit, { x, y: cy + 1.12, w: cw, h: 0.35, align: "center", fontFace: MONO, fontSize: 14, color: MUTE, margin: 0 });
    s.addText(st.label, { x: x + 0.15, y: cy + 1.48, w: cw - 0.3, h: 0.35, align: "center", fontFace: BODY, fontSize: 12, color: INK, margin: 0 });
  });

  s.addText("這就是 S1 開場「<5% 算力利用率」的根源：時間都花在搬權重，tensor core 在閒置。", { x: MX, y: 5.7, w: 11.9, h: 0.4, align: "center", fontFace: BODY, fontSize: 15, bold: true, color: MEM, margin: 0 });
  footer(s, 5);
})();

// =============================================================================
// Slide 6 — KV cache
// =============================================================================
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "05", "KV cache：省了重算，換來頻寬與容量壓力", COMP);

  const cw = 5.7, ch = 2.5;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addText("是什麼", { x: MX + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: MEM, margin: 0 });
  s.addText("快取過去 token 的 Key / Value，讓每一步不必對整段序列重算 attention——是讓 decode 還能用的關鍵優化。", { x: MX + 0.35, y: 2.7, w: cw - 0.7, h: 1.6, fontFace: BODY, fontSize: 15, color: INK, lineSpacingMultiple: 1.25, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addText("代價", { x: x2 + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: WARN, margin: 0 });
  s.addText([
    { text: "隨序列長度線性增長", options: { bullet: true, breakLine: true } },
    { text: "每步都要讀它 → 頻寬壓力", options: { bullet: true, breakLine: true } },
    { text: "要存住它 → 容量壓力（限制 batch 與 context 長度）", options: { bullet: true } },
  ], { x: x2 + 0.35, y: 2.7, w: cw - 0.7, h: 1.7, fontFace: BODY, fontSize: 15, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 6, margin: 0 });

  card(s, MX, 4.65, 11.9, 0.95, BG3);
  s.addText([
    { text: "KV 大小  ", options: { fontFace: MONO, color: MUTE } },
    { text: "≈ 2 × layers × heads × head_dim × seq_len × batch × dtype", options: { fontFace: MONO, color: INK, bold: true } },
  ], { x: MX, y: 4.65, w: 11.9, h: 0.95, align: "center", valign: "middle", fontSize: 16, margin: 0 });

  s.addText("KV cache 越長，decode 每步要讀的越多 → 越往 memory-bound 走（也是長 context 變慢的主因）。", { x: MX, y: 5.85, w: 11.9, h: 0.4, align: "center", fontFace: BODY, fontSize: 14.5, color: COMP, margin: 0 });
  footer(s, 6);
})();

// =============================================================================
// Slide 7 — Batch 的魔法
// =============================================================================
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "06", "Batch 的魔法：throughput vs latency", MEM);

  card(s, MX, 1.95, 5.5, 3.9, BG2);
  s.addText("為什麼加大 batch 有用", { x: MX + 0.35, y: 2.2, w: 4.8, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: INK, margin: 0 });
  s.addText([
    { text: "batch=1：讀一次權重只服務一個請求 → 浪費", options: { bullet: true, breakLine: true } },
    { text: "batch=N：同一次權重讀取服務 N 個請求", options: { bullet: true, breakLine: true } },
    { text: "→ 吞吐近乎線性升、單步延遲幾乎不變", options: { bullet: true, breakLine: true, color: MEM } },
    { text: "直到變 compute-bound 或 KV cache 裝不下", options: { bullet: true, color: WARN } },
  ], { x: MX + 0.35, y: 2.75, w: 4.8, h: 2.8, fontFace: BODY, fontSize: 15, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 9, margin: 0 });

  s.addChart(pres.charts.LINE, [{
    name: "tokens/s（示意）",
    labels: ["1", "8", "32", "64", "128", "256", "512", "1024"],
    values: [3, 25, 100, 200, 400, 800, 930, 930],
  }], {
    x: 6.5, y: 2.0, w: 6.1, h: 3.7, chartColors: [MEM], lineSize: 3, lineSmooth: false,
    showLegend: false, showTitle: false,
    chartArea: { fill: { color: BG2 } }, plotArea: { fill: { color: BG2 } },
    catAxisLabelColor: MUTE, valAxisLabelColor: MUTE, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
    catAxisTitle: "batch size", showCatAxisTitle: true, catAxisTitleColor: MUTE, catAxisTitleFontSize: 11,
    valAxisTitle: "tokens/s（千）", showValAxisTitle: true, valAxisTitleColor: MUTE, valAxisTitleFontSize: 11,
    valGridLine: { color: LINE, size: 0.5 }, catGridLine: { style: "none" },
    lineDataSymbol: "circle", lineDataSymbolSize: 5,
  });

  s.addText("吞吐升、但「單一請求的延遲」沒變快——這就是 demo run.py 會跑給你看的曲線。", { x: MX, y: 6.05, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: MEM, margin: 0 });
  footer(s, 7);
})();

// =============================================================================
// Slide 8 — 換場：為什麼拿 ASR 當例子
// =============================================================================
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "07", "換個例子：為什麼拿 ASR 來講", COMP);

  // 波形 → 文字
  const heights = [0.4, 0.7, 0.5, 0.9, 0.6, 1.0, 0.5, 0.8, 0.45, 0.7, 0.55, 0.85];
  parallelTicks(s, MX + 0.2, 2.3, 12, 0.16, 1.0, 0.14, MEM, heights);
  s.addText("語音 (audio)", { x: MX + 0.2, y: 3.45, w: 3.6, h: 0.3, fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  s.addShape(pres.shapes.LINE, { x: 4.9, y: 2.8, w: 1.1, h: 0, line: { color: INK, width: 2, endArrowType: "triangle" } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.2, y: 2.45, w: 3.2, h: 0.8, rectRadius: 0.08, fill: { color: BG2 }, line: { color: LINE, width: 1 } });
  s.addText("「今天天氣很好」", { x: 6.2, y: 2.45, w: 3.2, h: 0.8, align: "center", valign: "middle", fontFace: HEAD, fontSize: 18, bold: true, color: INK, margin: 0 });
  s.addText("文字 (text)", { x: 6.2, y: 3.45, w: 3.2, h: 0.3, align: "center", fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });

  card(s, MX, 4.4, 11.9, 1.5, BG3);
  s.addText("同一個任務、兩種架構、推論速度差很多 → 最適合示範「不是 FLOPs 決定速度」", { x: MX + 0.4, y: 4.6, w: 11.1, h: 0.5, fontFace: HEAD, fontSize: 18, bold: true, color: INK, margin: 0 });
  s.addText([
    { text: "Whisper（attention encoder–decoder）", options: { color: MEM, breakLine: true } },
    { text: "vs  wav2vec2 / Conformer + CTC", options: { color: COMP } },
  ], { x: MX + 0.4, y: 5.15, w: 11.1, h: 0.6, fontFace: BODY, fontSize: 15, bold: true, lineSpacingMultiple: 1.1, margin: 0 });
  footer(s, 8);
})();

// =============================================================================
// Slide 9 — ASR 架構一：Whisper
// =============================================================================
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "08", "ASR 架構一：Whisper（attention 解碼）", MEM);

  // encoder（平行）
  card(s, MX, 2.0, 5.5, 3.6, BG2);
  s.addText("Encoder", { x: MX + 0.35, y: 2.25, w: 4.8, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: COMP, margin: 0 });
  s.addText("吃整段 audio spectrogram，一次平行算 → 大 GEMM、compute-bound、快。", { x: MX + 0.35, y: 2.75, w: 4.8, h: 1.0, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, margin: 0 });
  parallelTicks(s, MX + 0.7, 4.3, 10, 0.2, 0.9, 0.22, COMP);
  s.addText("平行（一次）", { x: MX + 0.35, y: 5.25, w: 4.8, h: 0.3, fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });

  // decoder（序列）
  const x2 = MX + 5.5 + 0.5;
  card(s, x2, 2.0, 5.5, 3.6, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: 2.0, w: 5.5, h: 0.12, fill: { color: WARN }, line: { type: "none" } });
  s.addText("Decoder ← 延遲主因", { x: x2 + 0.35, y: 2.25, w: 4.8, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: MEM, margin: 0 });
  s.addText("自迴歸、逐 token 生成 transcript → GEMV、memory-bound → 主宰延遲。", { x: x2 + 0.35, y: 2.75, w: 4.8, h: 1.0, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, margin: 0 });
  chain(s, x2 + 0.7, 4.5, 8, 0.34, 0.18, MEM);
  s.addText("序列（逐 token）", { x: x2 + 0.35, y: 5.25, w: 4.8, h: 0.3, fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });

  s.addText("準確度通常更高，但自迴歸 decoder 讓推論延遲偏高。", { x: MX, y: 5.95, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: MUTE, margin: 0 });
  footer(s, 9);
})();

// =============================================================================
// Slide 10 — ASR 架構二：CTC
// =============================================================================
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "09", "ASR 架構二：wav2vec2 / Conformer + CTC", COMP);

  card(s, MX, 2.0, 11.9, 2.95, BG2);
  s.addText("Encoder only — 無自迴歸 decode", { x: MX + 0.4, y: 2.25, w: 11.0, h: 0.45, fontFace: HEAD, fontSize: 19, bold: true, color: COMP, margin: 0 });
  s.addText("Encoder 一次輸出「所有 frame」的字元機率，再用 CTC 對齊成文字——沒有逐 token 的序列相依。", { x: MX + 0.4, y: 2.8, w: 11.0, h: 0.7, fontFace: BODY, fontSize: 15.5, color: INK, lineSpacingMultiple: 1.2, margin: 0 });
  parallelTicks(s, MX + 0.7, 3.55, 16, 0.18, 0.7, 0.22, COMP);
  s.addText("所有 frame 一次平行輸出 → 高度平行、推論快、適合低延遲串流", { x: MX + 0.7, y: 4.45, w: 10.5, h: 0.3, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });

  s.addText("和 Whisper 對照：少了「自迴歸 decode」這段 memory-bound 的序列瓶頸 → 同樣轉錄、速度差很多。", { x: MX, y: 5.5, w: 11.9, h: 0.7, fontFace: BODY, fontSize: 15.5, color: COMP, lineSpacingMultiple: 1.2, margin: 0 });
  footer(s, 10);
})();

// =============================================================================
// Slide 11 — ASR 對照總結表
// =============================================================================
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "10", "對照總結：架構決定推論速度", MEM);

  const head = (t, c) => ({ text: t, options: { fill: { color: BG3 }, color: c || INK, bold: true, fontSize: 14 } });
  const cell = (t, c) => ({ text: t, options: { fill: { color: BG2 }, color: c || INK } });
  const rows = [
    [head("維度"), head("Whisper（attention）", MEM), head("CTC（wav2vec2）", COMP)],
    [cell("decode 方式"), cell("自迴歸、逐 token", MEM), cell("無，一次輸出所有 frame", COMP)],
    [cell("可平行度"), cell("decoder 低（序列相依）"), cell("高")],
    [cell("推論瓶頸"), cell("decoder：memory-bound", MEM), cell("encoder：compute-bound", COMP)],
    [cell("相對推論速度"), cell("較慢"), cell("較快")],
    [cell("常見取捨"), cell("準確度 / 可生成式輸出"), cell("低延遲 / 串流")],
  ];
  s.addTable(rows, {
    x: 1.0, y: 1.95, w: 11.3, colW: [2.7, 4.3, 4.3], rowH: 0.6,
    fontFace: BODY, fontSize: 14, color: INK, valign: "middle", align: "left",
    border: { type: "solid", color: LINE, pt: 1 },
  });
  s.addText("決定速度的是「記憶體存取型態與可平行度」，不是 FLOPs 總量。", { x: 1.0, y: 5.95, w: 11.3, h: 0.4, fontFace: BODY, fontSize: 16, bold: true, color: MEM, margin: 0 });
  footer(s, 11);
})();

// =============================================================================
// Slide 12 — Demo 預告
// =============================================================================
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "11", "動手：decode_memory_bound demo", COMP);

  card(s, MX, 1.95, 5.7, 3.9, BG2);
  s.addText("run.py — decode batch sweep", { x: MX + 0.35, y: 2.2, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: MEM, margin: 0 });
  s.addText([
    { text: "量不同 batch 的單步延遲與吞吐", options: { bullet: true, breakLine: true } },
    { text: "看 step_ms 幾乎不變、tokens/s 線性升", options: { bullet: true } },
  ], { x: MX + 0.35, y: 2.7, w: 5.0, h: 1.2, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 6, margin: 0 });
  s.addText("$ python run.py --peak-bw 3.35", { x: MX + 0.35, y: 4.6, w: 5.0, h: 0.55, fontFace: MONO, fontSize: 13, color: MEM, fill: { color: "0A1322" }, valign: "middle", margin: 8 });

  const x2 = MX + 5.7 + 0.5;
  card(s, x2, 1.95, 5.7, 3.9, BG2);
  s.addText("asr_proxy.py — 平行 vs 序列", { x: x2 + 0.35, y: 2.2, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: COMP, margin: 0 });
  s.addText([
    { text: "同樣 FLOPs：encoder 一次平行、decoder 逐步", options: { bullet: true, breakLine: true } },
    { text: "看 decoder / encoder 慢數倍～數十倍", options: { bullet: true } },
  ], { x: x2 + 0.35, y: 2.7, w: 5.0, h: 1.2, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 6, margin: 0 });
  s.addText("$ python asr_proxy.py --frames 256", { x: x2 + 0.35, y: 4.6, w: 5.0, h: 0.55, fontFace: MONO, fontSize: 13, color: COMP, fill: { color: "0A1322" }, valign: "middle", margin: 8 });

  s.addText("兩個 demo 的效應都需在 GPU 上才明顯（CPU 權重落 cache，看不出 HBM 頻寬瓶頸）。", { x: MX, y: 6.05, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 14, color: MUTE, margin: 0 });
  footer(s, 12);
})();

// =============================================================================
// Slide 13 — 收束 + S4
// =============================================================================
(() => {
  const s = pres.addSlide(); base(s);
  s.addText("本場帶走三件事", { x: MX, y: 0.8, w: 11.9, h: 0.7, fontFace: HEAD, fontSize: 32, bold: true, color: INK, margin: 0 });

  const items = [
    { n: "1", t: "訓練 compute-bound（怕裝不下）；decode memory-bound。", d: "decode 每 token 重讀整份權重，KV cache 還會加劇。", c: MEM },
    { n: "2", t: "加大 batch 換吞吐，救不了單一請求的延遲。", d: "同一次權重讀取攤平給更多請求，直到 compute-bound。", c: COMP },
    { n: "3", t: "ASR：平行 vs 自迴歸的架構差，決定推論速度。", d: "Whisper decoder 慢、CTC 快——不是 FLOPs 總量決定。", c: MEM },
  ];
  items.forEach((it, i) => {
    const y = 1.9 + i * 1.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 0.7, h: 0.7, rectRadius: 0.1, fill: { color: it.c }, line: { type: "none" } });
    s.addText(it.n, { x: MX, y, w: 0.7, h: 0.7, align: "center", valign: "middle", fontFace: MONO, fontSize: 26, bold: true, color: BG, margin: 0 });
    s.addText(it.t, { x: MX + 1.0, y: y - 0.05, w: 11.0, h: 0.5, fontFace: HEAD, fontSize: 19, bold: true, color: INK, margin: 0 });
    s.addText(it.d, { x: MX + 1.0, y: y + 0.42, w: 11.0, h: 0.4, fontFace: BODY, fontSize: 14, color: MUTE, margin: 0 });
  });

  card(s, MX, 5.6, 11.9, 1.1, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 5.6, w: 0.12, h: 1.1, fill: { color: MEM }, line: { type: "none" } });
  s.addText([
    { text: "下一場 S4　", options: { color: MEM, bold: true } },
    { text: "資料搬遷的關卡與記憶體方案（PCIe / NVLink / GPUDirect Storage / Unified Memory）", options: { color: INK } },
  ], { x: MX + 0.4, y: 5.6, w: 11.3, h: 1.1, valign: "middle", fontFace: BODY, fontSize: 16, margin: 0 });
})();

pres.writeFile({ fileName: "../s3_train_vs_infer_asr.pptx" }).then((f) => console.log("written:", f));
