// S1 — 為什麼會慢？Roofline 與記憶體階層
// 產生 ../s1_roofline.pptx。風格：深色「矽晶」，amber=compute、cyan=memory。
// 執行：node generate_s1.js
const pptxgen = require("pptxgenjs");

// ---- palette ----------------------------------------------------------------
const BG = "0E1726";   // 主背景（深藍灰）
const BG2 = "16233A";  // 卡片面板
const BG3 = "1C2E4A";  // 較亮面板
const INK = "EAF1FB";  // 主要文字（近白）
const MUTE = "8FA6C4"; // 次要文字
const LINE = "2A3D5C"; // 細邊／格線
const MEM = "38BDF8";  // cyan：記憶體 / 頻寬（本系列主題色）
const COMP = "F59E0B"; // amber：算力
const WARN = "FB7185"; // rose：瓶頸標示
const MEMTINT = "10455F";
const WARNTINT = "4A2433";

const HEAD = "PingFang TC";
const BODY = "PingFang TC";
const MONO = "Menlo";

// ---- layout -----------------------------------------------------------------
const W = 13.33, H = 7.5;
const MX = 0.7;
const TITLE_Y = 0.62;
const FOOT_Y = 7.05;
const TOTAL = 11;

const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 });

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "GPU 記憶體與資料搬遷讀書會";
pres.title = "S1 — 為什麼會慢？Roofline 與記憶體階層";

// ---- helpers ----------------------------------------------------------------
function base(slide) {
  slide.background = { color: BG };
}

function runningHeader(slide) {
  slide.addText("GPU 記憶體與資料搬遷讀書會 · S1", {
    x: W - 5.2, y: 0.3, w: 4.5, h: 0.3, align: "right",
    fontFace: BODY, fontSize: 10, color: MUTE, margin: 0,
  });
}

function footer(slide, n) {
  slide.addText("為什麼會慢？Roofline 與記憶體階層", {
    x: MX, y: FOOT_Y, w: 8, h: 0.3, fontFace: BODY, fontSize: 9, color: LINE === "" ? MUTE : "5C7299", margin: 0,
  });
  slide.addText(`${n} / ${TOTAL}`, {
    x: W - 1.6, y: FOOT_Y, w: 0.9, h: 0.3, align: "right",
    fontFace: MONO, fontSize: 9, color: "5C7299", margin: 0,
  });
}

function header(slide, num, title, accent) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: MX, y: TITLE_Y, w: 0.62, h: 0.62, rectRadius: 0.08,
    fill: { color: accent }, line: { type: "none" }, shadow: shadow(),
  });
  slide.addText(num, {
    x: MX, y: TITLE_Y, w: 0.62, h: 0.62, align: "center", valign: "middle",
    fontFace: MONO, fontSize: 22, bold: true, color: BG, margin: 0,
  });
  // 標題框可用近全寬：running header 在標題「上方」(y 0.3–0.6)，垂直不重疊
  slide.addText(title, {
    x: MX + 0.85, y: TITLE_Y, w: W - MX - 0.85 - 0.5, h: 0.62, valign: "middle",
    fontFace: HEAD, fontSize: 29, bold: true, color: INK, margin: 0,
  });
}

function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.1,
    fill: { color: fill }, line: { color: LINE, width: 1 }, shadow: shadow(),
  });
}

function coreGrid(slide, x, y, cols, rows, cell, gap, color) {
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      slide.addShape(pres.shapes.RECTANGLE, {
        x: x + c * (cell + gap), y: y + r * (cell + gap), w: cell, h: cell,
        fill: { color }, line: { type: "none" },
      });
    }
  }
}

// =============================================================================
// Slide 1 — 標題
// =============================================================================
(() => {
  const s = pres.addSlide();
  base(s);
  // 右側 roofline 裝飾字符（cyan 斜線 + amber 平頂 + 兩點）
  s.addShape(pres.shapes.LINE, { x: 8.7, y: 2.0, w: 2.0, h: 1.8, flipV: true, line: { color: MEM, width: 3 } });
  s.addShape(pres.shapes.LINE, { x: 10.7, y: 2.0, w: 2.1, h: 0, line: { color: COMP, width: 3 } });
  s.addShape(pres.shapes.OVAL, { x: 9.25, y: 3.05, w: 0.18, h: 0.18, fill: { color: MEM }, line: { type: "none" } });
  s.addShape(pres.shapes.OVAL, { x: 11.55, y: 1.88, w: 0.18, h: 0.18, fill: { color: COMP }, line: { type: "none" } });

  s.addText("GPU 記憶體與資料搬遷讀書會  ·  S1 / 共 4 場", {
    x: MX, y: 1.7, w: 9, h: 0.4, fontFace: BODY, fontSize: 15, color: MEM, bold: true, charSpacing: 1, margin: 0,
  });
  s.addText([
    { text: "為什麼會慢？", options: { breakLine: true } },
    { text: "Roofline 與記憶體階層", options: {} },
  ], { x: MX, y: 2.45, w: 11.5, h: 2.0, fontFace: HEAD, fontSize: 46, bold: true, color: INK, lineSpacingMultiple: 1.05, margin: 0 });

  s.addText("從計算機組織與 GPU 架構，判斷一個運算是被「算力」還是「記憶體頻寬」卡住。", {
    x: MX, y: 4.7, w: 11.0, h: 0.6, fontFace: BODY, fontSize: 18, color: MUTE, margin: 0,
  });
  s.addText("聽眾：data science 背景 · 目標：建立「先問 compute-bound 還是 memory-bound」的反射", {
    x: MX, y: 5.5, w: 11.0, h: 0.4, fontFace: BODY, fontSize: 13, color: "5C7299", margin: 0,
  });
})();

// =============================================================================
// Slide 2 — 開場謎題
// =============================================================================
(() => {
  const s = pres.addSlide();
  base(s); runningHeader(s);
  header(s, "01", "一個開場謎題", MEM);

  s.addText([
    { text: "同一張 H100，跑 batch=1 的 LLM 解碼（decode）時，", options: { breakLine: true } },
    { text: "tensor core 的算力利用率常常不到 5%。", options: { breakLine: true, bold: true, color: INK } },
    { text: "", options: { breakLine: true, fontSize: 8 } },
    { text: "我們買了全世界最貴的「算力」，", options: { breakLine: true } },
    { text: "卻幾乎用不到——為什麼？", options: { bold: true, color: MEM } },
  ], { x: MX, y: 1.95, w: 6.6, h: 3.2, fontFace: BODY, fontSize: 19, color: MUTE, lineSpacingMultiple: 1.25, margin: 0, valign: "top" });

  card(s, 8.0, 1.95, 4.6, 3.3, BG2);
  s.addText("< 5%", { x: 8.0, y: 2.25, w: 4.6, h: 1.5, align: "center", fontFace: MONO, fontSize: 80, bold: true, color: COMP, margin: 0 });
  s.addText("batch=1 解碼時的算力利用率", { x: 8.0, y: 3.85, w: 4.6, h: 0.5, align: "center", fontFace: BODY, fontSize: 16, color: INK, margin: 0 });
  s.addText("（瓶頸不在算力，而在資料搬運）", { x: 8.0, y: 4.35, w: 4.6, h: 0.5, align: "center", fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });

  s.addText("👉 答案不是「算力不夠」，而是「資料搬不夠快」。本場就在拆解這件事。", {
    x: MX, y: 5.7, w: 11.9, h: 0.5, fontFace: BODY, fontSize: 16, bold: true, color: MEM, margin: 0,
  });
  footer(s, 2);
})();

// =============================================================================
// Slide 3 — 兩種「慢」
// =============================================================================
(() => {
  const s = pres.addSlide();
  base(s); runningHeader(s);
  header(s, "02", "兩種「慢」：你被誰卡住？", MEM);

  const cy = 1.95, ch = 3.7, cw = 5.7;
  // Compute-bound
  card(s, MX, cy, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: cy, w: cw, h: 0.12, fill: { color: COMP }, line: { type: "none" } });
  s.addText("Compute-bound", { x: MX + 0.35, y: cy + 0.3, w: cw - 0.7, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: COMP, margin: 0 });
  s.addText("算力受限", { x: MX + 0.35, y: cy + 0.85, w: cw - 0.7, h: 0.4, fontFace: BODY, fontSize: 15, color: MUTE, margin: 0 });
  s.addText([
    { text: "瓶頸：峰值 FLOPS（算得不夠快）", options: { bullet: true, breakLine: true } },
    { text: "症狀：算力利用率高、加大 batch 也快不了", options: { bullet: true, breakLine: true } },
    { text: "典型：大矩陣乘法、模型訓練", options: { bullet: true } },
  ], { x: MX + 0.35, y: cy + 1.45, w: cw - 0.7, h: 2.0, fontFace: BODY, fontSize: 15.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 6, margin: 0 });

  // Memory-bound
  const x2 = MX + cw + 0.5;
  card(s, x2, cy, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: cy, w: cw, h: 0.12, fill: { color: MEM }, line: { type: "none" } });
  s.addText("Memory-bound", { x: x2 + 0.35, y: cy + 0.3, w: cw - 0.7, h: 0.5, fontFace: HEAD, fontSize: 22, bold: true, color: MEM, margin: 0 });
  s.addText("頻寬受限", { x: x2 + 0.35, y: cy + 0.85, w: cw - 0.7, h: 0.4, fontFace: BODY, fontSize: 15, color: MUTE, margin: 0 });
  s.addText([
    { text: "瓶頸：記憶體頻寬（資料搬不夠快）", options: { bullet: true, breakLine: true } },
    { text: "症狀：算力利用率低、加大 batch 吞吐會上升", options: { bullet: true, breakLine: true } },
    { text: "典型：自迴歸解碼、element-wise 運算", options: { bullet: true } },
  ], { x: x2 + 0.35, y: cy + 1.45, w: cw - 0.7, h: 2.0, fontFace: BODY, fontSize: 15.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 6, margin: 0 });

  s.addText("怎麼判斷自己在哪一邊？用一把尺：算術強度（下一頁）。", {
    x: MX, y: 5.95, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: MUTE, margin: 0,
  });
  footer(s, 3);
})();

// =============================================================================
// Slide 4 — CPU vs GPU 設計哲學
// =============================================================================
(() => {
  const s = pres.addSlide();
  base(s); runningHeader(s);
  header(s, "03", "CPU vs GPU：兩種設計哲學", COMP);

  const cy = 1.95, ch = 3.7, cw = 5.7;
  card(s, MX, cy, cw, ch, BG2);
  s.addText("CPU — latency-oriented", { x: MX + 0.35, y: cy + 0.3, w: cw - 0.7, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: INK, margin: 0 });
  s.addText("少數強核、大 cache、亂序執行；把「單一任務」做到最快。", { x: MX + 0.35, y: cy + 0.85, w: cw - 0.7, h: 0.9, fontFace: BODY, fontSize: 15, color: MUTE, lineSpacingMultiple: 1.2, margin: 0 });
  coreGrid(s, MX + 1.45, cy + 2.05, 2, 2, 0.62, 0.16, COMP);

  const x2 = MX + cw + 0.5;
  card(s, x2, cy, cw, ch, BG2);
  s.addText("GPU — throughput-oriented", { x: x2 + 0.35, y: cy + 0.3, w: cw - 0.7, h: 0.5, fontFace: HEAD, fontSize: 20, bold: true, color: INK, margin: 0 });
  s.addText("海量簡單核心、用大量 thread 互相掩護延遲；把「總吞吐」做到最大。", { x: x2 + 0.35, y: cy + 0.85, w: cw - 0.7, h: 0.9, fontFace: BODY, fontSize: 15, color: MUTE, lineSpacingMultiple: 1.2, margin: 0 });
  coreGrid(s, x2 + 0.95, cy + 2.0, 12, 5, 0.16, 0.07, MEM);

  s.addText("關鍵：GPU 靠「同時跑很多」藏住記憶體延遲——但前提是資料餵得上。否則核心再多也在空等。", {
    x: MX, y: 5.95, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: MEM, margin: 0,
  });
  footer(s, 4);
})();

// =============================================================================
// Slide 5 — 算術強度
// =============================================================================
(() => {
  const s = pres.addSlide();
  base(s); runningHeader(s);
  header(s, "04", "一把尺：算術強度 (Arithmetic Intensity)", MEM);

  card(s, MX, 1.95, 11.9, 1.7, BG2);
  s.addText([
    { text: "AI  =  ", options: { fontFace: MONO, color: INK } },
    { text: "完成運算的 FLOPs", options: { color: COMP } },
    { text: "  ÷  ", options: { fontFace: MONO, color: MUTE } },
    { text: "需搬動的 Bytes", options: { color: MEM } },
  ], { x: MX, y: 2.15, w: 11.9, h: 0.8, align: "center", fontFace: HEAD, fontSize: 30, bold: true, margin: 0 });
  s.addText("單位：FLOPs / Byte —— 每搬 1 個位元組，能換到幾次浮點運算", {
    x: MX, y: 2.95, w: 11.9, h: 0.5, align: "center", fontFace: BODY, fontSize: 14, color: MUTE, margin: 0,
  });

  const cy = 4.0, ch = 1.5, cw = 5.7;
  card(s, MX, cy, cw, ch, BG3);
  s.addText("AI 高 → compute-bound", { x: MX + 0.35, y: cy + 0.25, w: cw - 0.7, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: COMP, margin: 0 });
  s.addText("每搬一點資料就做很多運算，餵得飽算力。例：大方陣 GEMM，AI ≈ 數百。", { x: MX + 0.35, y: cy + 0.72, w: cw - 0.7, h: 0.7, fontFace: BODY, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.15, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, cy, cw, ch, BG3);
  s.addText("AI 低 → memory-bound", { x: x2 + 0.35, y: cy + 0.25, w: cw - 0.7, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: MEM, margin: 0 });
  s.addText("大多時間在等資料。例：矩陣×向量 GEMV，AI ≈ 1–2（fp16）。", { x: x2 + 0.35, y: cy + 0.72, w: cw - 0.7, h: 0.7, fontFace: BODY, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.15, margin: 0 });

  footer(s, 5);
})();

// =============================================================================
// Slide 6 — Roofline 圖（手繪）
// =============================================================================
(() => {
  const s = pres.addSlide();
  base(s); runningHeader(s);
  header(s, "05", "Roofline：一眼看出你的天花板", COMP);

  // plot box
  const x0 = 1.5, y0 = 5.7, pw = 6.5, ph = 3.2;
  const topY = y0 - ph;              // 2.5
  const xr = x0 + 0.55 * pw;         // ridge x
  // axes
  s.addShape(pres.shapes.LINE, { x: x0, y: topY, w: 0, h: ph, line: { color: MUTE, width: 1.5 } });
  s.addShape(pres.shapes.LINE, { x: x0, y: y0, w: pw, h: 0, line: { color: MUTE, width: 1.5 } });
  // bandwidth roof (diagonal up-right) + compute roof (flat)
  s.addShape(pres.shapes.LINE, { x: x0, y: topY, w: xr - x0, h: ph, flipV: true, line: { color: MEM, width: 3.5 } });
  s.addShape(pres.shapes.LINE, { x: xr, y: topY, w: x0 + pw - xr, h: 0, line: { color: COMP, width: 3.5 } });
  // ridge marker (dashed vertical)
  s.addShape(pres.shapes.LINE, { x: xr, y: topY, w: 0, h: ph, line: { color: "5C7299", width: 1, dashType: "dash" } });
  // data dots
  s.addShape(pres.shapes.OVAL, { x: x0 + 0.9 - 0.09, y: y0 - 1.0 - 0.09, w: 0.18, h: 0.18, fill: { color: MEM }, line: { color: BG, width: 1.5 } });
  s.addShape(pres.shapes.OVAL, { x: xr + 1.3 - 0.09, y: topY + 0.15 - 0.09, w: 0.18, h: 0.18, fill: { color: COMP }, line: { color: BG, width: 1.5 } });
  // labels
  s.addText("可達成\n算力", { x: 0.55, y: topY - 0.1, w: 0.9, h: 0.6, fontFace: BODY, fontSize: 11, color: MUTE, align: "center", margin: 0 });
  s.addText("算術強度 AI (FLOPs/Byte) →", { x: x0, y: y0 + 0.12, w: pw, h: 0.3, align: "center", fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });
  s.addText("ridge point", { x: xr - 1.0, y: topY - 0.42, w: 2.0, h: 0.3, align: "center", fontFace: MONO, fontSize: 10, color: "9FB2CC", margin: 0 });
  s.addText("頻寬上限", { x: x0 + 0.5, y: y0 - 1.7, w: 1.7, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: MEM, margin: 0 });
  s.addText("算力上限", { x: xr + 0.5, y: topY - 0.05, w: 1.7, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: COMP, margin: 0 });
  s.addText("GEMV", { x: x0 + 0.55, y: y0 - 0.95, w: 1.2, h: 0.25, fontFace: MONO, fontSize: 9.5, color: MEM, margin: 0 });
  s.addText("大 GEMM", { x: xr + 1.45, y: topY + 0.05, w: 1.4, h: 0.25, fontFace: MONO, fontSize: 9.5, color: COMP, margin: 0 });

  // 右側說明
  const rx = 9.0;
  card(s, rx, 1.95, 3.6, 4.0, BG2);
  s.addText([
    { text: "斜線 = 頻寬上限\n(memory-bound)", options: { bullet: { code: "2022" }, color: MEM, breakLine: true } },
    { text: "平頂 = 算力上限\n(compute-bound)", options: { bullet: { code: "2022" }, color: COMP, breakLine: true } },
    { text: "轉折 = ridge point", options: { bullet: { code: "2022" }, color: INK, breakLine: true } },
    { text: "點落在斜線上 → 受頻寬限，換更強的算力也沒用", options: { bullet: { code: "2022" }, color: INK } },
  ], { x: rx + 0.3, y: 2.25, w: 3.0, h: 3.4, fontFace: BODY, fontSize: 14.5, lineSpacingMultiple: 1.15, paraSpaceAfter: 12, margin: 0, valign: "top" });

  footer(s, 6);
})();

// =============================================================================
// Slide 7 — Ridge point 範例 H100
// =============================================================================
(() => {
  const s = pres.addSlide();
  base(s); runningHeader(s);
  header(s, "06", "Ridge point 範例：H100", MEM);

  const cy = 2.1, ch = 2.4, cw = 3.75, gap = 0.32;
  const stats = [
    { big: "990", unit: "TFLOPS", label: "BF16 tensor 峰值算力", color: COMP },
    { big: "3.35", unit: "TB/s", label: "HBM3 記憶體頻寬", color: MEM },
    { big: "≈300", unit: "FLOPs/Byte", label: "ridge point = 算力 ÷ 頻寬", color: INK },
  ];
  stats.forEach((st, i) => {
    const x = MX + i * (cw + gap);
    card(s, x, cy, cw, ch, BG2);
    s.addText(st.big, { x, y: cy + 0.35, w: cw, h: 1.0, align: "center", fontFace: MONO, fontSize: 56, bold: true, color: st.color, margin: 0 });
    s.addText(st.unit, { x, y: cy + 1.4, w: cw, h: 0.4, align: "center", fontFace: MONO, fontSize: 16, color: MUTE, margin: 0 });
    s.addText(st.label, { x: x + 0.2, y: cy + 1.85, w: cw - 0.4, h: 0.45, align: "center", fontFace: BODY, fontSize: 12.5, color: INK, margin: 0 });
  });

  s.addText("算術強度 < 300 的運算，再強的 tensor core 也餵不飽——你只能拿到「頻寬的速度」。", {
    x: MX, y: 4.95, w: 11.9, h: 0.5, align: "center", fontFace: BODY, fontSize: 17, bold: true, color: MEM, margin: 0,
  });
  s.addText("（數字為約略值，以官方規格為準）", {
    x: MX, y: 5.55, w: 11.9, h: 0.35, align: "center", fontFace: BODY, fontSize: 11, color: "5C7299", margin: 0,
  });
  footer(s, 7);
})();

// =============================================================================
// Slide 8 — 記憶體階層表
// =============================================================================
(() => {
  const s = pres.addSlide();
  base(s); runningHeader(s);
  header(s, "07", "記憶體階層：每遠一站，慢一個數量級", MEM);

  const hopt = { fontFace: BODY, fontSize: 13, color: INK, valign: "middle", border: { type: "solid", color: LINE, pt: 1 } };
  const head = (t) => ({ text: t, options: { fill: { color: BG3 }, color: INK, bold: true, fontSize: 13.5 } });
  const cell = (t, fill) => ({ text: t, options: fill ? { fill: { color: fill } } : { fill: { color: BG2 } } });
  const rows = [
    [head("層級"), head("約略頻寬"), head("備註")],
    [cell("暫存器 Register"), cell("數十 TB/s"), cell("晶片內最快")],
    [cell("共享記憶體 / L1"), cell("數十 TB/s"), cell("程式可控（tiling 的關鍵）")],
    [cell("L2 cache"), cell("數 ~ 數十 TB/s"), cell("晶片內快取")],
    [cell("HBM（GPU 全域記憶體）", MEMTINT), cell("2 ~ 4.8 TB/s", MEMTINT), cell("A100~2 / H100~3.35 / H200~4.8", MEMTINT)],
    [cell("NVLink（GPU↔GPU / C2C）"), cell("~900 GB/s"), cell("比 PCIe 快一個量級")],
    [cell("PCIe（Host↔Device）", WARNTINT), cell("Gen4 ~32 / Gen5 ~64 GB/s", WARNTINT), cell("最常被跨越的瓶頸", WARNTINT)],
    [cell("CPU DRAM（DDR5）"), cell("~50 ~ 100+ GB/s"), cell("主機記憶體")],
    [cell("NVMe SSD"), cell("~3 ~ 7 GB/s"), cell("資料集 / 權重來源")],
  ];
  s.addTable(rows, {
    x: 1.0, y: 1.85, w: 11.3, colW: [3.7, 3.3, 4.3], rowH: 0.42,
    ...hopt, align: "left",
  });
  s.addText("HBM 是 TB/s、PCIe 是 GB/s——差約 100×。這就是為什麼「跨 PCIe 進出」這麼貴。", {
    x: 1.0, y: 6.35, w: 11.3, h: 0.4, fontFace: BODY, fontSize: 14, color: MEM, margin: 0,
  });
  footer(s, 8);
})();

// =============================================================================
// Slide 9 — 心法 + pipeline 瓶頸圖
// =============================================================================
(() => {
  const s = pres.addSlide();
  base(s); runningHeader(s);
  header(s, "08", "心法：瓶頸＝資料必經的最慢那段路", MEM);

  const boxes = [
    { t: "NVMe SSD", sub: "~7 GB/s", c: BG2 },
    { t: "CPU DRAM", sub: "~100 GB/s", c: BG2 },
    { t: "GPU HBM", sub: "~3.3 TB/s", c: BG2 },
    { t: "運算單元\nSM / Register", sub: "數十 TB/s", c: BG2 },
  ];
  const bw = 2.5, bh = 1.5, by = 2.5, gap = 0.85;
  const startX = (W - (boxes.length * bw + (boxes.length - 1) * gap)) / 2;
  boxes.forEach((b, i) => {
    const x = startX + i * (bw + gap);
    card(s, x, by, bw, bh, b.c);
    s.addText(b.t, { x, y: by + 0.3, w: bw, h: 0.7, align: "center", fontFace: HEAD, fontSize: 17, bold: true, color: INK, margin: 0 });
    s.addText(b.sub, { x, y: by + 1.0, w: bw, h: 0.35, align: "center", fontFace: MONO, fontSize: 13, color: MEM, margin: 0 });
    // 連接器
    if (i < boxes.length - 1) {
      const cxL = x + bw, cxR = startX + (i + 1) * (bw + gap);
      const isPCIe = i === 1; // DRAM -> HBM 跨 PCIe
      s.addShape(pres.shapes.LINE, {
        x: cxL + 0.05, y: by + bh / 2, w: cxR - cxL - 0.1, h: 0,
        line: { color: isPCIe ? WARN : "5C7299", width: isPCIe ? 4 : 2, endArrowType: "triangle" },
      });
      if (isPCIe) {
        s.addText("PCIe ~32 GB/s", { x: cxL - 0.2, y: by - 0.5, w: cxR - cxL + 0.4, h: 0.3, align: "center", fontFace: MONO, fontSize: 11, bold: true, color: WARN, margin: 0 });
        s.addText("← 瓶頸", { x: cxL - 0.2, y: by + bh + 0.1, w: cxR - cxL + 0.4, h: 0.3, align: "center", fontFace: BODY, fontSize: 11, bold: true, color: WARN, margin: 0 });
      }
    }
  });

  s.addText("HBM 內部再快，只要資料被迫反覆跨 PCIe 進出，整段就被最慢的 PCIe 拖住——這就是「memory 一進一出」的代價。", {
    x: MX, y: 5.5, w: 11.9, h: 0.7, align: "center", fontFace: BODY, fontSize: 16, color: INK, lineSpacingMultiple: 1.2, margin: 0,
  });
  footer(s, 9);
})();

// =============================================================================
// Slide 10 — Demo 預告
// =============================================================================
(() => {
  const s = pres.addSlide();
  base(s); runningHeader(s);
  header(s, "09", "動手：roofline_mini demo", COMP);

  card(s, MX, 1.95, 5.7, 3.9, BG2);
  s.addText("這個 demo 做什麼", { x: MX + 0.35, y: 2.2, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: INK, margin: 0 });
  s.addText([
    { text: "跑不同形狀的矩陣乘法", options: { bullet: true, breakLine: true } },
    { text: "量「達成 TFLOPS」與「算術強度」", options: { bullet: true, breakLine: true } },
    { text: "看瘦長矩陣如何掉進 memory-bound", options: { bullet: true } },
  ], { x: MX + 0.35, y: 2.7, w: 5.0, h: 1.6, fontFace: BODY, fontSize: 15, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 6, margin: 0 });
  s.addText("$ python run.py \\\n    --peak-tflops 990 --peak-bw 3.35", {
    x: MX + 0.35, y: 4.5, w: 5.0, h: 0.9, fontFace: MONO, fontSize: 13, color: MEM, fill: { color: "0A1322" }, align: "left", valign: "middle", margin: 8,
  });

  const x2 = MX + 5.7 + 0.5;
  card(s, x2, 1.95, 5.7, 3.9, BG2);
  s.addText("會看到（示意）", { x: x2 + 0.35, y: 2.2, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: INK, margin: 0 });
  const trows = [
    [{ text: "shape", options: { color: MUTE, bold: true } }, { text: "AI", options: { color: MUTE, bold: true, align: "right" } }, { text: "TFLOPS", options: { color: MUTE, bold: true, align: "right" } }, { text: "bound", options: { color: MUTE, bold: true } }],
    [{ text: "GEMV" }, { text: "1.0", options: { align: "right" } }, { text: "8", options: { align: "right" } }, { text: "memory", options: { color: MEM } }],
    [{ text: "skinny 64" }, { text: "1.9", options: { align: "right" } }, { text: "61", options: { align: "right" } }, { text: "memory", options: { color: MEM } }],
    [{ text: "square 2048" }, { text: "341", options: { align: "right" } }, { text: "612", options: { align: "right" } }, { text: "compute", options: { color: COMP } }],
    [{ text: "square 8192" }, { text: "1365", options: { align: "right" } }, { text: "780", options: { align: "right" } }, { text: "compute", options: { color: COMP } }],
  ];
  s.addTable(trows, {
    x: x2 + 0.35, y: 2.75, w: 5.0, colW: [1.9, 0.9, 1.1, 1.1], rowH: 0.42,
    fontFace: MONO, fontSize: 12.5, color: INK, valign: "middle",
    fill: { color: "0A1322" }, border: { type: "solid", color: LINE, pt: 1 },
  });
  s.addText("（數字示意；實際依 GPU 而定）", { x: x2 + 0.35, y: 5.05, w: 5.0, h: 0.3, fontFace: BODY, fontSize: 10.5, color: "5C7299", margin: 0 });

  s.addText("你會親眼看到：同一張卡，矩陣形狀就決定你落在 roofline 的哪一區。", {
    x: MX, y: 6.05, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: MEM, margin: 0,
  });
  footer(s, 10);
})();

// =============================================================================
// Slide 11 — 收束 + 下一場
// =============================================================================
(() => {
  const s = pres.addSlide();
  base(s);
  s.addText("本場帶走三件事", { x: MX, y: 0.8, w: 11.9, h: 0.7, fontFace: HEAD, fontSize: 32, bold: true, color: INK, margin: 0 });

  const items = [
    { n: "1", t: "先問：compute-bound 還是 memory-bound？", d: "用算術強度（FLOPs/Byte）這把尺判斷。", c: MEM },
    { n: "2", t: "記憶體階層每遠一站，慢一個數量級。", d: "瓶頸＝資料必經的最慢那段路（常是 PCIe）。", c: COMP },
    { n: "3", t: "很多 inference 的慢，是頻寬問題、不是算力問題。", d: "換更貴的算力救不了 memory-bound。", c: MEM },
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
    { text: "下一場 S2　", options: { color: MEM, bold: true } },
    { text: "GPU 架構與 HBM：資料在晶片內怎麼走（SM / warp / tensor core / shared memory）", options: { color: INK } },
  ], { x: MX + 0.4, y: 5.6, w: 11.3, h: 1.1, valign: "middle", fontFace: BODY, fontSize: 16, margin: 0 });
})();

pres.writeFile({ fileName: "../s1_roofline.pptx" }).then((f) => {
  console.log("written:", f);
});
