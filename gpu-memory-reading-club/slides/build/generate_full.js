// 合輯 — GPU 記憶體與資料搬遷讀書會：S1–S5 全系列重編版
// 產生 ../full_series.pptx。沿用深色「矽晶」主題。
//
// 與單場版的差異（去重與重排）：
//   - 刪除各場「回顧上一場 / 下一場預告」頁
//   - 記憶體階層：S1 全景 + S2 晶片內 → 一張表（加「誰管理」欄）
//   - CPU vs GPU 哲學：S1 + S5「不是快是寬」→ 一頁
//   - 進出站：S2 pinned/pageable + S4 H2D/PCIe → 一頁
//   - tiling 兩頁合一；ASR 已移除（聚焦硬體×Transformer）；NVLink/UVM 已移除；比較表+決策樹合一
//   - 五頁 demo 預告 → 一頁總表（含實測數字）
//   - S5 的 Amdahl 前移到「機器」篇；其餘共同演化內容收在最後一篇
const pptxgen = require("pptxgenjs");

const BG = "0E1726", BG2 = "16233A", BG3 = "1C2E4A";
const INK = "EAF1FB", MUTE = "8FA6C4", LINE = "2A3D5C", FOOTC = "5C7299";
const MEM = "38BDF8", COMP = "F59E0B", WARN = "FB7185", GOOD = "34D399";
const MEMTINT = "10455F", COMPTINT = "4A3410", WARNTINT = "4A2433";
const HEAD = "PingFang TC", BODY = "PingFang TC", MONO = "Menlo";

const W = 13.33, H = 7.5, MX = 0.7, TITLE_Y = 0.62, FOOT_Y = 7.05, TOTAL = 34;
const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 });

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "GPU 記憶體與資料搬遷讀書會";
pres.title = "GPU × Transformer — 硬體架構聚焦版";

let PAGE = 0; // 自動頁碼
const base = (s) => { s.background = { color: BG }; PAGE += 1; };
function runningHeader(s) {
  s.addText("GPU 記憶體與資料搬遷讀書會 · 合輯", { x: W - 5.2, y: 0.3, w: 4.5, h: 0.3, align: "right", fontFace: BODY, fontSize: 10, color: MUTE, margin: 0 });
}
function footer(s, part) {
  s.addText(part, { x: MX, y: FOOT_Y, w: 8, h: 0.3, fontFace: BODY, fontSize: 9, color: FOOTC, margin: 0 });
  s.addText(`${PAGE} / ${TOTAL}`, { x: W - 1.6, y: FOOT_Y, w: 0.9, h: 0.3, align: "right", fontFace: MONO, fontSize: 9, color: FOOTC, margin: 0 });
}
function header(s, num, title, accent) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: TITLE_Y, w: 0.62, h: 0.62, rectRadius: 0.08, fill: { color: accent }, line: { type: "none" }, shadow: shadow() });
  s.addText(num, { x: MX, y: TITLE_Y, w: 0.62, h: 0.62, align: "center", valign: "middle", fontFace: MONO, fontSize: 20, bold: true, color: BG, margin: 0 });
  s.addText(title, { x: MX + 0.85, y: TITLE_Y, w: W - MX - 0.85 - 0.5, h: 0.62, valign: "middle", fontFace: HEAD, fontSize: 27, bold: true, color: INK, margin: 0 });
}
function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: LINE, width: 1 }, shadow: shadow() });
}
function coreGrid(s, x, y, cols, rows, cell, gap, color) {
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++)
    s.addShape(pres.shapes.RECTANGLE, { x: x + c * (cell + gap), y: y + r * (cell + gap), w: cell, h: cell, fill: { color }, line: { type: "none" } });
}
function parallelTicks(s, x, y, count, tw, th, gap, color, heights) {
  for (let i = 0; i < count; i++) {
    const hh = heights ? heights[i % heights.length] * th : th;
    s.addShape(pres.shapes.RECTANGLE, { x: x + i * (tw + gap), y: y + (th - hh), w: tw, h: hh, fill: { color }, line: { type: "none" } });
  }
}
function chain(s, x, y, count, r, gap, color) {
  for (let i = 0; i < count; i++) {
    const cx = x + i * (r + gap);
    s.addShape(pres.shapes.OVAL, { x: cx, y, w: r, h: r, fill: { color }, line: { type: "none" } });
    if (i < count - 1) s.addShape(pres.shapes.LINE, { x: cx + r, y: y + r / 2, w: gap, h: 0, line: { color, width: 1.5, endArrowType: "triangle" } });
  }
}
function miniRoofline(s, x0, y0, pw, ph) {
  const topY = y0 - ph, xr = x0 + 0.55 * pw;
  s.addShape(pres.shapes.LINE, { x: x0, y: topY, w: 0, h: ph, line: { color: MUTE, width: 1 } });
  s.addShape(pres.shapes.LINE, { x: x0, y: y0, w: pw, h: 0, line: { color: MUTE, width: 1 } });
  s.addShape(pres.shapes.LINE, { x: x0, y: topY, w: xr - x0, h: ph, flipV: true, line: { color: MEM, width: 3 } });
  s.addShape(pres.shapes.LINE, { x: xr, y: topY, w: x0 + pw - xr, h: 0, line: { color: COMP, width: 3 } });
  return xr;
}
function box(s, x, y, w, h, label, fill, txt) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: fill }, line: { color: LINE, width: 1 } });
  s.addText(label, { x, y, w, h, align: "center", valign: "middle", fontFace: HEAD, fontSize: 13, bold: true, color: txt || INK, margin: 0 });
}
function arrow(s, x, y, w, color, label, up) {
  s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width: 2.5, endArrowType: "triangle" } });
  if (label) s.addText(label, { x: x - 0.4, y: up ? y - 0.42 : y + 0.08, w: w + 0.8, h: 0.32, align: "center", fontFace: MONO, fontSize: 10.5, color, margin: 0 });
}
function takeaway(s, text, color) {
  s.addText(text, { x: MX, y: 6.05, w: 11.9, h: 0.55, fontFace: HEAD, fontSize: 15.5, bold: true, color: color || MEM, valign: "middle", margin: 0 });
}

const P1 = "Part 1 · 機器", P2 = "Part 2 · 一把尺", P3 = "Part 3 · 模型上機",
      P4 = "Part 4 · 資料搬遷", P5 = "Part 5 · 共同演化", P0 = "GPU 記憶體與資料搬遷 · 合輯";

// ============================================================ 1 標題
(() => {
  const s = pres.addSlide(); base(s);
  const xr = miniRoofline(s, 9.0, 3.4, 3.4, 1.6);
  s.addShape(pres.shapes.OVAL, { x: 9.5, y: 2.85, w: 0.16, h: 0.16, fill: { color: MEM }, line: { type: "none" } });
  s.addShape(pres.shapes.OVAL, { x: xr + 0.7, y: 1.74, w: 0.16, h: 0.16, fill: { color: COMP }, line: { type: "none" } });
  s.addText("roofline → 共同演化", { x: 8.8, y: 3.6, w: 3.8, h: 0.3, align: "center", fontFace: MONO, fontSize: 10, color: MUTE, margin: 0 });

  s.addText("GPU 記憶體與資料搬遷讀書會  ·  S1–S5 全系列合輯", { x: MX, y: 1.6, w: 9, h: 0.4, fontFace: BODY, fontSize: 15, color: MEM, bold: true, charSpacing: 1, margin: 0 });
  s.addText([
    { text: "速度的故事：", options: { breakLine: true } },
    { text: "平行度 × 資料搬運", options: {} },
  ], { x: MX, y: 2.35, w: 8.3, h: 2.0, fontFace: HEAD, fontSize: 46, bold: true, color: INK, lineSpacingMultiple: 1.06, margin: 0 });
  s.addText("從 roofline 與記憶體階層出發，看懂訓練/推論的瓶頸、資料搬遷的每道關卡，", { x: MX, y: 4.75, w: 11.5, h: 0.45, fontFace: BODY, fontSize: 17, color: MUTE, margin: 0 });
  s.addText("最後拉高視角：模型設計與計算機結構如何互相塑造（CNN → Transformer → 混合架構）。", { x: MX, y: 5.25, w: 11.5, h: 0.45, fontFace: BODY, fontSize: 17, color: MUTE, margin: 0 });
  s.addText("聽眾：data science 背景 · 原 S1–S5 五場內容重編去重", { x: MX, y: 6.05, w: 11.0, h: 0.4, fontFace: BODY, fontSize: 13, color: FOOTC, margin: 0 });
})();

// ============================================================ 2 兩個謎題
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "01", "開場謎題：最貴的算力，為什麼用不到？", COMP);

  card(s, 3.0, 2.1, 7.3, 3.35, BG2);
  s.addText("謎題", { x: 3.35, y: 2.35, w: 6.6, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: COMP, margin: 0 });
  s.addText("< 5%", { x: 3.0, y: 2.7, w: 7.3, h: 1.35, align: "center", fontFace: MONO, fontSize: 78, bold: true, color: COMP, margin: 0 });
  s.addText("同一張 H100，batch=1 的 LLM 解碼，tensor core 利用率常不到 5%——買了全世界最貴的算力，卻幾乎用不到。為什麼？", { x: 3.35, y: 4.25, w: 6.6, h: 1.0, align: "center", fontFace: BODY, fontSize: 15, color: INK, lineSpacingMultiple: 1.3, margin: 0 });

  takeaway(s, "答案不在「算力不夠」，而在「資料搬不夠快 + 平行度不夠」——整份合輯就在拆這件事。");
  footer(s, P0);
})();

// ============================================================ 3 路線圖
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "02", "路線圖：五個篇章", COMP);

  const parts = [
    ["1", "機器", "GPU 是台什麼樣的機器：SM / warp / Amdahl / 記憶體階層", MEM],
    ["2", "一把尺", "算術強度與 roofline：compute-bound 還是 memory-bound？", COMP],
    ["3", "模型上機", "訓練 vs 推論：prefill / decode / KV cache / batch（Transformer 推論）", MEM],
    ["4", "資料搬遷", "把搬運藏在運算後面：prefetch / overlap（壓軸）", COMP],
    ["5", "共同演化", "拉高視角：模型設計 ⇄ 計算機結構（CNN → Transformer → 混合）", GOOD],
  ];
  parts.forEach((p, i) => {
    const y = 1.95 + i * 0.92;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 0.7, h: 0.7, rectRadius: 0.1, fill: { color: p[3] }, line: { type: "none" } });
    s.addText(p[0], { x: MX, y, w: 0.7, h: 0.7, align: "center", valign: "middle", fontFace: MONO, fontSize: 24, bold: true, color: BG, margin: 0 });
    s.addText(p[1], { x: MX + 1.0, y, w: 2.2, h: 0.7, valign: "middle", fontFace: HEAD, fontSize: 19, bold: true, color: INK, margin: 0 });
    s.addText(p[2], { x: MX + 3.3, y, w: 9.0, h: 0.7, valign: "middle", fontFace: BODY, fontSize: 14.5, color: MUTE, margin: 0 });
  });
  s.addText("主線：先認識機器 → 拿到尺 → 量模型 → 修搬運 → 回頭看懂模型為什麼長這樣。", { x: MX, y: 6.55, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 14, color: FOOTC, margin: 0 });
  footer(s, P0);
})();

// ============================================================ Part 1 機器
// 4.5 — 互動環節：開互動地圖（刻意留白的講者指引頁）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);

  s.addText("🔍  互動環節", { x: MX, y: 1.7, w: 11.9, h: 0.9, align: "center", fontFace: HEAD, fontSize: 40, bold: true, color: MEM, margin: 0 });
  s.addText("講者：切出去打開互動式地圖，從 Cluster 一路切到 SM、再到 CUDA / Tensor core", { x: MX, y: 2.75, w: 11.9, h: 0.5, align: "center", fontFace: BODY, fontSize: 18, color: INK, margin: 0 });
  s.addText("interactive/gpu_map.html", { x: 3.9, y: 3.5, w: 5.5, h: 0.6, align: "center", valign: "middle", fontFace: MONO, fontSize: 17, bold: true, color: MEM, fill: { color: "0A1322" }, margin: 8 });

  // 路徑 glyph：Cluster ▸ Node ▸ GPU ▸ SM
  const steps = ["Cluster", "Node", "GPU", "SM", "運算單元"];
  const bw3 = 1.5, gap3 = 0.5, sx = (W - (5 * bw3 + 4 * gap3)) / 2, sy = 4.5;
  steps.forEach((t, i) => {
    const x = sx + i * (bw3 + gap3);
    const last = i === steps.length - 1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: sy, w: bw3, h: 0.6, rectRadius: 0.08, fill: { color: last ? COMPTINT : MEMTINT }, line: { color: last ? COMP : MEM, width: 1.2 } });
    s.addText(t, { x, y: sy, w: bw3, h: 0.6, align: "center", valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: last ? COMP : MEM, margin: 0 });
    if (i < steps.length - 1) s.addShape(pres.shapes.LINE, { x: x + bw3 + 0.08, y: sy + 0.3, w: gap3 - 0.16, h: 0, line: { color: FOOTC, width: 2, endArrowType: "triangle" } });
  });

  s.addText("點發亮的元件往內切　·　Esc 回上層　·　1–5 直接跳層　·　講完按 5 停在運算單元，回來接下一頁", { x: MX, y: 5.5, w: 11.9, h: 0.4, align: "center", fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  footer(s, P1);
})();

// 6 — warp / SIMT 藏延遲（S2-4）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "03", "warp 與 SIMT：用大量 thread 藏延遲", MEM);

  s.addText("32 threads = 1 warp，一起執行同一指令（SIMT）。一個 SM 同時駐留很多 warp。", { x: MX, y: 1.9, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: INK, margin: 0 });
  const lanes = 4, cols = 8, cw2 = 1.18, chh = 0.5, gx = 0.06, gy = 0.2;
  const ox = MX + 1.5, oy = 2.6;
  for (let r = 0; r < lanes; r++) {
    s.addText(`warp ${r}`, { x: MX, y: oy + r * (chh + gy), w: 1.3, h: chh, valign: "middle", fontFace: MONO, fontSize: 12, color: MUTE, margin: 0 });
    for (let c = 0; c < cols; c++) {
      const isCompute = Math.floor(c / 2) === r;
      s.addShape(pres.shapes.RECTANGLE, { x: ox + c * (cw2 + gx), y: oy + r * (chh + gy), w: cw2, h: chh, fill: isCompute ? { color: COMP } : { color: BG2 }, line: { color: isCompute ? COMP : LINE, width: 1 } });
    }
  }
  s.addText("時間 →", { x: ox, y: oy + lanes * (chh + gy) + 0.02, w: 3, h: 0.3, fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });
  s.addText("橘＝在算　深＝等 HBM", { x: ox + 5.0, y: oy + lanes * (chh + gy) + 0.02, w: 4.5, h: 0.3, align: "right", fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });

  s.addText("一個 warp 在等 HBM（數百 cycle）時，排程器切到別的 warp 繼續算 → SM 始終有人在算。", { x: MX, y: 5.55, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: INK, margin: 0 });
  takeaway(s, "藏延遲的前提：有夠多 warp、且頻寬餵得上——「大量平行」不是 GPU 的加分項，是它存在的前提。");
  footer(s, P1);
})();

// 7 — 餵飽一張卡 + Amdahl（S5-3）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "04", "餵飽一張卡要多少平行度？Amdahl 給天花板", MEM);

  card(s, MX, 1.95, 5.9, 3.6, BG2);
  s.addText("H100 SXM 的「寬」", { x: MX + 0.35, y: 2.2, w: 5.2, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: MEM, margin: 0 });
  coreGrid(s, MX + 0.45, 2.72, 12, 4, 0.3, 0.1, MEM);
  s.addText([
    { text: "132 SM × 128 lane ≈ 16,896 條 lane + 528 tensor core", options: { breakLine: true } },
    { text: "每條 lane 還要數十個 warp 待命藏延遲", options: { breakLine: true } },
    { text: "→ 同時要有 10⁵ 量級的 thread 在飛", options: { color: MEM, bold: true } },
  ], { x: MX + 0.35, y: 4.4, w: 5.3, h: 1.05, fontFace: BODY, fontSize: 12.5, color: INK, lineSpacingMultiple: 1.25, margin: 0 });

  const x2 = MX + 5.9 + 0.5;
  card(s, x2, 1.95, 5.5, 3.6, BG2);
  s.addText("Amdahl's Law", { x: x2 + 0.35, y: 2.2, w: 4.8, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: COMP, margin: 0 });
  s.addText("加速上限 = 1 / ((1−p) + p/N)", { x: x2 + 0.35, y: 2.68, w: 4.8, h: 0.45, fontFace: MONO, fontSize: 15, bold: true, color: INK, fill: { color: "0A1322" }, valign: "middle", margin: 8 });
  s.addText([
    { text: "p = 可平行比例，N = lane 數", options: { breakLine: true, color: MUTE } },
    { text: "p = 95%、N → ∞：最多也只有 20×", options: { breakLine: true, color: WARN, bold: true } },
    { text: "N 夠大之後，瓶頸只剩序列的 (1−p)", options: {} },
  ], { x: x2 + 0.35, y: 3.3, w: 4.8, h: 1.55, fontFace: BODY, fontSize: 14, color: INK, lineSpacingMultiple: 1.3, paraSpaceAfter: 7, margin: 0 });

  takeaway(s, "模型裡任何「序列相依」的部分，都是 p 的天花板——這句話 Part 3、Part 5 都會用到。", COMP);
  footer(s, P1);
})();

// 8 — 記憶體階層（S1-8 + S2-5 合併）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "05", "記憶體階層：每遠一站，慢一個數量級", MEM);

  const head = (t) => ({ text: t, options: { fill: { color: BG3 }, color: INK, bold: true, fontSize: 13 } });
  const cell = (t, fill) => ({ text: t, options: fill ? { fill: { color: fill } } : { fill: { color: BG2 } } });
  const rows = [
    [head("層級"), head("約略頻寬"), head("誰管理"), head("備註")],
    [cell("Register"), cell("數十 TB/s"), cell("編譯器"), cell("晶片內最快")],
    [cell("Shared memory / L1", COMPTINT), cell("數十 TB/s", COMPTINT), cell("程式可控", COMPTINT), cell("tiling 的槓桿（Part 2）", COMPTINT)],
    [cell("L2 cache"), cell("數 ~ 數十 TB/s"), cell("硬體"), cell("全 SM 共用")],
    [cell("HBM（GPU 全域）", MEMTINT), cell("2 ~ 4.8 TB/s", MEMTINT), cell("程式配置", MEMTINT), cell("A100~2 / H100~3.35 / H200~4.8", MEMTINT)],
    [cell("NVLink（GPU↔GPU）"), cell("~900 GB/s"), cell("—"), cell("比 PCIe 快一個量級")],
    [cell("PCIe（Host↔Device）", WARNTINT), cell("Gen4 ~32 / Gen5 ~64 GB/s", WARNTINT), cell("—", WARNTINT), cell("最常被跨越的瓶頸", WARNTINT)],
    [cell("CPU DRAM（DDR5）"), cell("~50 ~ 100+ GB/s"), cell("OS"), cell("主機記憶體")],
    [cell("NVMe SSD"), cell("~3 ~ 7 GB/s"), cell("OS"), cell("資料集 / 權重來源")],
  ];
  s.addTable(rows, { x: 0.8, y: 1.85, w: 11.7, colW: [2.9, 3.2, 1.7, 3.9], rowH: 0.42, fontFace: BODY, fontSize: 12.5, color: INK, valign: "middle", align: "left", border: { type: "solid", color: LINE, pt: 1 } });
  s.addText("HBM 是 TB/s、PCIe 是 GB/s——差約 100×；瓶頸＝資料必經的「最慢那段路」（常是 PCIe）。能程式控制的只有 shared memory 與 register。", { x: 0.8, y: 6.3, w: 11.7, h: 0.4, fontFace: BODY, fontSize: 14, color: MEM, margin: 0 });
  footer(s, P1);
})();

// 8 — 三層記憶體的每 GB 價格（新增）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "06", "為什麼越快的記憶體越小？看每 GB 價格（2026 初）", COMP);

  // 左：記憶體金字塔（窄=貴=小 在上）
  const cx = 3.7;
  const tiers = [
    { w: 2.2, label: "片上 SRAM", price: "$5,000/GB 起", sub: "KB ~ MB · 數十 TB/s", fill: "1B3D32", line: GOOD, txt: GOOD },
    { w: 4.2, label: "HBM3E", price: "$13–17/GB", sub: "數十 ~ 141 GB · ~3–5 TB/s", fill: MEMTINT, line: MEM, txt: MEM },
    { w: 6.0, label: "一般 DRAM（DDR5）", price: "$2–3/GB", sub: "TB 級 · ~100 GB/s", fill: BG3, line: MUTE, txt: INK },
  ];
  s.addText("↑ 越貴 · 越快 · 越小", { x: cx - 2.5, y: 1.82, w: 5, h: 0.3, align: "center", fontFace: BODY, fontSize: 11, color: COMP, margin: 0 });
  let yy = 2.18; const th = 1.0, gap = 0.16;
  tiers.forEach((t) => {
    const x = cx - t.w / 2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: yy, w: t.w, h: th, rectRadius: 0.06, fill: { color: t.fill }, line: { color: t.line, width: 1.5 } });
    s.addText(t.label, { x, y: yy + 0.1, w: t.w, h: 0.35, align: "center", fontFace: HEAD, fontSize: 13.5, bold: true, color: t.txt, margin: 0 });
    s.addText(t.price, { x, y: yy + 0.44, w: t.w, h: 0.32, align: "center", fontFace: MONO, fontSize: 13, bold: true, color: t.txt, margin: 0 });
    s.addText(t.sub, { x: x + 0.05, y: yy + 0.74, w: t.w - 0.1, h: 0.26, align: "center", fontFace: BODY, fontSize: 9.5, color: MUTE, margin: 0 });
    yy += th + gap;
  });
  s.addText("↓ 越便宜 · 越慢 · 越大", { x: cx - 2.5, y: 5.5, w: 5, h: 0.3, align: "center", fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });

  // 右：價差 + 用途/payoff
  const rx = 7.5;
  card(s, rx, 2.18, 5.1, 1.45, BG2);
  s.addText("價差有多誇張", { x: rx + 0.3, y: 2.32, w: 4.5, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0 });
  s.addText([
    { text: "SRAM ≈ HBM 的 ", options: {} }, { text: "~300–2000×", options: { color: WARN, bold: true } },
    { text: "\n HBM ≈ DDR5 的 ", options: {} }, { text: "~5–8×", options: { color: MEM, bold: true } },
  ], { x: rx + 0.3, y: 2.7, w: 4.5, h: 0.85, fontFace: BODY, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.25, margin: 0 });

  card(s, rx, 3.8, 5.1, 2.0, BG3);
  s.addText("所以呢", { x: rx + 0.3, y: 3.94, w: 4.5, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: COMP, margin: 0 });
  s.addText([
    { text: "晶片內貴到只放得起 KB~MB → SRAM 註定很小", options: { bullet: true, breakLine: true } },
    { text: "KV cache（GB）塞不進 SRAM → 只能住 HBM、decode 每步串流（Part 3）", options: { bullet: true, breakLine: true, color: MEM } },
    { text: "Groq 全 SRAM → 一顆只有 230MB、貴 → 要很多顆（Part 5）", options: { bullet: true, color: WARN } },
  ], { x: rx + 0.3, y: 4.34, w: 4.6, h: 1.45, fontFace: BODY, fontSize: 12.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 6, margin: 0 });

  takeaway(s, "頻寬階層（上一頁）與價格階層是同一回事：越快越貴越小 → 「容量／頻寬／成本」三角逼出整個記憶體階層。");
  s.addText("價格為 2026 初概略合約價（HBM3E 2025 H1 高峰曾 $17–20）；SRAM 為片上等效估計、非市售模組。", { x: MX, y: 6.62, w: 11.9, h: 0.3, fontFace: BODY, fontSize: 9.5, color: FOOTC, margin: 0 });
  footer(s, P1);
})();

// 8.5 — 方案比較表（橫向硬體比較，移到 Part 1 接在每 GB 價格之後）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "07", "方案速覽與選型心法", MEM);

  const head = (t) => ({ text: t, options: { fill: { color: BG3 }, color: INK, bold: true, fontSize: 12.5 } });
  const cell = (t, f, c) => ({ text: t, options: { fill: { color: f || BG2 }, color: c || INK } });
  const rows = [
    [head("方案"), head("記憶體"), head("容量"), head("頻寬"), head("定位")],
    [cell("RTX 4090"), cell("GDDR6X"), cell("24 GB"), cell("~1 TB/s"), cell("開發 / 本機")],
    [cell("A100 80GB"), cell("HBM2e"), cell("80 GB"), cell("~2 TB/s"), cell("訓練/推論")],
    [cell("H100 SXM"), cell("HBM3"), cell("80 GB"), cell("~3.35 TB/s", MEMTINT, MEM), cell("主力")],
    [cell("H200"), cell("HBM3e"), cell("141 GB"), cell("~4.8 TB/s", MEMTINT, MEM), cell("LLM 推論")],
    [cell("Apple M 系列"), cell("統一 LPDDR"), cell("128–192 GB+", COMPTINT, COMP), cell("~0.4–0.8 TB/s"), cell("本機大模型")],
    [cell("Grace Hopper"), cell("HBM3+LPDDR5X"), cell("96 + ~480 GB", COMPTINT, COMP), cell("HBM ~4 TB/s", MEMTINT, MEM), cell("超大模型")],
  ];
  s.addTable(rows, { x: MX, y: 1.85, w: 11.9, colW: [2.5, 2.6, 2.4, 2.4, 2.0], rowH: 0.44, fontFace: BODY, fontSize: 12, color: INK, valign: "middle", align: "left", border: { type: "solid", color: LINE, pt: 1 } });

  card(s, MX, 5.2, 11.9, 0.8, BG3);
  s.addText([
    { text: "決策：", options: { bold: true, color: INK } },
    { text: "裝得下 HBM → 純 GPU；裝不下但搬得動 → offload/UVM/GDS + overlap；要超大容量 → 統一記憶體。", options: { color: INK } },
  ], { x: MX + 0.4, y: 5.2, w: 11.1, h: 0.8, valign: "middle", fontFace: BODY, fontSize: 13.5, margin: 0 });
  takeaway(s, "memory-bound 推論看「頻寬 + 容量」，不是峰值 FLOPS。沒有最好的記憶體，只有最 match 工作集的。");
  footer(s, P1);
})();


// ============================================================ Part 2 一把尺
// 10 — 兩種慢（S1-3）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "08", "兩種「慢」：你被誰卡住？", MEM);

  const cy = 1.95, ch = 3.6, cw = 5.7;
  card(s, MX, cy, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: cy, w: cw, h: 0.12, fill: { color: COMP }, line: { type: "none" } });
  s.addText("Compute-bound", { x: MX + 0.35, y: cy + 0.3, w: cw - 0.7, h: 0.5, fontFace: HEAD, fontSize: 21, bold: true, color: COMP, margin: 0 });
  s.addText([
    { text: "瓶頸：峰值 FLOPS（算得不夠快）", options: { bullet: true, breakLine: true } },
    { text: "症狀：算力利用率高、加大 batch 也快不了", options: { bullet: true, breakLine: true } },
    { text: "典型：大矩陣乘法、模型訓練", options: { bullet: true } },
  ], { x: MX + 0.35, y: cy + 1.0, w: cw - 0.7, h: 2.2, fontFace: BODY, fontSize: 15, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 8, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, cy, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: cy, w: cw, h: 0.12, fill: { color: MEM }, line: { type: "none" } });
  s.addText("Memory-bound", { x: x2 + 0.35, y: cy + 0.3, w: cw - 0.7, h: 0.5, fontFace: HEAD, fontSize: 21, bold: true, color: MEM, margin: 0 });
  s.addText([
    { text: "瓶頸：記憶體頻寬（資料搬不夠快）", options: { bullet: true, breakLine: true } },
    { text: "症狀：算力利用率低、加大 batch 吞吐會升", options: { bullet: true, breakLine: true } },
    { text: "典型：自迴歸解碼、element-wise 運算", options: { bullet: true } },
  ], { x: x2 + 0.35, y: cy + 1.0, w: cw - 0.7, h: 2.2, fontFace: BODY, fontSize: 15, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 8, margin: 0 });

  takeaway(s, "怎麼判斷在哪一邊？用一把尺：算術強度（下一頁）。", MUTE);
  footer(s, P2);
})();

// 11 — 算術強度（S1-5）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "09", "一把尺：算術強度 (Arithmetic Intensity)", MEM);

  card(s, MX, 1.95, 11.9, 1.7, BG2);
  s.addText([
    { text: "AI  =  ", options: { fontFace: MONO, color: INK } },
    { text: "完成運算的 FLOPs", options: { color: COMP } },
    { text: "  ÷  ", options: { fontFace: MONO, color: MUTE } },
    { text: "需搬動的 Bytes", options: { color: MEM } },
  ], { x: MX, y: 2.15, w: 11.9, h: 0.8, align: "center", fontFace: HEAD, fontSize: 30, bold: true, margin: 0 });
  s.addText("單位：FLOPs / Byte —— 每搬 1 個位元組，能換到幾次浮點運算", { x: MX, y: 2.95, w: 11.9, h: 0.5, align: "center", fontFace: BODY, fontSize: 14, color: MUTE, margin: 0 });

  const cy = 4.0, ch = 1.5, cw = 5.7;
  card(s, MX, cy, cw, ch, BG3);
  s.addText("AI 高 → compute-bound", { x: MX + 0.35, y: cy + 0.25, w: cw - 0.7, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: COMP, margin: 0 });
  s.addText("每搬一點資料就做很多運算。例：大方陣 GEMM，AI ≈ 數百。", { x: MX + 0.35, y: cy + 0.72, w: cw - 0.7, h: 0.7, fontFace: BODY, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.15, margin: 0 });
  const x2 = MX + cw + 0.5;
  card(s, x2, cy, cw, ch, BG3);
  s.addText("AI 低 → memory-bound", { x: x2 + 0.35, y: cy + 0.25, w: cw - 0.7, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: MEM, margin: 0 });
  s.addText("大多時間在等資料。例：矩陣×向量 GEMV，AI ≈ 1–2（fp16）。", { x: x2 + 0.35, y: cy + 0.72, w: cw - 0.7, h: 0.7, fontFace: BODY, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.15, margin: 0 });
  footer(s, P2);
})();

// 12 — Roofline 圖（S1-6）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "10", "Roofline：一眼看出你的天花板", COMP);

  const x0 = 1.5, y0 = 5.7, pw = 6.5, ph = 3.2;
  const topY = y0 - ph, xr = x0 + 0.55 * pw;
  s.addShape(pres.shapes.LINE, { x: x0, y: topY, w: 0, h: ph, line: { color: MUTE, width: 1.5 } });
  s.addShape(pres.shapes.LINE, { x: x0, y: y0, w: pw, h: 0, line: { color: MUTE, width: 1.5 } });
  s.addShape(pres.shapes.LINE, { x: x0, y: topY, w: xr - x0, h: ph, flipV: true, line: { color: MEM, width: 3.5 } });
  s.addShape(pres.shapes.LINE, { x: xr, y: topY, w: x0 + pw - xr, h: 0, line: { color: COMP, width: 3.5 } });
  s.addShape(pres.shapes.LINE, { x: xr, y: topY, w: 0, h: ph, line: { color: FOOTC, width: 1, dashType: "dash" } });
  s.addShape(pres.shapes.OVAL, { x: x0 + 0.9 - 0.09, y: y0 - 1.0 - 0.09, w: 0.18, h: 0.18, fill: { color: MEM }, line: { color: BG, width: 1.5 } });
  s.addShape(pres.shapes.OVAL, { x: xr + 1.3 - 0.09, y: topY + 0.15 - 0.09, w: 0.18, h: 0.18, fill: { color: COMP }, line: { color: BG, width: 1.5 } });
  s.addText("可達成\n算力", { x: 0.55, y: topY - 0.1, w: 0.9, h: 0.6, fontFace: BODY, fontSize: 11, color: MUTE, align: "center", margin: 0 });
  s.addText("算術強度 AI (FLOPs/Byte) →", { x: x0, y: y0 + 0.12, w: pw, h: 0.3, align: "center", fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });
  s.addText("ridge point", { x: xr - 1.0, y: topY - 0.42, w: 2.0, h: 0.3, align: "center", fontFace: MONO, fontSize: 10, color: "9FB2CC", margin: 0 });
  s.addText("頻寬上限", { x: x0 + 0.5, y: y0 - 1.7, w: 1.7, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: MEM, margin: 0 });
  s.addText("算力上限", { x: xr + 0.5, y: topY - 0.05, w: 1.7, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: COMP, margin: 0 });
  s.addText("GEMV", { x: x0 + 0.55, y: y0 - 0.95, w: 1.2, h: 0.25, fontFace: MONO, fontSize: 9.5, color: MEM, margin: 0 });
  s.addText("大 GEMM", { x: xr + 1.45, y: topY + 0.05, w: 1.4, h: 0.25, fontFace: MONO, fontSize: 9.5, color: COMP, margin: 0 });

  const rx = 9.0;
  card(s, rx, 1.95, 3.6, 4.0, BG2);
  s.addText([
    { text: "斜線 = 頻寬上限\n(memory-bound)", options: { bullet: { code: "2022" }, color: MEM, breakLine: true } },
    { text: "平頂 = 算力上限\n(compute-bound)", options: { bullet: { code: "2022" }, color: COMP, breakLine: true } },
    { text: "轉折 = ridge point", options: { bullet: { code: "2022" }, color: INK, breakLine: true } },
    { text: "點落在斜線上 → 受頻寬限，換更強算力也沒用", options: { bullet: { code: "2022" }, color: INK } },
  ], { x: rx + 0.3, y: 2.25, w: 3.0, h: 3.4, fontFace: BODY, fontSize: 14, lineSpacingMultiple: 1.15, paraSpaceAfter: 11, margin: 0, valign: "top" });
  footer(s, P2);
})();

// 13 — ridge point H100（S1-7）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "11", "Ridge point 範例：H100", MEM);

  const cy = 2.1, ch = 2.4, cw = 3.75, gap = 0.32;
  const stats = [
    { big: "990", unit: "TFLOPS", label: "BF16 tensor 峰值算力", color: COMP },
    { big: "3.35", unit: "TB/s", label: "HBM3 記憶體頻寬", color: MEM },
    { big: "≈300", unit: "FLOPs/Byte", label: "ridge point = 算力 ÷ 頻寬", color: INK },
  ];
  stats.forEach((st, i) => {
    const x = MX + i * (cw + gap);
    card(s, x, cy, cw, ch, BG2);
    s.addText(st.big, { x, y: cy + 0.35, w: cw, h: 1.0, align: "center", fontFace: MONO, fontSize: 54, bold: true, color: st.color, margin: 0 });
    s.addText(st.unit, { x, y: cy + 1.4, w: cw, h: 0.4, align: "center", fontFace: MONO, fontSize: 16, color: MUTE, margin: 0 });
    s.addText(st.label, { x: x + 0.2, y: cy + 1.85, w: cw - 0.4, h: 0.45, align: "center", fontFace: BODY, fontSize: 12.5, color: INK, margin: 0 });
  });
  s.addText("算術強度 < 300 的運算，再強的 tensor core 也餵不飽——你只能拿到「頻寬的速度」。", { x: MX, y: 4.95, w: 11.9, h: 0.5, align: "center", fontFace: BODY, fontSize: 17, bold: true, color: MEM, margin: 0 });
  s.addText("（數字為約略值，以官方規格為準）", { x: MX, y: 5.55, w: 11.9, h: 0.35, align: "center", fontFace: BODY, fontSize: 11, color: FOOTC, margin: 0 });
  footer(s, P2);
})();

// 13.5 — 時脈與運算單元：峰值算力哪來的（新增）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "12", "990 TFLOPS 哪來的？時脈 × 單元數 × 每 cycle 運算數", COMP);

  card(s, MX, 1.9, 11.9, 0.95, BG2);
  s.addText([
    { text: "峰值算力  =  ", options: { fontFace: MONO, color: INK } },
    { text: "單元數", options: { color: MEM, bold: true } },
    { text: "  ×  ", options: { fontFace: MONO, color: MUTE } },
    { text: "每 cycle 運算數", options: { color: COMP, bold: true } },
    { text: "  ×  ", options: { fontFace: MONO, color: MUTE } },
    { text: "時脈 (clock)", options: { color: WARN, bold: true } },
  ], { x: MX, y: 1.9, w: 11.9, h: 0.95, align: "center", valign: "middle", fontFace: HEAD, fontSize: 23, margin: 0 });

  // 同一顆 H100、兩條路徑
  const head2 = (t) => ({ text: t, options: { fill: { color: BG3 }, color: INK, bold: true, fontSize: 13 } });
  const cell2 = (t, c, f) => ({ text: t, options: { fill: { color: f || BG2 }, color: c || INK } });
  const rows = [
    [head2("同一顆 H100"), head2("單元數"), head2("每 cycle 運算數"), head2("時脈"), head2("= 峰值")],
    [cell2("CUDA core 路徑（fp32）"), cell2("132 SM × 128"), cell2("× 2（FMA：a×b+c 一次算 2 FLOPs）"), cell2("~2.0 GHz", WARN), cell2("≈ 67 TFLOPS")],
    [cell2("Tensor core 路徑（bf16）"), cell2("132 SM × 4"), cell2("× ~1024（小矩陣乘加一口氣做完）", COMP, COMPTINT), cell2("~1.8 GHz", WARN), cell2("≈ 990 TFLOPS", COMP, COMPTINT)],
  ];
  s.addTable(rows, { x: MX, y: 3.1, w: 11.9, colW: [2.9, 2.0, 4.0, 1.4, 1.6], rowH: 0.62, fontFace: BODY, fontSize: 12.5, color: INK, valign: "middle", align: "left", border: { type: "solid", color: LINE, pt: 1 } });

  card(s, MX, 5.2, 11.9, 0.78, BG3);
  s.addText([
    { text: "頻寬是同一條公式：", options: { bold: true, color: MEM } },
    { text: "HBM3 ≈ 5120-bit 介面 × ~5.2 Gb/s/pin ÷ 8 ≈ 3.35 TB/s ——「算力與頻寬都是：寬 × 時脈」。", options: { color: INK } },
  ], { x: MX + 0.35, y: 5.2, w: 11.2, h: 0.78, valign: "middle", fontFace: BODY, fontSize: 13.5, margin: 0 });

  takeaway(s, "兩條路徑時脈幾乎一樣、峰值差 ~15×——算力來自「每 cycle 運算數」不是 clock；GPU 時脈甚至比 CPU 低（Part 1）。", COMP);
  footer(s, P2);
})();

// 14 — tiling 合併頁（S2-7 + S2-8）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "13", "tiling：把資料留在晶片內重複用 → AI 變高", COMP);

  const cw = 3.7, ch = 2.6;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addText("naive", { x: MX + 0.3, y: 2.15, w: cw - 0.6, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: WARN, margin: 0 });
  s.addText("每個輸出都從 HBM 重讀整行整列 → HBM 讀 ~O(N³)。", { x: MX + 0.3, y: 2.55, w: cw - 0.6, h: 1.0, fontFace: BODY, fontSize: 12.5, color: INK, lineSpacingMultiple: 1.2, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.3, y: 3.8, w: 3.0, h: 0.4, fill: { color: WARN }, line: { type: "none" } });
  s.addText("HBM 讀取：大", { x: MX + 0.4, y: 3.8, w: 2.8, h: 0.4, valign: "middle", fontFace: BODY, fontSize: 12, bold: true, color: BG, margin: 0 });

  const x2 = MX + cw + 0.4;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addText("tiled", { x: x2 + 0.3, y: 2.15, w: cw - 0.6, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: MEM, margin: 0 });
  s.addText("block 載入 shared memory 一次、晶片內重複用 T 次 → HBM 讀 ÷T。", { x: x2 + 0.3, y: 2.55, w: cw - 0.6, h: 1.0, fontFace: BODY, fontSize: 12.5, color: INK, lineSpacingMultiple: 1.2, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: x2 + 0.3, y: 3.8, w: 1.1, h: 0.4, fill: { color: MEM }, line: { type: "none" } });
  s.addText("小", { x: x2 + 0.45, y: 3.8, w: 0.9, h: 0.4, valign: "middle", fontFace: BODY, fontSize: 12, bold: true, color: BG, margin: 0 });

  // 右側：roofline 右推
  const x3 = x2 + cw + 0.5;
  const xr = miniRoofline(s, x3 + 0.3, 4.4, 3.2, 2.2);
  s.addShape(pres.shapes.OVAL, { x: 10.0 - 0.08, y: 3.55 - 0.08, w: 0.16, h: 0.16, fill: { color: WARN }, line: { color: BG, width: 1 } });
  s.addShape(pres.shapes.OVAL, { x: xr + 0.45 - 0.08, y: 2.2 - 0.08, w: 0.16, h: 0.16, fill: { color: COMP }, line: { color: BG, width: 1 } });
  s.addShape(pres.shapes.LINE, { x: 10.15, y: 2.4, w: 1.2, h: 1.05, flipV: true, line: { color: INK, width: 2, dashType: "dash", endArrowType: "triangle" } });
  s.addText("tiling →", { x: 10.35, y: 2.45, w: 1.6, h: 0.3, fontFace: MONO, fontSize: 11, bold: true, color: INK, margin: 0 });
  s.addText("同樣 FLOPs、更少 bytes\n→ 推進 compute-bound", { x: x3, y: 4.65, w: 4.0, h: 0.7, fontFace: BODY, fontSize: 12, color: COMP, lineSpacingMultiple: 1.15, margin: 0 });

  takeaway(s, "這就是 cuBLAS / tensor core kernel 都重度 tiling 的原因——Part 5 的 FlashAttention 是同一招的演算法版。");
  footer(s, P2);
})();

// ============================================================ Part 3 模型上機
// 15 — Training（S3-3）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "14", "Training：吃算力，但更怕「裝不下」", COMP);

  const cw = 5.7, ch = 3.6;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 1.95, w: cw, h: 0.12, fill: { color: COMP }, line: { type: "none" } });
  s.addText("為什麼通常 compute-bound", { x: MX + 0.35, y: 2.25, w: cw - 0.7, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: COMP, margin: 0 });
  s.addText([
    { text: "大 batch → 大 GEMM → AI 高", options: { bullet: true, breakLine: true } },
    { text: "forward / backward 都是大矩陣乘法", options: { bullet: true, breakLine: true } },
    { text: "算力利用率高、餵得飽 tensor core", options: { bullet: true } },
  ], { x: MX + 0.35, y: 2.85, w: cw - 0.7, h: 2.4, fontFace: BODY, fontSize: 15, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 8, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: 1.95, w: cw, h: 0.12, fill: { color: MEM }, line: { type: "none" } });
  s.addText("但記憶體同時要放下", { x: x2 + 0.35, y: 2.25, w: cw - 0.7, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: MEM, margin: 0 });
  const items = [["權重 weights", COMP], ["啟動值 activations（留著等 backward）", MEM], ["梯度 gradients", COMP], ["優化器狀態（Adam ≈ 2× 參數，fp32）", WARN]];
  items.forEach((it, i) => {
    const y = 2.9 + i * 0.58;
    s.addShape(pres.shapes.RECTANGLE, { x: x2 + 0.35, y: y + 0.04, w: 0.22, h: 0.22, fill: { color: it[1] }, line: { type: "none" } });
    s.addText(it[0], { x: x2 + 0.72, y: y - 0.04, w: cw - 1.1, h: 0.4, fontFace: BODY, fontSize: 13.5, color: INK, valign: "middle", margin: 0 });
  });
  takeaway(s, "訓練的痛點常是「裝不下」(容量) 不是「餵不飽」(頻寬)——所以有梯度檢查點、ZeRO、offload。", COMP);
  footer(s, P3);
})();

// 16 — Inference 兩階段（S3-4）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "15", "Inference 兩階段：prefill vs decode", MEM);

  s.addText("時間軸 →", { x: MX, y: 1.95, w: 2, h: 0.3, fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 2.35, w: 3.2, h: 0.7, fill: { color: COMP }, line: { type: "none" } });
  s.addText("prefill（一次吃整段 prompt）", { x: MX, y: 2.35, w: 3.2, h: 0.7, align: "center", valign: "middle", fontFace: BODY, fontSize: 13, bold: true, color: BG, margin: 0 });
  for (let i = 0; i < 9; i++) s.addShape(pres.shapes.RECTANGLE, { x: MX + 3.4 + i * 0.85, y: 2.35, w: 0.62, h: 0.7, fill: { color: MEM }, line: { color: BG, width: 1 } });
  s.addText("decode（一次一 token，逐步生成）", { x: MX + 3.4, y: 3.15, w: 7.6, h: 0.3, fontFace: BODY, fontSize: 13, bold: true, color: MEM, margin: 0 });

  const cw = 5.7, cy = 3.95, ch = 1.9;
  card(s, MX, cy, cw, ch, BG2);
  s.addText("Prefill — compute-bound", { x: MX + 0.35, y: cy + 0.25, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: COMP, margin: 0 });
  s.addText("整段 prompt 一起算 → 大 GEMM、AI 高，像訓練的 forward。", { x: MX + 0.35, y: cy + 0.75, w: cw - 0.7, h: 1.0, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, margin: 0 });
  const x2 = MX + cw + 0.5;
  card(s, x2, cy, cw, ch, BG2);
  s.addText("Decode — memory-bound", { x: x2 + 0.35, y: cy + 0.25, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: MEM, margin: 0 });
  s.addText("一次一 token、batch 小 → GEMV、AI ≈ 1。下一頁解謎題①。", { x: x2 + 0.35, y: cy + 0.75, w: cw - 0.7, h: 1.0, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, margin: 0 });
  footer(s, P3);
})();

// 17 — decode memory-bound（S3-5）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "16", "謎題①解答：decode 每 token 重讀整份權重", MEM);

  card(s, MX, 1.95, 11.9, 1.1, BG3);
  s.addText("每產生 1 個 token，就要把「整份權重」從 HBM 讀一遍。", { x: MX, y: 1.95, w: 11.9, h: 1.1, align: "center", valign: "middle", fontFace: HEAD, fontSize: 21, bold: true, color: INK, margin: 0 });

  const cy = 3.4, ch = 1.9, cw = 3.75, gap = 0.32;
  const stats = [
    { big: "14", unit: "GB", label: "7B 模型 fp16 權重", color: MEM },
    { big: "4.2", unit: "ms / token", label: "14 GB ÷ 3.35 TB/s", color: COMP },
    { big: "~240", unit: "tokens/s", label: "batch=1 的延遲上限", color: INK },
  ];
  stats.forEach((st, i) => {
    const x = MX + i * (cw + gap);
    card(s, x, cy, cw, ch, BG2);
    s.addText(st.big, { x, y: cy + 0.28, w: cw, h: 0.85, align: "center", fontFace: MONO, fontSize: 44, bold: true, color: st.color, margin: 0 });
    s.addText(st.unit, { x, y: cy + 1.12, w: cw, h: 0.35, align: "center", fontFace: MONO, fontSize: 14, color: MUTE, margin: 0 });
    s.addText(st.label, { x: x + 0.15, y: cy + 1.48, w: cw - 0.3, h: 0.35, align: "center", fontFace: BODY, fontSize: 12, color: INK, margin: 0 });
  });
  takeaway(s, "「<5% 算力利用率」的根源：時間都花在搬權重，tensor core 在閒置。換更強算力救不了。");
  footer(s, P3);
})();

// 18 — KV cache（S3-6）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "17", "KV cache：省了重算，換來頻寬與容量壓力", COMP);

  const cw = 5.7, ch = 2.5;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addText("是什麼", { x: MX + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: MEM, margin: 0 });
  s.addText("快取過去 token 的 Key / Value，讓每一步不必對整段序列重算 attention——是讓 decode 還能用的關鍵優化。", { x: MX + 0.35, y: 2.7, w: cw - 0.7, h: 1.6, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.25, margin: 0 });
  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addText("代價", { x: x2 + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: WARN, margin: 0 });
  s.addText([
    { text: "隨序列長度線性增長", options: { bullet: true, breakLine: true } },
    { text: "每步都要讀它 → 頻寬壓力", options: { bullet: true, breakLine: true } },
    { text: "要存住它 → 容量壓力（限制 batch / context）", options: { bullet: true } },
  ], { x: x2 + 0.35, y: 2.7, w: cw - 0.7, h: 1.7, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 6, margin: 0 });

  card(s, MX, 4.65, 11.9, 0.9, BG3);
  s.addText([
    { text: "KV 大小 ≈ ", options: { fontFace: MONO, color: MUTE } },
    { text: "2 × layers × heads × head_dim × seq_len × batch × dtype", options: { fontFace: MONO, color: INK, bold: true } },
  ], { x: MX, y: 4.65, w: 11.9, h: 0.9, align: "center", valign: "middle", fontSize: 15, margin: 0 });
  takeaway(s, "context 越長 decode 每步要讀的越多 → 越 memory-bound。Part 5 的 GQA / Mamba 都是衝著它來的。", COMP);
  footer(s, P3);
})();

// 19 — batch 魔法（S3-7）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "18", "Batch 的魔法：throughput vs latency", MEM);

  card(s, MX, 1.95, 5.5, 3.9, BG2);
  s.addText("為什麼加大 batch 有用", { x: MX + 0.35, y: 2.2, w: 4.8, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: INK, margin: 0 });
  s.addText([
    { text: "batch=1：讀一次權重只服務一個請求 → 浪費", options: { bullet: true, breakLine: true } },
    { text: "batch=N：同一次權重讀取服務 N 個請求", options: { bullet: true, breakLine: true } },
    { text: "→ 吞吐近乎線性升、單步延遲幾乎不變", options: { bullet: true, breakLine: true, color: MEM } },
    { text: "直到變 compute-bound 或 KV cache 裝不下", options: { bullet: true, color: WARN } },
  ], { x: MX + 0.35, y: 2.75, w: 4.8, h: 2.8, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 9, margin: 0 });

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
  takeaway(s, "吞吐升、但「單一請求的延遲」沒變快——batch 是攤平權重讀取，不是加速。");
  footer(s, P3);
})();

// ============================================================ Part 4 資料搬遷
// 26 — prefetch / overlap 壓軸（S4-10）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "19", "壓軸：prefetch / overlap —— 把搬運藏在運算後面", MEM);

  const blk = 0.95, h1 = 0.5;
  s.addText("naive", { x: MX, y: 2.3, w: 1.4, h: h1, valign: "middle", fontFace: MONO, fontSize: 13, color: WARN, margin: 0 });
  const nx = MX + 1.5;
  ["搬0", "算0", "搬1", "算1", "搬2", "算2"].forEach((t, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: nx + i * blk, y: 2.3, w: blk - 0.04, h: h1, fill: { color: i % 2 ? COMP : MEM }, line: { color: BG, width: 1 } });
    s.addText(t, { x: nx + i * blk, y: 2.3, w: blk - 0.04, h: h1, align: "center", valign: "middle", fontFace: MONO, fontSize: 10, color: BG, margin: 0 });
  });
  const nEnd = nx + 6 * blk;
  s.addShape(pres.shapes.LINE, { x: nEnd, y: 2.2, w: 0, h: 2.5, line: { color: WARN, width: 1, dashType: "dash" } });

  s.addText("overlapped", { x: MX, y: 3.4, w: 1.5, h: h1, valign: "middle", fontFace: MONO, fontSize: 13, color: MEM, margin: 0 });
  ["搬0", "搬1", "搬2"].forEach((t, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: nx + i * blk, y: 3.3, w: blk - 0.04, h: h1, fill: { color: MEM }, line: { color: BG, width: 1 } });
    s.addText(t, { x: nx + i * blk, y: 3.3, w: blk - 0.04, h: h1, align: "center", valign: "middle", fontFace: MONO, fontSize: 10, color: BG, margin: 0 });
  });
  ["算0", "算1", "算2"].forEach((t, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: nx + (i + 1) * blk, y: 3.9, w: blk - 0.04, h: h1, fill: { color: COMP }, line: { color: BG, width: 1 } });
    s.addText(t, { x: nx + (i + 1) * blk, y: 3.9, w: blk - 0.04, h: h1, align: "center", valign: "middle", fontFace: MONO, fontSize: 10, color: BG, margin: 0 });
  });
  const oEnd = nx + 4 * blk;
  s.addShape(pres.shapes.LINE, { x: oEnd, y: 3.2, w: 0, h: 1.5, line: { color: MEM, width: 1, dashType: "dash" } });
  s.addText("省下的時間", { x: oEnd + 0.1, y: 4.6, w: nEnd - oEnd, h: 0.3, align: "center", fontFace: BODY, fontSize: 11, bold: true, color: GOOD, margin: 0 });
  s.addShape(pres.shapes.LINE, { x: oEnd, y: 4.55, w: nEnd - oEnd, h: 0, line: { color: GOOD, width: 1.5, endArrowType: "triangle" } });

  s.addText("第二條 stream 預取下一批：compute 一邊算、copy 一邊搬。DataLoader 的 num_workers + pin_memory + prefetch 是同一招。", { x: MX, y: 5.35, w: 11.9, h: 0.6, fontFace: BODY, fontSize: 14.5, color: MEM, lineSpacingMultiple: 1.2, margin: 0 });
  takeaway(s, "搬運省不掉時就把它藏起來，讓昂貴的算力不再空等——資料搬遷篇的收束。", COMP);
  footer(s, P4);
})();

// ============================================================ Part 5 共同演化
// 27 — 平行軸（S5-4）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "20", "平行度不是硬體給的，是模型「暴露」出來的", GOOD);

  const axes = [
    ["batch", "樣本之間天然獨立", MEM],
    ["pixel / 空間", "卷積對每個位置獨立", MEM],
    ["channel / head", "特徵維度切開算", MEM],
    ["token / 時間步", "RNN：序列相依 ✗　attention：獨立 ✓", COMP],
    ["layer (pipeline)", "層間相依，只能流水線", WARN],
  ];
  axes.forEach((a, i) => {
    const y = 1.95 + i * 0.76;
    card(s, MX, y, 3.2, 0.6, BG3);
    s.addText(a[0], { x: MX + 0.25, y, w: 2.9, h: 0.6, valign: "middle", fontFace: MONO, fontSize: 13.5, bold: true, color: a[2], margin: 0 });
    s.addText(a[1], { x: MX + 3.5, y, w: 8.4, h: 0.6, valign: "middle", fontFace: BODY, fontSize: 14, color: INK, margin: 0 });
  });
  takeaway(s, "模型設計 = 決定把計算依賴圖畫成「深的鏈」還是「寬的樹」。RNN 暴露 B×H，attention 暴露 B×T×H。");
  footer(s, P5);
})();

// 28 — CNN / hardware lottery（S5-5）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "21", "硬體 → 模型：CNN 等了 23 年的不是想法", MEM);

  const ty = 2.3;
  s.addShape(pres.shapes.LINE, { x: MX + 0.3, y: ty + 0.5, w: 10.8, h: 0, line: { color: LINE, width: 2 } });
  [[MX + 0.6, "1989", "LeCun ConvNet", "想法已存在，CPU 訓不大", MUTE],
   [MX + 5.0, "2012", "AlexNet ×2 GTX 580", "遊戲卡訓一週，ImageNet 拉開 10pp", COMP],
   [MX + 9.4, "之後", "VGG / ResNet…", "捲積=GEMM，吃滿 GPU", MEM]].forEach((p) => {
    s.addShape(pres.shapes.OVAL, { x: p[0], y: ty + 0.38, w: 0.24, h: 0.24, fill: { color: p[4] }, line: { color: BG, width: 1 } });
    s.addText(p[1], { x: p[0] - 0.8, y: ty - 0.15, w: 1.9, h: 0.35, align: "center", fontFace: MONO, fontSize: 14, bold: true, color: p[4], margin: 0 });
    s.addText(p[2], { x: p[0] - 1.5, y: ty + 0.8, w: 3.3, h: 0.35, align: "center", fontFace: HEAD, fontSize: 13.5, bold: true, color: INK, margin: 0 });
    s.addText(p[3], { x: p[0] - 1.5, y: ty + 1.15, w: 3.3, h: 0.6, align: "center", fontFace: BODY, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.1, margin: 0 });
  });

  card(s, MX, 4.35, 11.9, 1.3, BG2);
  s.addText([
    { text: "為什麼 CNN 跟 GPU 一拍即合：", options: { bold: true, color: MEM } },
    { text: "捲積經 im2col 攤平就是大 GEMM；pixel / channel / batch 全平行；權重重複用 → AI 高 → compute-bound。", options: { color: INK } },
  ], { x: MX + 0.35, y: 4.35, w: 11.2, h: 1.3, valign: "middle", fontFace: BODY, fontSize: 14.5, lineSpacingMultiple: 1.25, margin: 0 });
  takeaway(s, "Hardware lottery（Sara Hooker）：不是最好的想法贏，是最合當代硬體的想法贏。", COMP);
  footer(s, P5);
})();

// 29 — Transformer 為平行而生（S5-6）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "22", "硬體 → 模型：Transformer 為平行而生", COMP);

  const cw = 5.7, ch = 3.2;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addText("RNN 訓練：T 步的鏈", { x: MX + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 16.5, bold: true, color: WARN, margin: 0 });
  for (let i = 0; i < 4; i++) {
    s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.45 + i * 1.25, y: 2.8, w: 0.8, h: 0.6, fill: { color: WARNTINT }, line: { color: WARN, width: 1 } });
    s.addText(`h${i}`, { x: MX + 0.45 + i * 1.25, y: 2.8, w: 0.8, h: 0.6, align: "center", valign: "middle", fontFace: MONO, fontSize: 12, color: WARN, margin: 0 });
    if (i < 3) s.addShape(pres.shapes.LINE, { x: MX + 1.27 + i * 1.25, y: 3.1, w: 0.4, h: 0, line: { color: WARN, width: 2, endArrowType: "triangle" } });
  }
  s.addText("h_t 依賴 h_{t−1} → token 軸只能一步步來；暴露平行度 B×H。", { x: MX + 0.35, y: 3.65, w: cw - 0.7, h: 1.2, fontFace: BODY, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.25, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addText("Attention：整段一次 matmul", { x: x2 + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 16.5, bold: true, color: GOOD, margin: 0 });
  coreGrid(s, x2 + 0.5, 2.75, 8, 2, 0.42, 0.14, GOOD);
  s.addText("整段序列攤成 (B·T)×H 的大 GEMM → token 軸從鏈變寬；暴露 B×T×H。", { x: x2 + 0.35, y: 3.95, w: cw - 0.7, h: 0.95, fontFace: BODY, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.25, margin: 0 });

  s.addText("論文自己說的（2017）：RNN 的序列本質 “precludes parallelization”；Transformer “allowing significantly more parallelization”。", { x: MX, y: 5.35, w: 11.9, h: 0.55, fontFace: BODY, fontSize: 13, italic: true, color: MUTE, lineSpacingMultiple: 1.2, margin: 0 });
  takeaway(s, "Transformer 贏不是因為 FLOPs 少（O(T²) 其實更多），是把計算重排成 GPU 吃得下的形狀。", COMP);
  footer(s, P5);
})();

// 29.5 — 互動環節②：玩具級 Transformer 地圖（講者指引頁，無編號）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);

  s.addText("🔬  互動環節 ②", { x: MX, y: 1.45, w: 11.9, h: 0.9, align: "center", fontFace: HEAD, fontSize: 38, bold: true, color: GOOD, margin: 0 });
  s.addText("講者：切出去打開玩具級 Transformer 地圖（T=5、d=6、2 heads——同構縮小版，含 encoder⟷decoder 全景與單一 matmul 的 L2/HBM 搬運）", { x: MX, y: 2.5, w: 11.9, h: 0.5, align: "center", fontFace: BODY, fontSize: 16, color: INK, margin: 0 });
  s.addText("interactive/transformer_map.html", { x: 3.5, y: 3.2, w: 6.3, h: 0.6, align: "center", valign: "middle", fontFace: MONO, fontSize: 17, bold: true, color: GOOD, fill: { color: "0A1322" }, margin: 8 });

  // 層級路徑（6 層）
  const steps = ["全景 Enc/Dec", "Block", "Attention", "Head", "計算子", "硬體 3 機"];
  const n3 = steps.length, bw3 = 1.74, gap3 = 0.34, sx = (W - (n3 * bw3 + (n3 - 1) * gap3)) / 2, sy = 4.15;
  steps.forEach((t, i) => {
    const last = i === n3 - 1;
    const x = sx + i * (bw3 + gap3);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: sy, w: bw3, h: 0.55, rectRadius: 0.08, fill: { color: last ? COMPTINT : MEMTINT }, line: { color: last ? COMP : MEM, width: 1.2 } });
    s.addText(t, { x, y: sy, w: bw3, h: 0.55, align: "center", valign: "middle", fontFace: MONO, fontSize: 12, bold: true, color: last ? COMP : MEM, margin: 0 });
    if (!last) s.addShape(pres.shapes.LINE, { x: x + bw3 + 0.04, y: sy + 0.275, w: gap3 - 0.08, h: 0, line: { color: FOOTC, width: 2, endArrowType: "triangle" } });
  });

  // 模式列
  s.addText([
    { text: "三種執行模式切著看：", options: { color: MUTE } },
    { text: "訓練", options: { color: COMP, bold: true } }, { text: " / ", options: { color: FOOTC } },
    { text: "Prefill", options: { color: GOOD, bold: true } }, { text: " / ", options: { color: FOOTC } },
    { text: "Decode", options: { color: MEM, bold: true } },
    { text: "　——資料流、KV cache、GEMM→GEMV、利用率全部跟著變", options: { color: MUTE } },
  ], { x: MX, y: 5.05, w: 11.9, h: 0.4, align: "center", fontFace: BODY, fontSize: 14, margin: 0 });
  s.addText("建議動線：全景（encoder→memory M→decoder cross）→ Block → Attention（分叉＝平行軸）→ Head 看矩陣 → 計算子：一個 matmul 的 L2⟷HBM 搬運 → 硬體層 GPU/TPU/Groq；全程配 T/P/D 看 compute-bound→memory-bound、KV cache 串流 → 回來接下一頁", { x: MX, y: 5.55, w: 11.9, h: 0.75, align: "center", fontFace: BODY, fontSize: 11, color: FOOTC, margin: 0 });
  footer(s, P5);
})();

// 30 — 模型 → 硬體（S5-7）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "23", "模型 → 硬體：箭頭反過來", MEM);

  const rows = [
    ["Tensor core（Volta, 2017）", "DL 負載九成是 GEMM → 做專用矩陣乘加單元"],
    ["TPU systolic array", "整顆晶片就是一台矩陣乘法機"],
    ["H100「Transformer Engine」", "硬體功能直接用模型命名；fp8 為 transformer 訓練而做"],
    ["H200：HBM 加到 141 GB", "不是算力不夠，是 KV cache（Part 3）吃容量+頻寬"],
    ["精度 fp32→fp16→bf16→fp8→fp4", "模型端證明訓得動 ⇄ 硬體端做出來，一步步往返"],
  ];
  rows.forEach((r, i) => {
    const y = 1.95 + i * 0.8;
    card(s, MX, y, 4.6, 0.64, MEMTINT);
    s.addText(r[0], { x: MX + 0.25, y, w: 4.2, h: 0.64, valign: "middle", fontFace: HEAD, fontSize: 13, bold: true, color: MEM, margin: 0 });
    s.addText(r[1], { x: MX + 4.95, y, w: 7.5, h: 0.64, valign: "middle", fontFace: BODY, fontSize: 13.5, color: INK, margin: 0 });
  });
  takeaway(s, "這是一個迴圈，不是單向因果：硬體決定哪些模型贏，贏的模型再回頭改造硬體。", COMP);
  footer(s, P5);
})();

// 30.5 — TPU：另一條硬體路線（新增）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "24", "TPU：另一條路 — systolic array（脈動陣列）", COMP);

  const cw = 5.7, ch = 3.6;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 1.95, w: cw, h: 0.12, fill: { color: COMP }, line: { type: "none" } });
  s.addText("設計哲學：GEMM 專用到極致", { x: MX + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: COMP, margin: 0 });
  s.addText([
    { text: "MXU = 128×128 個 MAC 排成陣列", options: { bullet: true, breakLine: true } },
    { text: "weight-stationary：權重釘在格子裡，資料流過去", options: { bullet: true, breakLine: true } },
    { text: "沒有 warp、沒有動態排程——資料跟著 clock 齊步走", options: { bullet: true, breakLine: true } },
    { text: "XLA 編譯器把整個計算圖先排好（靜態）", options: { bullet: true, breakLine: true } },
    { text: "代價：動態形狀 / 稀疏 / 分支不友善", options: { bullet: true, color: MUTE } },
  ], { x: MX + 0.35, y: 2.7, w: cw - 0.7, h: 2.7, fontFace: BODY, fontSize: 13, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 6, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: 1.95, w: cw, h: 0.12, fill: { color: MEM }, line: { type: "none" } });
  s.addText("Transformer 在 TPU 上", { x: x2 + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: MEM, margin: 0 });
  s.addText([
    { text: "訓練 / prefill 的大 GEMM = 甜蜜點，利用率高", options: { bullet: true, breakLine: true } },
    { text: "decode 的小 GEMV 一樣餵不滿陣列 → 一樣 memory-bound", options: { bullet: true, breakLine: true, color: MEM, bold: true } },
    { text: "bf16 就是 TPU 帶進主流的（模型→硬體→模型的迴圈）", options: { bullet: true, breakLine: true } },
    { text: "也配 HBM——頻寬牆跨硬體成立", options: { bullet: true, color: MUTE } },
  ], { x: x2 + 0.35, y: 2.7, w: cw - 0.7, h: 2.7, fontFace: BODY, fontSize: 13, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 7, margin: 0 });

  takeaway(s, "GPU 用「上萬 lane + 動態排程」、TPU 用「固定資料流 + 編譯器」——兩條路都是為 GEMM 而生；瓶頸物理相同。下一頁看第三條路 Groq。");
  footer(s, P5);
})();

// 30.7 — Groq：KV cache 串流 vs 全 SRAM（新增，回答「KV 塞不下 L2」）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "25", "Groq LPU：KV 塞不下晶片 → 乾脆把記憶體全做成 SRAM", WARN);

  const cw = 5.7, ch = 3.95;
  // 左：GPU/TPU 的 decode 痛點 — KV 住 HBM、每步串流
  card(s, MX, 1.9, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 1.9, w: cw, h: 0.12, fill: { color: MEM }, line: { type: "none" } });
  s.addText("GPU / TPU 的 decode 痛點", { x: MX + 0.35, y: 2.15, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: MEM, margin: 0 });
  // HBM → SRAM → TC 流程
  const by = 2.75, bh = 0.66;
  box(s, MX + 0.3, by, 1.55, bh, "HBM\n權重+KV ~GB", MEMTINT, MEM);
  s.addShape(pres.shapes.LINE, { x: MX + 1.9, y: by + bh / 2, w: 0.5, h: 0, line: { color: WARN, width: 2.5, endArrowType: "triangle" } });
  box(s, MX + 2.45, by, 1.5, bh, "SRAM\nL2 50MB", BG3, GOOD);
  s.addShape(pres.shapes.LINE, { x: MX + 4.0, y: by + bh / 2, w: 0.45, h: 0, line: { color: MUTE, width: 2, endArrowType: "triangle" } });
  box(s, MX + 4.5, by, 1.0, bh, "tensor\ncore", COMPTINT, COMP);
  s.addText("每步重串流", { x: MX + 1.85, y: by - 0.32, w: 0.9, h: 0.3, align: "center", fontFace: MONO, fontSize: 9, color: WARN, margin: 0 });
  s.addText([
    { text: "KV cache 可達 GB ≫ L2 50MB ≫ shared 228KB", options: { bullet: true, breakLine: true } },
    { text: "→ 放不進晶片 → 每 token 從 HBM 重新串流", options: { bullet: true, breakLine: true, color: WARN, bold: true } },
    { text: "decode 卡在 3.35 TB/s 的 HBM 頻寬", options: { bullet: true } },
  ], { x: MX + 0.35, y: 3.7, w: cw - 0.7, h: 1.9, fontFace: BODY, fontSize: 12.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 6, margin: 0 });

  // 右：Groq 的答案 — 全 SRAM
  const x2 = MX + cw + 0.5;
  card(s, x2, 1.9, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: 1.9, w: cw, h: 0.12, fill: { color: WARN }, line: { type: "none" } });
  s.addText("Groq 的答案：砍掉 HBM、全用 SRAM", { x: x2 + 0.35, y: 2.15, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: WARN, margin: 0 });
  // 多晶片 glyph
  for (let i = 0; i < 18; i++) {
    s.addShape(pres.shapes.RECTANGLE, { x: x2 + 0.4 + (i % 9) * 0.55, y: 2.75 + Math.floor(i / 9) * 0.55, w: 0.42, h: 0.42, fill: { color: i % 4 === 0 ? "4A2433" : BG3 }, line: { color: WARN, width: 1 } });
  }
  s.addText("~230 MB SRAM / chip × 幾十~幾百顆", { x: x2 + 0.35, y: 3.95, w: cw - 0.7, h: 0.3, fontFace: MONO, fontSize: 11, color: MUTE, margin: 0 });
  s.addText([
    { text: "SRAM ~80 TB/s ≈ HBM 的 20×+ → 沒有那條串流", options: { bullet: true, breakLine: true, color: GOOD, bold: true } },
    { text: "一樣 memory-bound，但「記憶體」快 20×+ → decode token/s 大幅領先", options: { bullet: true, breakLine: true } },
    { text: "代價：一顆只 230MB → 切片散到很多顆 + 確定性網路", options: { bullet: true, color: MUTE } },
  ], { x: x2 + 0.35, y: 4.35, w: cw - 0.7, h: 1.3, fontFace: BODY, fontSize: 12.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 6, margin: 0 });

  takeaway(s, "記憶體牆是物理：GPU 在 HBM 上想辦法（FlashAttention 省 S、GQA 省 KV）、Groq 直接換掉 HBM——三條路都在繞同一道牆。互動地圖硬體層可切 GPU/TPU/Groq。", WARN);
  footer(s, P5);
})();

// 31 — Case CNN（S5-8）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "26", "Case study CNN：FLOPs ≠ 速度", COMP);

  const cw = 5.7, ch = 3.5;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addText("MobileNet：depthwise separable", { x: MX + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: WARN, margin: 0 });
  s.addText([
    { text: "紙上 FLOPs ÷ 8–9（9C² → 9C + C²）", options: { bullet: true, breakLine: true } },
    { text: "但 depthwise 每 byte 只做幾次運算 → AI 個位數、memory-bound", options: { bullet: true, breakLine: true } },
    { text: "channel 軸被拆散 → 餵不滿 tensor core", options: { bullet: true, breakLine: true } },
    { text: "它的目標硬體本來就是手機 CPU", options: { bullet: true, color: MUTE } },
  ], { x: MX + 0.35, y: 2.7, w: cw - 0.7, h: 2.5, fontFace: BODY, fontSize: 13, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 7, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addText("ConvNeXt（2022）：反向操作", { x: x2 + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: GOOD, margin: 0 });
  s.addText([
    { text: "用 transformer 時代配方重設計 ResNet", options: { bullet: true, breakLine: true } },
    { text: "敢用 7×7 大 kernel：FLOPs 多但平行度好、AI 高", options: { bullet: true, breakLine: true } },
    { text: "GPU 上「多而齊」常勝過「少而碎」", options: { bullet: true, color: GOOD, bold: true } },
  ], { x: x2 + 0.35, y: 2.7, w: cw - 0.7, h: 2.1, fontFace: BODY, fontSize: 13, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 7, margin: 0 });

  takeaway(s, "紙上 FLOPs 是 CPU 思維；GPU 上要問 bytes 與平行度——和 decode memory-bound 同一個道理。");
  footer(s, P5);
})();

// 32 — Case Transformer（S5-9）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "27", "Case study Transformer：演算法遷就記憶體階層", MEM);

  const cw = 5.7, ch = 3.6;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 1.95, w: cw, h: 0.12, fill: { color: MEM }, line: { type: "none" } });
  s.addText("FlashAttention（2022）", { x: MX + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 16.5, bold: true, color: MEM, margin: 0 });
  s.addText([
    { text: "數學完全不變（exact attention）", options: { bullet: true, breakLine: true } },
    { text: "tiling + 線上 softmax → T×T 矩陣從不落地 HBM（Part 2 的招）", options: { bullet: true, breakLine: true } },
    { text: "同樣 FLOPs、bytes 大減 → 快 2–4×、記憶體 O(T²)→O(T)", options: { bullet: true, color: MEM, bold: true } },
    { text: "改的不是數學，是資料的「住址」", options: { bullet: true, color: MUTE } },
  ], { x: MX + 0.35, y: 2.7, w: cw - 0.7, h: 2.6, fontFace: BODY, fontSize: 13, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 7, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: 1.95, w: cw, h: 0.12, fill: { color: COMP }, line: { type: "none" } });
  s.addText("MQA / GQA", { x: x2 + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 16.5, bold: true, color: COMP, margin: 0 });
  s.addText([
    { text: "Part 3：decode 每 token 要搬整份 KV cache", options: { bullet: true, breakLine: true } },
    { text: "乾脆改模型：多個 Q head 共用一組 KV head", options: { bullet: true, breakLine: true } },
    { text: "Llama 2 70B：64 Q head 共用 8 組 KV → KV bytes ÷8", options: { bullet: true, color: COMP, bold: true } },
    { text: "為了頻寬連架構都改，接受輕微品質代價", options: { bullet: true, color: MUTE } },
  ], { x: x2 + 0.35, y: 2.7, w: cw - 0.7, h: 2.6, fontFace: BODY, fontSize: 13, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 7, margin: 0 });

  footer(s, P5);
})();

// 33 — Case 混合（S5-10）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "28", "Case study 混合架構：一個模型、兩種硬體人格", GOOD);

  card(s, MX, 1.95, 11.9, 1.8, BG2);
  s.addText("Mamba / SSM（2023）：雙目標共同設計", { x: MX + 0.35, y: 2.15, w: 11.2, h: 0.4, fontFace: HEAD, fontSize: 16.5, bold: true, color: GOOD, margin: 0 });
  s.addText([
    { text: "訓練時：parallel scan 攤平 T 軸 → 平行、compute-bound、吃滿 GPU　", options: { color: INK } },
    { text: "｜　", options: { color: FOOTC } },
    { text: "推論時：退回遞迴、每 token O(1) 狀態 → 沒有越長越肥的 KV cache", options: { color: INK } },
  ], { x: MX + 0.35, y: 2.62, w: 11.2, h: 0.9, fontFace: BODY, fontSize: 14, lineSpacingMultiple: 1.25, margin: 0 });

  card(s, MX, 4.0, 11.9, 1.5, BG3);
  s.addText([
    { text: "但固定狀態 = 有損記憶，精確長程檢索不如 attention → 2024 起主流是混血：", options: { breakLine: true, color: INK } },
    { text: "Jamba / Griffin（attention × SSM 交錯）；Conformer = 卷積（局部）+ attention（全局），也是同類混血思路。", options: { color: MEM } },
  ], { x: MX + 0.35, y: 4.0, w: 11.2, h: 1.5, valign: "middle", fontFace: BODY, fontSize: 14, lineSpacingMultiple: 1.3, margin: 0 });
  takeaway(s, "架構設計越來越像：在硬體約束下解最佳化問題。", COMP);
  footer(s, P5);
})();

// 34 — 三個硬體問題（S5-11）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "29", "設計／選模型前，先問三個硬體問題", COMP);

  const qs = [
    ["①", "平行度在哪個軸？有多寬？", "餵不餵得飽上萬條 lane（B / T / pixel / channel / head）", MEM],
    ["②", "每搬一個 byte 做幾次運算？", "AI 落在 roofline 哪一區（Part 2 的尺）", COMP],
    ["③", "序列相依的鏈有多長？", "Amdahl 的 (1−p)：訓練看 T 軸、推論看 decode 步數", WARN],
  ];
  qs.forEach((q, i) => {
    const y = 1.95 + i * 1.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 0.7, h: 0.7, rectRadius: 0.1, fill: { color: q[3] }, line: { type: "none" } });
    s.addText(q[0], { x: MX, y, w: 0.7, h: 0.7, align: "center", valign: "middle", fontFace: HEAD, fontSize: 22, bold: true, color: BG, margin: 0 });
    s.addText(q[1], { x: MX + 1.0, y: y - 0.02, w: 11.0, h: 0.45, fontFace: HEAD, fontSize: 17.5, bold: true, color: INK, margin: 0 });
    s.addText(q[2], { x: MX + 1.0, y: y + 0.4, w: 11.0, h: 0.35, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  });

  card(s, MX, 5.1, 11.9, 0.95, BG2);
  s.addText([
    { text: "用三問掃模型史：", options: { bold: true, color: COMP } },
    { text: "CNN（✓✓✓）→ RNN（✗鏈長）→ Transformer（訓練✓✓✓；decode ✗鏈長✗AI低）→ Flash/GQA（修 bytes）→ Mamba/混合（修鏈）", options: { color: INK } },
  ], { x: MX + 0.35, y: 5.1, w: 11.2, h: 0.95, valign: "middle", fontFace: BODY, fontSize: 13, lineSpacingMultiple: 1.2, margin: 0 });
  footer(s, P5);
})();

// ============================================================ 收束
// 35 — Demo 總表
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "30", "動手做：五個可重現的 demo", MEM);

  const head = (t) => ({ text: t, options: { fill: { color: BG3 }, color: INK, bold: true, fontSize: 12.5 } });
  const cell = (t, c) => ({ text: t, options: { fill: { color: BG2 }, color: c || INK } });
  const rows = [
    [head("demo"), head("對應篇章"), head("展示概念"), head("關鍵觀察")],
    [cell("01_roofline_mini"), cell("Part 2"), cell("compute vs memory bound"), cell("瘦長矩陣掉進 memory-bound 區")],
    [cell("02_pinned_vs_pageable"), cell("Part 4"), cell("進出站 / DMA"), cell("pinned H2D 快 ~2×")],
    [cell("03_decode_memory_bound"), cell("Part 3"), cell("decode 瓶頸 / batch 攤平"), cell("step 延遲平、tokens/s 線性升")],
    [cell("04_prefetch_overlap"), cell("Part 4"), cell("把搬運藏在運算後面"), cell("overlap 提升 ~1.7×（示意）")],
    [cell("05_flops_vs_parallelism"), cell("Part 5"), cell("FLOPs ≠ 速度"), cell("LSTM 輸給 FLOPs 多 1.75× 的 transformer；depthwise ÷8.7 FLOPs 只 ÷3.7 時間（M2 實測）", MEM)],
  ];
  s.addTable(rows, { x: MX, y: 1.95, w: 11.9, colW: [2.9, 1.3, 3.0, 4.7], rowH: 0.62, fontFace: BODY, fontSize: 12, color: INK, valign: "middle", align: "left", border: { type: "solid", color: LINE, pt: 1 } });
  s.addText("程式在 repo 的 demos/ 下，每個資料夾一個 run.py + README。計時規範：warmup → 同步圍住 → 取中位數。", { x: MX, y: 6.05, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  footer(s, P0);
})();

// 36 — 全系列帶走
(() => {
  const s = pres.addSlide(); base(s);
  s.addText("全系列帶走三句話", { x: MX, y: 0.8, w: 11.9, h: 0.7, fontFace: HEAD, fontSize: 32, bold: true, color: INK, margin: 0 });
  const items = [
    { n: "1", t: "先問 compute-bound 還是 memory-bound——用算術強度那把尺。", d: "很多「慢」是頻寬問題不是算力問題；換更貴的算力救不了 memory-bound。", c: MEM },
    { n: "2", t: "瓶頸＝資料必經的最慢那段路；省不掉就藏起來。", d: "tiling 把資料留在快的地方、pinned+overlap 把搬運藏在運算後面、選型看頻寬+容量。", c: COMP },
    { n: "3", t: "模型與硬體互相塑造——看新架構先問三個硬體問題。", d: "平行軸？每 byte 幾次運算？序列鏈多長？CNN → Transformer → 混合，就是輪流補洞。", c: GOOD },
  ];
  items.forEach((it, i) => {
    const y = 1.9 + i * 1.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 0.7, h: 0.7, rectRadius: 0.1, fill: { color: it.c }, line: { type: "none" } });
    s.addText(it.n, { x: MX, y, w: 0.7, h: 0.7, align: "center", valign: "middle", fontFace: MONO, fontSize: 26, bold: true, color: BG, margin: 0 });
    s.addText(it.t, { x: MX + 1.0, y: y - 0.05, w: 11.0, h: 0.5, fontFace: HEAD, fontSize: 18.5, bold: true, color: INK, margin: 0 });
    s.addText(it.d, { x: MX + 1.0, y: y + 0.42, w: 11.0, h: 0.4, fontFace: BODY, fontSize: 13.5, color: MUTE, margin: 0 });
  });
  card(s, MX, 5.6, 11.9, 1.1, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 5.6, w: 0.12, h: 1.1, fill: { color: MEM }, line: { type: "none" } });
  s.addText([
    { text: "速度的故事，大半是「資料在哪、怎麼搬、誰能平行」的故事。 🎉", options: { color: MEM, bold: true } },
  ], { x: MX + 0.4, y: 5.6, w: 11.3, h: 1.1, valign: "middle", fontFace: BODY, fontSize: 17, margin: 0 });
})();

pres.writeFile({ fileName: "../full_series.pptx" }).then((f) => console.log("written:", f, `(${PAGE} slides)`));
