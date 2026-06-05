// S4 — 資料搬遷的關卡與記憶體方案（系列終章）
// 產生 ../s4_data_movement.pptx。沿用 S1–S3 的深色「矽晶」主題。
const pptxgen = require("pptxgenjs");

const BG = "0E1726", BG2 = "16233A", BG3 = "1C2E4A";
const INK = "EAF1FB", MUTE = "8FA6C4", LINE = "2A3D5C", FOOTC = "5C7299";
const MEM = "38BDF8", COMP = "F59E0B", WARN = "FB7185", GOOD = "34D399";
const MEMTINT = "10455F", COMPTINT = "4A3410", WARNTINT = "4A2433";
const HEAD = "PingFang TC", BODY = "PingFang TC", MONO = "Menlo";

const W = 13.33, H = 7.5, MX = 0.7, TITLE_Y = 0.62, FOOT_Y = 7.05, TOTAL = 12;
const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 });

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "GPU 記憶體與資料搬遷讀書會";
pres.title = "S4 — 資料搬遷的關卡與記憶體方案";

const base = (s) => { s.background = { color: BG }; };
function runningHeader(s) {
  s.addText("GPU 記憶體與資料搬遷讀書會 · S4", { x: W - 5.2, y: 0.3, w: 4.5, h: 0.3, align: "right", fontFace: BODY, fontSize: 10, color: MUTE, margin: 0 });
}
function footer(s, n) {
  s.addText("資料搬遷的關卡與記憶體方案", { x: MX, y: FOOT_Y, w: 8, h: 0.3, fontFace: BODY, fontSize: 9, color: FOOTC, margin: 0 });
  s.addText(`${n} / ${TOTAL}`, { x: W - 1.6, y: FOOT_Y, w: 0.9, h: 0.3, align: "right", fontFace: MONO, fontSize: 9, color: FOOTC, margin: 0 });
}
function header(s, num, title, accent) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: TITLE_Y, w: 0.62, h: 0.62, rectRadius: 0.08, fill: { color: accent }, line: { type: "none" }, shadow: shadow() });
  s.addText(num, { x: MX, y: TITLE_Y, w: 0.62, h: 0.62, align: "center", valign: "middle", fontFace: MONO, fontSize: 22, bold: true, color: BG, margin: 0 });
  s.addText(title, { x: MX + 0.85, y: TITLE_Y, w: W - MX - 0.85 - 0.5, h: 0.62, valign: "middle", fontFace: HEAD, fontSize: 28, bold: true, color: INK, margin: 0 });
}
function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: LINE, width: 1 }, shadow: shadow() });
}
function box(s, x, y, w, h, label, fill, txt) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: fill }, line: { color: LINE, width: 1 } });
  s.addText(label, { x, y, w, h, align: "center", valign: "middle", fontFace: HEAD, fontSize: 13.5, bold: true, color: txt || INK, margin: 0 });
}
function arrow(s, x, y, w, color, label, up) {
  s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width: 2.5, endArrowType: "triangle" } });
  if (label) s.addText(label, { x: x - 0.4, y: up ? y - 0.42 : y + 0.08, w: w + 0.8, h: 0.32, align: "center", fontFace: MONO, fontSize: 10.5, color, margin: 0 });
}

// 1 — 標題
(() => {
  const s = pres.addSlide(); base(s);
  // 關卡 glyph：一排小方塊串接
  const gx = 9.1, gy = 2.7, bw = 0.62, gap = 0.3;
  ["SSD", "DRAM", "HBM", "SM"].forEach((t, i) => {
    const x = gx + i * (bw + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: gy, w: bw, h: bw, rectRadius: 0.05, fill: { color: i < 2 ? BG2 : MEMTINT }, line: { color: i < 2 ? LINE : MEM, width: 1 } });
    if (i < 3) s.addShape(pres.shapes.LINE, { x: x + bw, y: gy + bw / 2, w: gap, h: 0, line: { color: i === 1 ? WARN : MUTE, width: 2, endArrowType: "triangle" } });
  });
  s.addText("SSD → DRAM →[PCIe]→ HBM → SM", { x: 8.6, y: 3.6, w: 4.6, h: 0.3, align: "center", fontFace: MONO, fontSize: 9.5, color: MUTE, margin: 0 });

  s.addText("GPU 記憶體與資料搬遷讀書會  ·  S4 / 共 4 場（終章）", { x: MX, y: 1.7, w: 9, h: 0.4, fontFace: BODY, fontSize: 15, color: MEM, bold: true, charSpacing: 1, margin: 0 });
  s.addText([{ text: "資料搬遷的關卡", options: { breakLine: true } }, { text: "與記憶體方案", options: {} }],
    { x: MX, y: 2.45, w: 8.0, h: 2.0, fontFace: HEAD, fontSize: 44, bold: true, color: INK, lineSpacingMultiple: 1.06, margin: 0 });
  s.addText("把資料送到 GPU 要過幾道關？每道關怎麼選方案。", { x: MX, y: 4.8, w: 8.0, h: 0.5, fontFace: BODY, fontSize: 18, color: MUTE, margin: 0 });
  s.addText("承接 S1 階層 + S2 晶片內 —— 這場講「晶片外」的搬運，並收束整個系列。", { x: MX, y: 5.5, w: 8.2, h: 0.4, fontFace: BODY, fontSize: 13, color: FOOTC, margin: 0 });
})();

// 2 — 全圖
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "01", "全圖：從 SSD 到運算單元", MEM);

  const stations = [
    { t: "NVMe SSD", bw: "~7 GB/s", chip: false },
    { t: "CPU DRAM", bw: "~100 GB/s", chip: false },
    { t: "GPU HBM", bw: "~3.3 TB/s", chip: true },
    { t: "L2 / SM", bw: "數十 TB/s", chip: true },
  ];
  const bw = 2.4, bh = 1.3, gap = 0.75, by = 2.9;
  const startX = (W - (stations.length * bw + (stations.length - 1) * gap)) / 2;
  stations.forEach((st, i) => {
    const x = startX + i * (bw + gap);
    card(s, x, by, bw, bh, st.chip ? MEMTINT : BG2);
    s.addText(st.t, { x, y: by + 0.28, w: bw, h: 0.5, align: "center", fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0 });
    s.addText(st.bw, { x, y: by + 0.78, w: bw, h: 0.35, align: "center", fontFace: MONO, fontSize: 12, color: MEM, margin: 0 });
    if (i < stations.length - 1) {
      const cxL = x + bw, cxR = startX + (i + 1) * (bw + gap), pcie = i === 1;
      s.addShape(pres.shapes.LINE, { x: cxL + 0.05, y: by + bh / 2, w: cxR - cxL - 0.1, h: 0, line: { color: pcie ? WARN : "5C7299", width: pcie ? 4 : 2, endArrowType: "triangle" } });
      if (pcie) s.addText("PCIe", { x: cxL, y: by - 0.42, w: cxR - cxL, h: 0.3, align: "center", fontFace: MONO, fontSize: 11, bold: true, color: WARN, margin: 0 });
    }
  });
  s.addShape(pres.shapes.RECTANGLE, { x: startX - 0.15, y: by + bh + 0.25, w: 2 * bw + gap + 0.3, h: 0.06, fill: { color: COMP }, line: { type: "none" } });
  s.addText("本場聚焦：晶片外", { x: startX - 0.15, y: by + bh + 0.4, w: 2 * bw + gap + 0.3, h: 0.3, align: "center", fontFace: BODY, fontSize: 12, bold: true, color: COMP, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: startX + 2 * (bw + gap) - 0.15, y: by + bh + 0.25, w: 2 * bw + gap + 0.3, h: 0.06, fill: { color: MEM }, line: { type: "none" } });
  s.addText("S2 晶片內", { x: startX + 2 * (bw + gap) - 0.15, y: by + bh + 0.4, w: 2 * bw + gap + 0.3, h: 0.3, align: "center", fontFace: BODY, fontSize: 12, bold: true, color: MEM, margin: 0 });

  s.addText("S1 教你判斷瓶頸、S2 講晶片內；本場把「晶片外」這幾段的搬運方案講完。", { x: MX, y: 6.1, w: 11.9, h: 0.4, align: "center", fontFace: BODY, fontSize: 15, color: MEM, margin: 0 });
  footer(s, 2);
})();

// 3 — H2D / D2H / PCIe
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "02", "進出站：H2D / D2H 與 PCIe 瓶頸", WARN);

  box(s, 2.0, 2.6, 3.0, 1.3, "Host\nCPU DRAM", BG2);
  box(s, 8.3, 2.6, 3.0, 1.3, "Device\nGPU HBM", MEMTINT);
  arrow(s, 5.15, 2.95, 3.0, WARN, "H2D（host→device）", true);
  arrow(s, 5.15, 3.55, 3.0, MUTE, "D2H（device→host）", false);
  s.addText("PCIe", { x: 5.15, y: 4.15, w: 3.0, h: 0.3, align: "center", fontFace: MONO, fontSize: 12, bold: true, color: WARN, margin: 0 });

  card(s, MX, 4.7, 11.9, 1.2, BG2);
  s.addText([
    { text: "PCIe Gen4 ~32 / Gen5 ~64 GB/s", options: { bullet: true, breakLine: true } },
    { text: "比 HBM 慢約 100×；頻繁進出 = 被 PCIe 鎖死。D2H（搬結果回去）也要算。", options: { bullet: true } },
  ], { x: MX + 0.4, y: 4.85, w: 11.1, h: 0.95, fontFace: BODY, fontSize: 15, color: INK, lineSpacingMultiple: 1.15, paraSpaceAfter: 4, margin: 0 });

  s.addText("關鍵：減少跨 PCIe 的次數，或把它「藏」起來（overlap，本場壓軸）。", { x: MX, y: 6.1, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: WARN, margin: 0 });
  footer(s, 3);
})();

// 4 — NVLink
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "03", "NVLink：多卡與 CPU–GPU 的高速橋", MEM);

  s.addText("卡間/晶片間頻寬對照（約略值）", { x: MX, y: 2.0, w: 11.9, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0 });
  // bars
  const bx = 2.8, scale = 8.4 / 900;
  const bars = [["PCIe Gen4", 32, MUTE], ["PCIe Gen5", 64, MUTE], ["NVLink 4", 900, MEM]];
  bars.forEach((b, i) => {
    const y = 2.7 + i * 0.85;
    s.addText(b[0], { x: MX, y, w: 1.9, h: 0.5, valign: "middle", fontFace: MONO, fontSize: 13, color: INK, margin: 0 });
    s.addShape(pres.shapes.RECTANGLE, { x: bx, y: y + 0.05, w: Math.max(0.25, b[1] * scale), h: 0.4, fill: { color: b[2] }, line: { type: "none" } });
    s.addText(`${b[1]} GB/s`, { x: bx + Math.max(0.25, b[1] * scale) + 0.15, y, w: 2, h: 0.5, valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: b[2] === MEM ? MEM : MUTE, margin: 0 });
  });

  card(s, MX, 5.25, 11.9, 0.95, BG2);
  s.addText("NVLink 比 PCIe 快約一個量級 → 多卡並行（tensor / pipeline parallel）搬權重、啟動值、梯度才划算；NVLink-C2C 還用來連 CPU–GPU（Grace Hopper）。", { x: MX + 0.4, y: 5.25, w: 11.1, h: 0.95, valign: "middle", fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.15, margin: 0 });
  footer(s, 4);
})();

// 5 — SSD → GPU
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "04", "SSD → GPU：傳統路徑 vs GPUDirect Storage", COMP);

  s.addText("傳統", { x: MX, y: 2.05, w: 2, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: WARN, margin: 0 });
  box(s, MX, 2.45, 2.6, 0.95, "NVMe SSD", BG2);
  arrow(s, MX + 2.65, 2.45, 1.1, MUTE, null);
  box(s, MX + 3.85, 2.45, 2.6, 0.95, "CPU bounce\nbuffer", WARNTINT, WARN);
  arrow(s, MX + 6.5, 2.45, 1.1, MUTE, null);
  box(s, MX + 7.7, 2.45, 2.6, 0.95, "GPU HBM", MEMTINT);
  s.addText("CPU 介入、兩跳", { x: MX, y: 3.5, w: 6, h: 0.3, fontFace: BODY, fontSize: 12, color: WARN, margin: 0 });

  s.addText("GPUDirect Storage", { x: MX, y: 4.35, w: 4, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: MEM, margin: 0 });
  box(s, MX, 4.75, 2.6, 0.95, "NVMe SSD", BG2);
  arrow(s, MX + 2.65, 4.75, 5.0, MEM, "DMA 直達（繞過 CPU）", false);
  box(s, MX + 7.7, 4.75, 2.6, 0.95, "GPU HBM", MEMTINT);

  s.addText("用途：大資料集載入、大權重載入、KV cache offload。資料越大，CPU 越是瓶頸 → GDS 把 CPU 移出資料路徑。", { x: MX, y: 6.1, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 14.5, color: COMP, margin: 0 });
  footer(s, 5);
})();

// 6 — Unified Memory (一) NVIDIA UVM
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "05", "Unified Memory（一）：NVIDIA UVM（遷移式）", MEM);

  card(s, MX, 1.95, 11.9, 1.5, BG3);
  s.addText("單一指標、CPU/GPU 都能存取；page fault 觸發「頁面遷移」，可超額配置（用超過 HBM 容量），cudaMemPrefetchAsync 可預取藏延遲。", { x: MX + 0.4, y: 1.95, w: 11.1, h: 1.5, valign: "middle", fontFace: BODY, fontSize: 16, color: INK, lineSpacingMultiple: 1.25, margin: 0 });

  const cw = 5.7, cy = 3.75, ch = 1.95;
  card(s, MX, cy, cw, ch, BG2);
  s.addText("好處", { x: MX + 0.35, y: cy + 0.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: GOOD, margin: 0 });
  s.addText([
    { text: "程式簡單（不必手動 copy）", options: { bullet: true, breakLine: true } },
    { text: "能跑超過 HBM 容量的工作集", options: { bullet: true } },
  ], { x: MX + 0.35, y: cy + 0.7, w: cw - 0.7, h: 1.1, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 5, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, cy, cw, ch, BG2);
  s.addText("壞處", { x: x2 + 0.35, y: cy + 0.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: WARN, margin: 0 });
  s.addText([
    { text: "page fault 有開銷", options: { bullet: true, breakLine: true } },
    { text: "存取型態差會 thrashing（來回搬）", options: { bullet: true } },
  ], { x: x2 + 0.35, y: cy + 0.7, w: cw - 0.7, h: 1.1, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 5, margin: 0 });

  s.addText("本質仍是「遷移」——資料還是要搬，只是自動化了。", { x: MX, y: 5.9, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 14.5, color: MEM, margin: 0 });
  footer(s, 6);
})();

// 7 — Unified Memory (二) Apple / Grace Hopper
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "06", "Unified Memory（二）：Apple 零複製、Grace Hopper", COMP);

  const cw = 5.7, ch = 3.0;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addText("Apple 統一記憶體", { x: MX + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: MEM, margin: 0 });
  s.addText([
    { text: "CPU/GPU 共用同一塊實體記憶體", options: { bullet: true, breakLine: true } },
    { text: "零複製（根本不搬）", options: { bullet: true, breakLine: true, color: GOOD } },
    { text: "代價：頻寬較低（LPDDR）→ 跑得動大模型但慢", options: { bullet: true } },
  ], { x: MX + 0.35, y: 2.7, w: cw - 0.7, h: 2.1, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 7, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addText("Grace Hopper（GH200）", { x: x2 + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: COMP, margin: 0 });
  s.addText([
    { text: "Grace(LPDDR5X) + Hopper(HBM3)", options: { bullet: true, breakLine: true } },
    { text: "NVLink-C2C ~900 GB/s 硬體一致性連接", options: { bullet: true, breakLine: true } },
    { text: "→ 又大又快的統一記憶體", options: { bullet: true, color: GOOD } },
  ], { x: x2 + 0.35, y: 2.7, w: cw - 0.7, h: 2.1, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 7, margin: 0 });

  card(s, MX, 5.15, 11.9, 1.05, BG3);
  s.addText([
    { text: "三種「unified」機制不同：", options: { bold: true, color: INK } },
    { text: "遷移（NVIDIA UVM）", options: { color: MEM } },
    { text: " / ", options: { color: MUTE } },
    { text: "零複製（Apple）", options: { color: GOOD } },
    { text: " / ", options: { color: MUTE } },
    { text: "一致性互連（Grace Hopper）", options: { color: COMP } },
  ], { x: MX + 0.4, y: 5.15, w: 11.1, h: 1.05, valign: "middle", fontFace: BODY, fontSize: 15, margin: 0 });
  footer(s, 7);
})();

// 8 — 比較表
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "07", "各方案速覽：容量 / 頻寬 / 定位", MEM);

  const head = (t) => ({ text: t, options: { fill: { color: BG3 }, color: INK, bold: true, fontSize: 13 } });
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
  s.addTable(rows, { x: MX, y: 1.95, w: 11.9, colW: [2.5, 2.6, 2.4, 2.4, 2.0], rowH: 0.5, fontFace: BODY, fontSize: 12.5, color: INK, valign: "middle", align: "left", border: { type: "solid", color: LINE, pt: 1 } });
  s.addText("選型心法：memory-bound 推論看「頻寬 + 容量」，不是峰值 FLOPS。", { x: MX, y: 6.05, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: MEM, margin: 0 });
  footer(s, 8);
})();

// 9 — 心法決策樹
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "08", "心法：把工作集 match 到對的記憶體", COMP);

  const steps = [
    { q: "工作集裝得下 HBM？", a: "→ 純 GPU，最快", c: GOOD },
    { q: "裝不下但搬得動？", a: "→ offload / UVM / GPUDirect（接受搬運成本，盡量 overlap）", c: MEM },
    { q: "要超大容量？", a: "→ 統一記憶體（Apple / Grace Hopper，換取不同頻寬）", c: COMP },
  ];
  steps.forEach((st, i) => {
    const y = 2.1 + i * 1.25;
    card(s, MX, y, 11.9, 1.0, BG2);
    s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 0.12, h: 1.0, fill: { color: st.c }, line: { type: "none" } });
    s.addText(st.q, { x: MX + 0.45, y: y + 0.12, w: 4.2, h: 0.76, valign: "middle", fontFace: HEAD, fontSize: 17, bold: true, color: INK, margin: 0 });
    s.addText(st.a, { x: MX + 4.9, y: y + 0.12, w: 6.7, h: 0.76, valign: "middle", fontFace: BODY, fontSize: 15, color: st.c, margin: 0 });
  });
  s.addText("沒有「最好」的記憶體，只有最 match 你工作集的。", { x: MX, y: 6.1, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, bold: true, color: COMP, margin: 0 });
  footer(s, 9);
})();

// 10 — 壓軸：prefetch / overlap
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "09", "壓軸：prefetch / overlap —— 預先把資料搬到對的地方", MEM);

  const blk = 0.95, h1 = 0.5;
  // naive：copy/compute 交錯，序列
  s.addText("naive", { x: MX, y: 2.4, w: 1.4, h: h1, valign: "middle", fontFace: MONO, fontSize: 13, color: WARN, margin: 0 });
  const nx = MX + 1.5;
  ["搬0", "算0", "搬1", "算1", "搬2", "算2"].forEach((t, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: nx + i * blk, y: 2.4, w: blk - 0.04, h: h1, fill: { color: i % 2 ? COMP : MEM }, line: { color: BG, width: 1 } });
    s.addText(t, { x: nx + i * blk, y: 2.4, w: blk - 0.04, h: h1, align: "center", valign: "middle", fontFace: MONO, fontSize: 10, color: BG, margin: 0 });
  });
  const nEnd = nx + 6 * blk;
  s.addShape(pres.shapes.LINE, { x: nEnd, y: 2.3, w: 0, h: 2.5, line: { color: WARN, width: 1, dashType: "dash" } });

  // overlapped：copy lane + compute lane（錯開一格）
  s.addText("overlapped", { x: MX, y: 3.5, w: 1.5, h: h1, valign: "middle", fontFace: MONO, fontSize: 13, color: MEM, margin: 0 });
  ["搬0", "搬1", "搬2"].forEach((t, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: nx + i * blk, y: 3.4, w: blk - 0.04, h: h1, fill: { color: MEM }, line: { color: BG, width: 1 } });
    s.addText(t, { x: nx + i * blk, y: 3.4, w: blk - 0.04, h: h1, align: "center", valign: "middle", fontFace: MONO, fontSize: 10, color: BG, margin: 0 });
  });
  ["算0", "算1", "算2"].forEach((t, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: nx + (i + 1) * blk, y: 4.0, w: blk - 0.04, h: h1, fill: { color: COMP }, line: { color: BG, width: 1 } });
    s.addText(t, { x: nx + (i + 1) * blk, y: 4.0, w: blk - 0.04, h: h1, align: "center", valign: "middle", fontFace: MONO, fontSize: 10, color: BG, margin: 0 });
  });
  const oEnd = nx + 4 * blk;
  s.addShape(pres.shapes.LINE, { x: oEnd, y: 3.3, w: 0, h: 1.5, line: { color: MEM, width: 1, dashType: "dash" } });
  s.addText("省下的時間", { x: oEnd + 0.1, y: 4.7, w: nEnd - oEnd, h: 0.3, align: "center", fontFace: BODY, fontSize: 11, bold: true, color: GOOD, margin: 0 });
  s.addShape(pres.shapes.LINE, { x: oEnd, y: 4.65, w: nEnd - oEnd, h: 0, line: { color: GOOD, width: 1.5, endArrowType: "triangle" } });

  s.addText("用第二條 stream 預取下一批，compute 一邊算、copy 一邊搬 → 搬運被藏在運算後面。DataLoader 的 num_workers + pin_memory + prefetch 是同一招。", { x: MX, y: 5.5, w: 11.9, h: 0.7, fontFace: BODY, fontSize: 15, color: MEM, lineSpacingMultiple: 1.2, margin: 0 });
  footer(s, 10);
})();

// 11 — Demo 預告
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "10", "動手：prefetch_overlap demo（壓軸）", COMP);

  card(s, MX, 1.95, 5.7, 3.9, BG2);
  s.addText("量什麼", { x: MX + 0.35, y: 2.2, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: MEM, margin: 0 });
  s.addText([
    { text: "naive（搬完才算）vs overlapped（邊搬邊算）", options: { bullet: true, breakLine: true } },
    { text: "看 prefetch 把搬運藏起來省多少", options: { bullet: true } },
  ], { x: MX + 0.35, y: 2.7, w: 5.0, h: 1.4, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 6, margin: 0 });
  s.addText("$ python run.py --num 16 --iters 6", { x: MX + 0.35, y: 4.7, w: 5.0, h: 0.55, fontFace: MONO, fontSize: 12.5, color: MEM, fill: { color: "0A1322" }, valign: "middle", margin: 8 });

  const x2 = MX + 5.7 + 0.5;
  card(s, x2, 1.95, 5.7, 3.9, BG2);
  s.addText("會看到（示意）", { x: x2 + 0.35, y: 2.2, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: INK, margin: 0 });
  const trows = [
    [{ text: "做法", options: { color: MUTE, bold: true } }, { text: "時間", options: { color: MUTE, bold: true, align: "right" } }],
    [{ text: "naive（搬完才算）" }, { text: "100 ms", options: { align: "right" } }],
    [{ text: "overlapped（邊搬邊算）", options: { color: MEM } }, { text: "58 ms", options: { align: "right", color: MEM } }],
    [{ text: "加速", options: { color: GOOD } }, { text: "~1.7x", options: { align: "right", color: GOOD, bold: true } }],
  ];
  s.addTable(trows, { x: x2 + 0.35, y: 2.85, w: 5.0, colW: [3.3, 1.7], rowH: 0.55, fontFace: MONO, fontSize: 13, color: INK, valign: "middle", fill: { color: "0A1322" }, border: { type: "solid", color: LINE, pt: 1 } });
  s.addText("（數字示意；搬運與運算時間越接近，效益越大）", { x: x2 + 0.35, y: 5.2, w: 5.0, h: 0.3, fontFace: BODY, fontSize: 10.5, color: FOOTC, margin: 0 });

  s.addText("這就是整個系列的收束：把搬運「藏」起來，讓昂貴的算力不再空等。", { x: MX, y: 6.05, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: COMP, margin: 0 });
  footer(s, 11);
})();

// 12 — 系列總結
(() => {
  const s = pres.addSlide(); base(s);
  s.addText("系列總結：速度的故事，大半是「搬運」的故事", { x: MX, y: 0.7, w: 11.9, h: 0.7, fontFace: HEAD, fontSize: 28, bold: true, color: INK, margin: 0 });

  const items = [
    { n: "S1", t: "Roofline：先問 compute-bound 還是 memory-bound。", c: MEM },
    { n: "S2", t: "晶片內：HBM / tiling / pinned —— 把資料留在「快的地方」。", c: COMP },
    { n: "S3", t: "訓練 vs 推論：decode memory-bound、KV cache、ASR 架構決定速度。", c: MEM },
    { n: "S4", t: "晶片外：PCIe / NVLink / GDS / Unified Memory，選對方案 + prefetch overlap。", c: COMP },
  ];
  items.forEach((it, i) => {
    const y = 1.75 + i * 0.92;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 0.95, h: 0.7, rectRadius: 0.1, fill: { color: it.c }, line: { type: "none" } });
    s.addText(it.n, { x: MX, y, w: 0.95, h: 0.7, align: "center", valign: "middle", fontFace: MONO, fontSize: 20, bold: true, color: BG, margin: 0 });
    s.addText(it.t, { x: MX + 1.25, y, w: 10.8, h: 0.7, valign: "middle", fontFace: BODY, fontSize: 16.5, color: INK, margin: 0 });
  });

  card(s, MX, 5.55, 11.9, 1.15, BG3);
  s.addText([
    { text: "判斷瓶頸 → 縮短/隱藏搬運 → 選對記憶體。　", options: { bold: true, color: INK } },
    { text: "速度的故事，大半是「資料在哪、怎麼搬」的故事。 🎉", options: { color: MEM, bold: true } },
  ], { x: MX + 0.4, y: 5.55, w: 11.1, h: 1.15, valign: "middle", fontFace: BODY, fontSize: 16, lineSpacingMultiple: 1.15, margin: 0 });
})();

pres.writeFile({ fileName: "../s4_data_movement.pptx" }).then((f) => console.log("written:", f));
