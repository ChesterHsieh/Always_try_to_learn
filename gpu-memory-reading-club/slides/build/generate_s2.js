// S2 — GPU 架構與 HBM：資料在晶片內怎麼走
// 產生 ../s2_gpu_hbm.pptx。沿用 S1/S3 的深色「矽晶」主題。
const pptxgen = require("pptxgenjs");

const BG = "0E1726", BG2 = "16233A", BG3 = "1C2E4A";
const INK = "EAF1FB", MUTE = "8FA6C4", LINE = "2A3D5C", FOOTC = "5C7299";
const MEM = "38BDF8", COMP = "F59E0B", WARN = "FB7185";
const MEMTINT = "10455F", COMPTINT = "4A3410", WARNTINT = "4A2433";
const HEAD = "PingFang TC", BODY = "PingFang TC", MONO = "Menlo";

const W = 13.33, H = 7.5, MX = 0.7, TITLE_Y = 0.62, FOOT_Y = 7.05, TOTAL = 11;
const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 });

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "GPU 記憶體與資料搬遷讀書會";
pres.title = "S2 — GPU 架構與 HBM：資料在晶片內怎麼走";

const base = (s) => { s.background = { color: BG }; };
function runningHeader(s) {
  s.addText("GPU 記憶體與資料搬遷讀書會 · S2", { x: W - 5.2, y: 0.3, w: 4.5, h: 0.3, align: "right", fontFace: BODY, fontSize: 10, color: MUTE, margin: 0 });
}
function footer(s, n) {
  s.addText("GPU 架構與 HBM：資料在晶片內怎麼走", { x: MX, y: FOOT_Y, w: 8, h: 0.3, fontFace: BODY, fontSize: 9, color: FOOTC, margin: 0 });
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
function smGrid(s, x, y, cols, rows, cell, gap, color) {
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++)
    s.addShape(pres.shapes.RECTANGLE, { x: x + c * (cell + gap), y: y + r * (cell + gap), w: cell, h: cell, fill: { color }, line: { type: "none" } });
}
// 同心方框：晶片內階層 glyph（outer→inner）
function nestedBoxes(s, x, y, w, h, rings) {
  const n = rings.length;
  const sx = (w / 2) / n, sy = (h / 2) / n;
  rings.forEach((r, i) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: x + i * sx, y: y + i * sy, w: w - 2 * i * sx, h: h - 2 * i * sy, rectRadius: 0.06,
      fill: { color: r.fill }, line: { color: r.line || LINE, width: 1.2 },
    });
  });
}
function miniRoofline(s, x0, y0, pw, ph) {
  const topY = y0 - ph, xr = x0 + 0.55 * pw;
  s.addShape(pres.shapes.LINE, { x: x0, y: topY, w: 0, h: ph, line: { color: MUTE, width: 1 } });
  s.addShape(pres.shapes.LINE, { x: x0, y: y0, w: pw, h: 0, line: { color: MUTE, width: 1 } });
  s.addShape(pres.shapes.LINE, { x: x0, y: topY, w: xr - x0, h: ph, flipV: true, line: { color: MEM, width: 3 } });
  s.addShape(pres.shapes.LINE, { x: xr, y: topY, w: x0 + pw - xr, h: 0, line: { color: COMP, width: 3 } });
}

// 1 — 標題
(() => {
  const s = pres.addSlide(); base(s);
  nestedBoxes(s, 9.3, 1.9, 3.3, 3.0, [
    { fill: "0C2A3C", line: MEM }, { fill: "10384E", line: MEM }, { fill: "3A2A0C", line: COMP }, { fill: BG3, line: INK },
  ]);
  s.addText("HBM → L2 → shared → register", { x: 8.7, y: 5.0, w: 4.4, h: 0.3, align: "center", fontFace: MONO, fontSize: 10, color: MUTE, margin: 0 });
  s.addText("GPU 記憶體與資料搬遷讀書會  ·  S2 / 共 4 場", { x: MX, y: 1.7, w: 9, h: 0.4, fontFace: BODY, fontSize: 15, color: MEM, bold: true, charSpacing: 1, margin: 0 });
  s.addText([{ text: "GPU 架構與 HBM", options: { breakLine: true } }, { text: "資料在晶片內怎麼走", options: {} }],
    { x: MX, y: 2.45, w: 8.3, h: 2.0, fontFace: HEAD, fontSize: 44, bold: true, color: INK, lineSpacingMultiple: 1.06, margin: 0 });
  s.addText("把放大鏡轉到晶片內：一個 kernel 怎麼從 HBM 把資料一路搬到運算單元。", { x: MX, y: 4.8, w: 8.2, h: 0.6, fontFace: BODY, fontSize: 18, color: MUTE, margin: 0 });
  s.addText("承接 S1 記憶體階層 —— 這場放大「晶片內」那幾層。", { x: MX, y: 5.55, w: 8.2, h: 0.4, fontFace: BODY, fontSize: 13, color: FOOTC, margin: 0 });
})();

// 2 — 回顧：放大晶片內
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "01", "放大鏡：S1 階層的「晶片內」這段", MEM);

  const cw = 5.7, ch = 3.7;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addText("S1 全景（8 層）", { x: MX + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: MUTE, margin: 0 });
  s.addText([
    "register", "shared / L1", "L2", "HBM", "── 以下晶片外 ──", "NVLink", "PCIe", "DRAM / SSD",
  ].map((t, i) => ({ text: t, options: { breakLine: true, color: i < 4 ? INK : FOOTC, fontSize: i === 4 ? 12 : 14 } })),
    { x: MX + 0.35, y: 2.7, w: cw - 0.7, h: 2.8, fontFace: MONO, lineSpacingMultiple: 1.15, margin: 0, valign: "top" });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG3);
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: 1.95, w: cw, h: 0.12, fill: { color: MEM }, line: { type: "none" } });
  s.addText("本場聚焦：晶片內 4 層", { x: x2 + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: MEM, margin: 0 });
  const layers = [["register", "每 thread，最快"], ["shared / L1", "每 SM，程式可控"], ["L2", "全 SM 共用，硬體"], ["HBM（global）", "整卡，2–4.8 TB/s"]];
  layers.forEach((l, i) => {
    const y = 2.75 + i * 0.62;
    s.addText(l[0], { x: x2 + 0.35, y, w: 2.4, h: 0.4, fontFace: MONO, fontSize: 14, bold: true, color: INK, valign: "middle", margin: 0 });
    s.addText(l[1], { x: x2 + 2.7, y, w: cw - 3.0, h: 0.4, fontFace: BODY, fontSize: 13, color: MUTE, valign: "middle", margin: 0 });
  });

  s.addText("資料離運算單元越近越快；今天看 kernel 怎麼把資料往「近的地方」搬。", { x: MX, y: 5.95, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: MEM, margin: 0 });
  footer(s, 2);
})();

// 3 — SM / core / tensor core
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "02", "GPU 解剖：SM、CUDA core、Tensor core", COMP);

  card(s, MX, 1.95, 5.5, 3.9, BG2);
  s.addText("GPU = 一堆 SM", { x: MX + 0.35, y: 2.2, w: 4.8, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: INK, margin: 0 });
  s.addText("（streaming multiprocessor，數十～上百個）", { x: MX + 0.35, y: 2.62, w: 4.8, h: 0.35, fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  smGrid(s, MX + 0.55, 3.2, 8, 4, 0.42, 0.16, MEM);

  const x2 = MX + 5.5 + 0.5;
  card(s, x2, 1.95, 5.7, 3.9, BG2);
  s.addText("一個 SM 裡有什麼", { x: x2 + 0.35, y: 2.2, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: COMP, margin: 0 });
  s.addText([
    { text: "CUDA core：通用浮點/整數運算", options: { bullet: true, breakLine: true } },
    { text: "Tensor core：矩陣乘加（餵 GEMM、衝高 AI）", options: { bullet: true, breakLine: true } },
    { text: "Shared memory + L1：自己的高速暫存", options: { bullet: true, breakLine: true } },
    { text: "Register file + warp 排程器", options: { bullet: true } },
  ], { x: x2 + 0.35, y: 2.75, w: 5.0, h: 2.9, fontFace: BODY, fontSize: 15, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 9, margin: 0 });

  footer(s, 3);
})();

// 4 — warp / SIMT / latency hiding
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "03", "warp 與 SIMT：用大量 thread 藏延遲", MEM);

  s.addText("32 threads = 1 warp，一起執行同一指令（SIMT）。一個 SM 同時駐留很多 warp。", { x: MX, y: 1.9, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15.5, color: INK, margin: 0 });

  // 4 個 warp 的時間軸：每個時段恰有一個 warp 在算 → SM 始終忙
  const lanes = 4, cols = 8, cw = 1.18, chh = 0.52, gx = 0.06, gy = 0.22;
  const ox = MX + 1.5, oy = 2.7;
  for (let r = 0; r < lanes; r++) {
    s.addText(`warp ${r}`, { x: MX, y: oy + r * (chh + gy), w: 1.3, h: chh, valign: "middle", fontFace: MONO, fontSize: 12, color: MUTE, margin: 0 });
    for (let c = 0; c < cols; c++) {
      const isCompute = Math.floor(c / 2) === r; // warp r 在 col 2r..2r+1 運算
      s.addShape(pres.shapes.RECTANGLE, {
        x: ox + c * (cw + gx), y: oy + r * (chh + gy), w: cw, h: chh,
        fill: isCompute ? { color: COMP } : { color: BG2 }, line: { color: isCompute ? COMP : LINE, width: 1 },
      });
    }
  }
  s.addText("時間 →", { x: ox, y: oy + lanes * (chh + gy) + 0.02, w: 3, h: 0.3, fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });
  s.addText("橘＝在算　深＝等 HBM", { x: ox + 5.0, y: oy + lanes * (chh + gy) + 0.02, w: 4.5, h: 0.3, align: "right", fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });

  s.addText("一個 warp 在等 HBM（數百 cycle）時，排程器切到別的 warp 繼續算 → SM 始終有人在算，延遲被藏住。", { x: MX, y: 6.0, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: MEM, margin: 0 });
  footer(s, 4);
})();

// 5 — on-chip hierarchy 表
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "04", "晶片內記憶體階層：誰能控制？", MEM);

  const head = (t) => ({ text: t, options: { fill: { color: BG3 }, color: INK, bold: true, fontSize: 13.5 } });
  const cell = (t, f) => ({ text: t, options: { fill: { color: f || BG2 } } });
  const rows = [
    [head("層級"), head("範圍 scope"), head("約略頻寬"), head("誰管理")],
    [cell("Register"), cell("每 thread"), cell("數十 TB/s"), cell("編譯器")],
    [cell("Shared memory / L1", COMPTINT), cell("每 SM", COMPTINT), cell("數十 TB/s", COMPTINT), cell("程式可控 ← tiling 槓桿", COMPTINT)],
    [cell("L2 cache"), cell("全 SM 共用"), cell("數 ~ 數十 TB/s"), cell("硬體")],
    [cell("HBM（global）", MEMTINT), cell("整卡", MEMTINT), cell("2 ~ 4.8 TB/s", MEMTINT), cell("程式配置、硬體搬", MEMTINT)],
  ];
  s.addTable(rows, { x: 1.0, y: 2.0, w: 11.3, colW: [3.3, 2.4, 2.7, 2.9], rowH: 0.62, fontFace: BODY, fontSize: 13.5, color: INK, valign: "middle", align: "left", border: { type: "solid", color: LINE, pt: 1 } });
  s.addText("能「程式控制」的是 shared memory 與 register —— 這就是 tiling 能發揮的地方（下一頁）。", { x: 1.0, y: 5.9, w: 11.3, h: 0.4, fontFace: BODY, fontSize: 15, color: COMP, margin: 0 });
  footer(s, 5);
})();

// 6 — HBM vs GDDR
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "05", "HBM vs GDDR：為什麼資料中心卡用 HBM", COMP);

  const cw = 5.7, ch = 3.5;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 1.95, w: cw, h: 0.12, fill: { color: MEM }, line: { type: "none" } });
  s.addText("HBM（資料中心）", { x: MX + 0.35, y: 2.25, w: cw - 0.7, h: 0.45, fontFace: HEAD, fontSize: 19, bold: true, color: MEM, margin: 0 });
  s.addText([
    { text: "3D 堆疊 DRAM + 超寬匯流排（數千 bit）", options: { bullet: true, breakLine: true } },
    { text: "頻寬 TB/s 級、單位能耗更省", options: { bullet: true, breakLine: true } },
    { text: "代價：製程複雜、貴", options: { bullet: true } },
  ], { x: MX + 0.35, y: 2.8, w: cw - 0.7, h: 1.8, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 7, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.35, y: 4.85, w: 4.7, h: 0.4, fill: { color: MEM }, line: { type: "none" } });
  s.addText("~3350 GB/s", { x: MX + 0.45, y: 4.85, w: 4.5, h: 0.4, valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: BG, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: 1.95, w: cw, h: 0.12, fill: { color: COMP }, line: { type: "none" } });
  s.addText("GDDR（消費卡）", { x: x2 + 0.35, y: 2.25, w: cw - 0.7, h: 0.45, fontFace: HEAD, fontSize: 19, bold: true, color: COMP, margin: 0 });
  s.addText([
    { text: "傳統 DRAM、較窄匯流排、較高時脈", options: { bullet: true, breakLine: true } },
    { text: "成本低、容量/頻寬夠日常與開發", options: { bullet: true, breakLine: true } },
    { text: "頻寬約 HBM 的 1/3", options: { bullet: true } },
  ], { x: x2 + 0.35, y: 2.8, w: cw - 0.7, h: 1.8, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 7, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: x2 + 0.35, y: 4.85, w: 1.5, h: 0.4, fill: { color: COMP }, line: { type: "none" } });
  s.addText("~1000 GB/s", { x: x2 + 0.45, y: 4.85, w: 2.5, h: 0.4, valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: INK, margin: 0 });

  s.addText("推論/訓練吃頻寬 → 資料中心卡用 HBM 換頻寬與能效（數字為約略值）。", { x: MX, y: 5.95, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: MEM, margin: 0 });
  footer(s, 6);
})();

// 7 — shared memory + tiling
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "06", "Shared memory 與 tiling：把資料留在晶片內重複用", MEM);

  const cw = 5.7, ch = 3.6;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addText("naive", { x: MX + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: WARN, margin: 0 });
  s.addText("每個輸出元素都從 HBM 重讀整行整列 → 大量重複的 HBM 讀取。", { x: MX + 0.35, y: 2.65, w: cw - 0.7, h: 0.9, fontFace: BODY, fontSize: 14, color: INK, lineSpacingMultiple: 1.2, margin: 0 });
  s.addText("HBM 讀取量", { x: MX + 0.35, y: 3.7, w: cw - 0.7, h: 0.3, fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.35, y: 4.05, w: 4.9, h: 0.5, fill: { color: WARN }, line: { type: "none" } });
  s.addText("大", { x: MX + 0.5, y: 4.05, w: 1, h: 0.5, valign: "middle", fontFace: HEAD, fontSize: 16, bold: true, color: BG, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addText("tiled", { x: x2 + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: MEM, margin: 0 });
  s.addText("把一個 block 載入 shared memory 一次，在晶片內重複用 → HBM 讀取大幅減少。", { x: x2 + 0.35, y: 2.65, w: cw - 0.7, h: 0.9, fontFace: BODY, fontSize: 14, color: INK, lineSpacingMultiple: 1.2, margin: 0 });
  s.addText("HBM 讀取量", { x: x2 + 0.35, y: 3.7, w: cw - 0.7, h: 0.3, fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: x2 + 0.35, y: 4.05, w: 1.4, h: 0.5, fill: { color: MEM }, line: { type: "none" } });
  s.addText("小", { x: x2 + 0.5, y: 4.05, w: 1, h: 0.5, valign: "middle", fontFace: HEAD, fontSize: 16, bold: true, color: BG, margin: 0 });

  s.addText("重複用 = 同樣 FLOPs、但更少 bytes → 算術強度 AI 提高（下一頁把它放回 roofline）。", { x: MX, y: 5.95, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: COMP, margin: 0 });
  footer(s, 7);
})();

// 8 — tiling 效果 → roofline
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "07", "tiling 的效果：把 memory-bound 推成 compute-bound", COMP);

  miniRoofline(s, 1.6, 4.9, 4.6, 2.6);
  s.addText("算術強度 AI →", { x: 1.6, y: 5.05, w: 4.6, h: 0.3, align: "center", fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });
  // 由低 AI（memory-bound）往右推到 compute-bound
  s.addShape(pres.shapes.OVAL, { x: 2.5 - 0.09, y: 4.0 - 0.09, w: 0.18, h: 0.18, fill: { color: WARN }, line: { color: BG, width: 1 } });
  s.addShape(pres.shapes.OVAL, { x: 4.9 - 0.09, y: 2.55 - 0.09, w: 0.18, h: 0.18, fill: { color: COMP }, line: { color: BG, width: 1 } });
  // 虛線箭頭：從 rose 點（低 AI）連到 amber 點（高 AI），示意 tiling 把運算往右推
  s.addShape(pres.shapes.LINE, { x: 2.72, y: 2.7, w: 2.0, h: 1.15, flipV: true, line: { color: INK, width: 2, dashType: "dash", endArrowType: "triangle" } });
  s.addText("tiling →", { x: 3.25, y: 2.68, w: 1.9, h: 0.3, fontFace: MONO, fontSize: 12, bold: true, color: INK, margin: 0 });
  s.addText("naive", { x: 1.75, y: 4.05, w: 1.2, h: 0.25, fontFace: MONO, fontSize: 9.5, color: WARN, margin: 0 });
  s.addText("tiled", { x: 5.05, y: 2.42, w: 1.2, h: 0.25, fontFace: MONO, fontSize: 9.5, color: COMP, margin: 0 });

  const x2 = 7.0;
  card(s, x2, 2.1, 5.6, 3.4, BG2);
  s.addText("數字感", { x: x2 + 0.35, y: 2.35, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: INK, margin: 0 });
  s.addText([
    { text: "naive GEMM 從 HBM 讀 ~ O(N³) bytes", options: { bullet: true, breakLine: true } },
    { text: "tile 邊長 T → HBM 流量 ÷ T", options: { bullet: true, breakLine: true, color: MEM } },
    { text: "T 越大省越多，受 shared memory 容量限", options: { bullet: true } },
  ], { x: x2 + 0.35, y: 2.9, w: 5.0, h: 2.0, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 8, margin: 0 });

  s.addText("這就是為什麼 cuBLAS／tensor core kernel 都重度 tiling —— 把寶貴的 HBM 頻寬花在刀口上。", { x: MX, y: 6.05, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: COMP, margin: 0 });
  footer(s, 8);
})();

// 9 — pinned vs pageable
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "08", "進出晶片的橋：pinned vs pageable", MEM);

  const boxW = 2.5, boxH = 0.95;
  const drawBox = (x, y, label, color, txtColor) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: boxW, h: boxH, rectRadius: 0.06, fill: { color }, line: { color: LINE, width: 1 } });
    s.addText(label, { x, y, w: boxW, h: boxH, align: "center", valign: "middle", fontFace: HEAD, fontSize: 14, bold: true, color: txtColor || INK, margin: 0 });
  };
  const arrow = (x, y, w, color, label) => {
    s.addShape(pres.shapes.LINE, { x, y: y + boxH / 2, w, h: 0, line: { color, width: 2.5, endArrowType: "triangle" } });
    if (label) s.addText(label, { x: x - 0.3, y: y - 0.45, w: w + 0.6, h: 0.35, align: "center", fontFace: MONO, fontSize: 10.5, color, margin: 0 });
  };

  // pageable（多一跳）
  s.addText("pageable", { x: MX, y: 2.0, w: 2.0, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: WARN, margin: 0 });
  let y = 2.45;
  drawBox(MX, y, "CPU DRAM", BG2);
  arrow(MX + boxW + 0.05, y, 1.2, WARN, "複製");
  drawBox(MX + boxW + 1.3, y, "pinned\nbounce buffer", WARNTINT, WARN);
  arrow(MX + 2 * boxW + 1.35, y, 1.2, MUTE, "PCIe");
  drawBox(MX + 2 * boxW + 2.6, y, "GPU HBM", BG2);
  s.addText("多一跳、不能 async", { x: MX, y: y + boxH + 0.12, w: 6, h: 0.3, fontFace: BODY, fontSize: 12, color: WARN, margin: 0 });

  // pinned（直達）
  s.addText("pinned", { x: MX, y: 4.55, w: 2.0, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: MEM, margin: 0 });
  y = 5.0;
  drawBox(MX, y, "pinned DRAM", MEMTINT, MEM);
  arrow(MX + boxW + 0.05, y, 2.45, MEM, "PCIe（DMA 直達）");
  drawBox(MX + 2 * boxW + 2.6, y, "GPU HBM", BG2);
  s.addText("DMA 直達、可 async 與運算重疊", { x: MX, y: y + boxH + 0.12, w: 7, h: 0.3, fontFace: BODY, fontSize: 12, color: MEM, margin: 0 });

  footer(s, 9);
})();

// 10 — Demo 預告
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "09", "動手：pinned_vs_pageable demo", COMP);

  card(s, MX, 1.95, 5.7, 3.9, BG2);
  s.addText("量什麼", { x: MX + 0.35, y: 2.2, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: MEM, margin: 0 });
  s.addText([
    { text: "不同大小的 H2D 傳輸頻寬", options: { bullet: true, breakLine: true } },
    { text: "pageable vs pinned 對照", options: { bullet: true } },
  ], { x: MX + 0.35, y: 2.7, w: 5.0, h: 1.2, fontFace: BODY, fontSize: 15, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 6, margin: 0 });
  s.addText("$ python run.py --sizes-mb 1,4,16,64,256", { x: MX + 0.35, y: 4.6, w: 5.0, h: 0.55, fontFace: MONO, fontSize: 12.5, color: MEM, fill: { color: "0A1322" }, valign: "middle", margin: 8 });

  const x2 = MX + 5.7 + 0.5;
  card(s, x2, 1.95, 5.7, 3.9, BG2);
  s.addText("會看到（示意）", { x: x2 + 0.35, y: 2.2, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: INK, margin: 0 });
  const trows = [
    [{ text: "size", options: { color: MUTE, bold: true } }, { text: "pageable", options: { color: MUTE, bold: true, align: "right" } }, { text: "pinned", options: { color: MUTE, bold: true, align: "right" } }, { text: "x", options: { color: MUTE, bold: true, align: "right" } }],
    [{ text: "16MB" }, { text: "11", options: { align: "right" } }, { text: "21", options: { align: "right", color: MEM } }, { text: "1.9", options: { align: "right" } }],
    [{ text: "64MB" }, { text: "12", options: { align: "right" } }, { text: "24", options: { align: "right", color: MEM } }, { text: "2.0", options: { align: "right" } }],
    [{ text: "256MB" }, { text: "12", options: { align: "right" } }, { text: "25", options: { align: "right", color: MEM } }, { text: "2.1", options: { align: "right" } }],
  ];
  s.addTable(trows, { x: x2 + 0.35, y: 2.75, w: 5.0, colW: [1.4, 1.4, 1.2, 1.0], rowH: 0.5, fontFace: MONO, fontSize: 12.5, color: INK, valign: "middle", fill: { color: "0A1322" }, border: { type: "solid", color: LINE, pt: 1 } });
  s.addText("GB/s（數字示意；實際依 PCIe 代數而定）", { x: x2 + 0.35, y: 5.0, w: 5.0, h: 0.3, fontFace: BODY, fontSize: 10.5, color: FOOTC, margin: 0 });

  s.addText("同一條 PCIe，記憶體 pin 不 pin 差約 2×。", { x: MX, y: 6.05, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: MEM, margin: 0 });
  footer(s, 10);
})();

// 11 — 收束 + S3
(() => {
  const s = pres.addSlide(); base(s);
  s.addText("本場帶走三件事", { x: MX, y: 0.8, w: 11.9, h: 0.7, fontFace: HEAD, fontSize: 32, bold: true, color: INK, margin: 0 });
  const items = [
    { n: "1", t: "SM 用「大量 warp」藏記憶體延遲。", d: "一個 warp 等 HBM，就切到別的 warp——前提是頻寬餵得上。", c: MEM },
    { n: "2", t: "晶片內階層：HBM → L2 → shared → register。", d: "能程式控制的是 shared memory 與 register。", c: COMP },
    { n: "3", t: "tiling 與 pinned：把資料留在/搬到「快的地方」。", d: "tiling 提高 AI 往 roofline 右推；pinned 讓 H2D 快又能 async。", c: MEM },
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
  s.addText([{ text: "下一場 S3　", options: { color: MEM, bold: true } }, { text: "Training vs Inference 的瓶頸差異（以 ASR 為例）", options: { color: INK } }],
    { x: MX + 0.4, y: 5.6, w: 11.3, h: 1.1, valign: "middle", fontFace: BODY, fontSize: 16, margin: 0 });
})();

pres.writeFile({ fileName: "../s2_gpu_hbm.pptx" }).then((f) => console.log("written:", f));
