// 第二堂課 — Transformer × GPU 框架：逐 block 上機（單卡）→ 單卡放不下時的資料平行
// 產生 ../class2_transformer_gpu.pptx。沿用第一堂（full_series）的深色「矽晶」主題。
//
// 兩大部分：
//   Part A · 單卡逐 block × GPU：玩具 transformer（T=6, d=6, 2 heads）每個 block
//            對應到哪個 NVIDIA GPU 單元（tensor core / CUDA core / HBM / L2 / shared / warp / SM）
//            → 整個模型一張卡跑得完。
//   Part B · 單卡放不下的問題：權重 + KV cache + 訓練狀態為何會爆掉一顆 HBM，
//            資料平行(DP)是最直覺的第一招，以及 DP 切資料、不切模型，救不了「裝不下」的模型。
//
// 搭配互動教具：interactive/gpu_map.html（第 4 頁）、interactive/transformer_map.html（第 11 頁）
const pptxgen = require("pptxgenjs");

const BG = "0E1726", BG2 = "16233A", BG3 = "1C2E4A";
const INK = "EAF1FB", MUTE = "8FA6C4", LINE = "2A3D5C", FOOTC = "5C7299";
const MEM = "38BDF8", COMP = "F59E0B", WARN = "FB7185", GOOD = "34D399", PURP = "A78BFA";
const MEMTINT = "10455F", COMPTINT = "4A3410", WARNTINT = "4A2433", GOODTINT = "123D31", PURPTINT = "2A2150";
const HEAD = "PingFang TC", BODY = "PingFang TC", MONO = "Menlo";

const W = 13.33, H = 7.5, MX = 0.7, TITLE_Y = 0.62, FOOT_Y = 7.05, TOTAL = 24;
const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 });

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "GPU 記憶體與資料搬遷讀書會";
pres.title = "第二堂課 · Transformer × GPU 框架";

let PAGE = 0;
const base = (s) => { s.background = { color: BG }; PAGE += 1; };
function runningHeader(s) {
  s.addText("讀書會 · 第二堂課 · Transformer × GPU", { x: W - 5.6, y: 0.3, w: 4.9, h: 0.3, align: "right", fontFace: BODY, fontSize: 10, color: MUTE, margin: 0 });
}
function footer(s, part) {
  s.addText(part, { x: MX, y: FOOT_Y, w: 9, h: 0.3, fontFace: BODY, fontSize: 9, color: FOOTC, margin: 0 });
  s.addText(`${PAGE} / ${TOTAL}`, { x: W - 1.6, y: FOOT_Y, w: 0.9, h: 0.3, align: "right", fontFace: MONO, fontSize: 9, color: FOOTC, margin: 0 });
}
function header(s, num, title, accent) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: TITLE_Y, w: 0.62, h: 0.62, rectRadius: 0.08, fill: { color: accent }, line: { type: "none" }, shadow: shadow() });
  s.addText(num, { x: MX, y: TITLE_Y, w: 0.62, h: 0.62, align: "center", valign: "middle", fontFace: MONO, fontSize: 20, bold: true, color: BG, margin: 0 });
  s.addText(title, { x: MX + 0.85, y: TITLE_Y, w: W - MX - 0.85 - 0.5, h: 0.62, valign: "middle", fontFace: HEAD, fontSize: 25, bold: true, color: INK, margin: 0 });
}
function card(s, x, y, w, h, fill, lineColor) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: lineColor || LINE, width: 1 }, shadow: shadow() });
}
function box(s, x, y, w, h, label, fill, txtColor, fs) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: fill }, line: { color: LINE, width: 1 } });
  s.addText(label, { x, y, w, h, align: "center", valign: "middle", fontFace: HEAD, fontSize: fs || 13, bold: true, color: txtColor || INK, margin: 0 });
}
function obox(s, x, y, w, h, label, edge, txtColor, fs) { // 外框 box（透明填底、彩色邊）
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: BG2 }, line: { color: edge, width: 1.5 } });
  s.addText(label, { x, y, w, h, align: "center", valign: "middle", fontFace: HEAD, fontSize: fs || 12.5, bold: true, color: txtColor || edge, margin: 0 });
}
function arrow(s, x, y, w, color, label, up) {
  s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width: 2.5, endArrowType: "triangle" } });
  if (label) s.addText(label, { x: x - 0.5, y: up ? y - 0.4 : y + 0.07, w: w + 1.0, h: 0.3, align: "center", fontFace: MONO, fontSize: 10, color, margin: 0 });
}
function varrow(s, x, y, h, color, down) { // 垂直箭頭
  s.addShape(pres.shapes.LINE, { x, y, w: 0, h, line: { color, width: 2.5, endArrowType: "triangle", ...(down ? {} : { flipV: true }) } });
}
function takeaway(s, text, color) {
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 6.02, w: 0.09, h: 0.62, fill: { color: color || MEM }, line: { type: "none" } });
  s.addText(text, { x: MX + 0.22, y: 6.0, w: 11.7, h: 0.66, fontFace: HEAD, fontSize: 15, bold: true, color: color || MEM, valign: "middle", margin: 0 });
}
function pill(s, x, y, w, h, text, edge, fill, txt, fs) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: h / 2, fill: { color: fill || BG2 }, line: { color: edge, width: 1 } });
  s.addText(text, { x, y, w, h, align: "center", valign: "middle", fontFace: BODY, fontSize: fs || 11, bold: true, color: txt || edge, margin: 0 });
}
function gpuGlyph(s, x, y, w, h, label, edge, sub) { // 一張 GPU（含 4×2 core 點陣）
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: BG2 }, line: { color: edge, width: 1.5 }, shadow: shadow() });
  s.addText(label, { x, y: y + 0.06, w, h: 0.3, align: "center", fontFace: MONO, fontSize: 11, bold: true, color: edge, margin: 0 });
  const cols = 4, rows = 2, cell = 0.14, gap = 0.07;
  const gw = cols * cell + (cols - 1) * gap, gh = rows * cell + (rows - 1) * gap;
  const gx = x + (w - gw) / 2, gy = y + h - gh - 0.18;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++)
    s.addShape(pres.shapes.RECTANGLE, { x: gx + c * (cell + gap), y: gy + r * (cell + gap), w: cell, h: cell, fill: { color: edge }, line: { type: "none" } });
  if (sub) s.addText(sub, { x, y: y + h - 0.16, w, h: 0.16, align: "center", fontFace: BODY, fontSize: 7.5, color: MUTE, margin: 0 });
}
function matGlyph(s, x, y, cols, rows, cell, color) { // 小矩陣格點
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++)
    s.addShape(pres.shapes.RECTANGLE, { x: x + c * (cell + 0.02), y: y + r * (cell + 0.02), w: cell, h: cell, fill: { color }, line: { type: "none" } });
}

// 逐 block 導覽列
const BLOCKS = ["Embed+PE", "QKV 投影", "多頭切分", "Attention", "Add & Norm", "FFN"];
function blockStepper(s, active) {
  const y = 1.44, x0 = MX, w = 1.85, gap = 0.12, h = 0.48;
  BLOCKS.forEach((b, i) => {
    const on = i === active;
    const x = x0 + i * (w + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: on ? COMP : BG2 }, line: { color: on ? COMP : LINE, width: 1 } });
    s.addText(`${i}·${b}`, { x, y, w, h, align: "center", valign: "middle", fontFace: BODY, fontSize: 11.5, bold: on, color: on ? BG : MUTE, margin: 0 });
    if (i < BLOCKS.length - 1) s.addText("▸", { x: x + w - 0.02, y, w: gap + 0.04, h, align: "center", valign: "middle", fontFace: BODY, fontSize: 10, color: FOOTC, margin: 0 });
  });
}
// 「→ 對應 GPU」右側卡
function gpuMapCard(s, rows, bound, boundColor) {
  const x = 8.55, y = 2.35, w = 4.05, h = 3.35;
  card(s, x, y, w, h, BG2, boundColor);
  s.addText("→ 對應 NVIDIA GPU", { x: x + 0.25, y: y + 0.18, w: w - 0.5, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0 });
  const ry0 = y + 0.72, rh = (h - 1.55) / rows.length;
  rows.forEach(([k, v], i) => {
    const ry = ry0 + i * rh;
    s.addText(k, { x: x + 0.25, y: ry, w: 1.55, h: rh, valign: "middle", fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
    s.addText(v, { x: x + 1.8, y: ry, w: w - 2.05, h: rh, valign: "middle", fontFace: BODY, fontSize: 12, bold: true, color: INK, margin: 0 });
  });
  pill(s, x + 0.25, y + h - 0.62, w - 0.5, 0.42, bound, boundColor, boundColor === COMP ? COMPTINT : MEMTINT, boundColor, 13);
}

const PA = "Part A · 單卡：Transformer 逐 block × GPU", PB = "Part B · 多卡：資料平行與 NVIDIA 互連", P0 = "讀書會 · 第二堂課";

// ============================================================ 1 標題
(() => {
  const s = pres.addSlide(); base(s);
  // 右側：一張卡 → 多張卡 縮圖
  gpuGlyph(s, 9.55, 2.55, 1.2, 1.2, "1 GPU", MEM, "單卡跑得完");
  s.addText("Part A", { x: 9.55, y: 3.82, w: 1.2, h: 0.25, align: "center", fontFace: MONO, fontSize: 10, color: MEM, margin: 0 });
  [0, 1, 2, 3].forEach((i) => gpuGlyph(s, 11.05 + (i % 2) * 0.78, 2.55 + Math.floor(i / 2) * 0.72, 0.68, 0.62, "", COMP));
  s.addShape(pres.shapes.LINE, { x: 10.8, y: 3.15, w: 0.22, h: 0, line: { color: MUTE, width: 2, endArrowType: "triangle" } });
  s.addText("Part B", { x: 11.05, y: 3.82, w: 1.5, h: 0.25, align: "center", fontFace: MONO, fontSize: 10, color: COMP, margin: 0 });

  s.addText("GPU 記憶體與資料搬遷讀書會 · 第二堂課", { x: MX, y: 1.55, w: 9, h: 0.4, fontFace: BODY, fontSize: 15, color: MEM, bold: true, charSpacing: 1, margin: 0 });
  s.addText([
    { text: "Transformer 上機：", options: { breakLine: true } },
    { text: "從一張卡的每個 block，到多卡的通訊", options: {} },
  ], { x: MX, y: 2.35, w: 8.5, h: 2.2, fontFace: HEAD, fontSize: 38, bold: true, color: INK, lineSpacingMultiple: 1.08, margin: 0 });
  s.addText("Part A：把玩具 Transformer 逐 block 攤開，看每一步跑在 GPU 的哪個單元、被算力還是頻寬卡住。", { x: MX, y: 4.85, w: 8.3, h: 0.5, fontFace: BODY, fontSize: 15, color: MUTE, margin: 0 });
  s.addText("Part B：模型大到一張卡裝不下——資料平行有哪些問題，NVIDIA 用哪些跨 GPU 通訊新技術補上。", { x: MX, y: 5.35, w: 8.3, h: 0.5, fontFace: BODY, fontSize: 15, color: MUTE, margin: 0 });
  s.addText("延續第一堂：roofline · 記憶體階層 · KV cache · decode 的 memory-bound", { x: MX, y: 6.15, w: 11, h: 0.4, fontFace: BODY, fontSize: 12.5, color: FOOTC, margin: 0 });
})();

// ============================================================ 2 路線圖
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "00", "這堂課的兩條線", MEM);
  // Part A card
  card(s, MX, 1.9, 5.85, 3.9, BG2, MEM);
  pill(s, MX + 0.3, 2.15, 1.4, 0.42, "Part A", MEM, MEMTINT, MEM, 13);
  s.addText("單卡逐 block × GPU", { x: MX + 1.85, y: 2.15, w: 3.8, h: 0.42, valign: "middle", fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0 });
  [
    "玩具 Transformer：T=6、d=6、2 heads（同構真實模型）",
    "逐 block 拆：Embed → QKV → 多頭 → Attention → Add&Norm → FFN",
    "每個 block 落在哪個單元：tensor core / CUDA core / HBM / L2 / shared",
    "誰是 compute-bound、誰是 memory-bound",
    "結論：整個模型一張卡（一顆 HBM）跑得完",
  ].forEach((t, i) => {
    s.addShape(pres.shapes.OVAL, { x: MX + 0.35, y: 2.85 + i * 0.57 + 0.06, w: 0.1, h: 0.1, fill: { color: MEM }, line: { type: "none" } });
    s.addText(t, { x: MX + 0.62, y: 2.85 + i * 0.57, w: 5.0, h: 0.5, valign: "middle", fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  });
  // Part B card
  const bx = MX + 6.15;
  card(s, bx, 1.9, 5.85, 3.9, BG2, COMP);
  pill(s, bx + 0.3, 2.15, 1.4, 0.42, "Part B", COMP, COMPTINT, COMP, 13);
  s.addText("多卡的資料平行與互連", { x: bx + 1.85, y: 2.15, w: 3.8, h: 0.42, valign: "middle", fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0 });
  [
    "一張卡裝不下：權重 + KV cache + optimizer 狀態",
    "資料平行 (DP) 的四個問題（本堂重點）",
    "切模型本身：張量 TP / 管線 PP / 專家 EP",
    "通訊變瓶頸 → NVLink5 / NVSwitch / SHARP / NCCL",
    "跨節點 InfiniBand / Spectrum-X；2026：Rubin、CMX",
  ].forEach((t, i) => {
    s.addShape(pres.shapes.OVAL, { x: bx + 0.35, y: 2.85 + i * 0.57 + 0.06, w: 0.1, h: 0.1, fill: { color: COMP }, line: { type: "none" } });
    s.addText(t, { x: bx + 0.62, y: 2.85 + i * 0.57, w: 5.0, h: 0.5, valign: "middle", fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  });
  takeaway(s, "一條主線：先看懂一張卡怎麼跑一個 Transformer，再看模型變大後、卡與卡之間要搬什麼。");
  footer(s, P0);
})();

// ============================================================ 3 回顧玩具 transformer
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "01", "回顧：我們的玩具 Transformer", MEM);
  s.addText("真實模型 d=12288、96 heads、128K context——大到畫不出來。把維度縮到極小、但保留每一個分叉，結構與尺寸無關。", { x: MX, y: 1.42, w: 11.9, h: 0.5, fontFace: BODY, fontSize: 13.5, color: MUTE, margin: 0 });
  // 規格表
  const specs = [["序列長 T", "6", "128K+"], ["d_model", "6", "12288"], ["heads", "2", "96+"], ["d_head", "3", "128"], ["FFN hidden", "12（2×）", "4×d"]];
  card(s, MX, 2.2, 4.5, 3.5, BG2);
  s.addText("玩具規格", { x: MX + 0.3, y: 2.38, w: 4, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: MEM, margin: 0 });
  s.addText([{ text: "維度", options: { bold: true } }, { text: "　　玩具", options: { bold: true } }, { text: "　　真實(GPT級)", options: { bold: true } }], { x: MX + 0.3, y: 2.8, w: 3.9, h: 0.3, fontFace: MONO, fontSize: 10.5, color: FOOTC, margin: 0 });
  specs.forEach(([k, v, r], i) => {
    const y = 3.2 + i * 0.48;
    s.addText(k, { x: MX + 0.3, y, w: 1.7, h: 0.42, valign: "middle", fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
    s.addText(v, { x: MX + 1.95, y, w: 1.2, h: 0.42, valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: MEM, margin: 0 });
    s.addText(r, { x: MX + 3.05, y, w: 1.3, h: 0.42, valign: "middle", fontFace: MONO, fontSize: 11, color: FOOTC, margin: 0 });
  });
  // 全景流程（右）
  const bx = 5.7;
  card(s, bx, 2.2, 6.9, 3.5, BG2);
  s.addText("一層 Decoder block 的資料流（GPT 這類只留 decoder）", { x: bx + 0.3, y: 2.38, w: 6.4, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0 });
  const flow = [["Embed + 位置", MEM], ["Masked Self-Attn", MEM], ["Add & Norm", MUTE], ["FFN 6→12→6", COMP], ["Add & Norm", MUTE], ["→ logits", GOOD]];
  flow.forEach(([t, c], i) => {
    const y = 2.9 + i * 0.44;
    obox(s, bx + 0.35, y, 4.0, 0.36, t, c, c, 11.5);
    if (i < flow.length - 1) s.addText("↓", { x: bx + 0.35, y: y + 0.34, w: 4.0, h: 0.12, align: "center", fontFace: BODY, fontSize: 9, color: FOOTC, margin: 0 });
  });
  s.addText("我 愛 喝 咖 啡 →", { x: bx + 4.6, y: 3.0, w: 2.1, h: 0.4, valign: "middle", fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0 });
  s.addText("attention 是 token 間唯一交換資訊的地方；FFN 對每個 token 獨立（token 軸天生平行）。", { x: bx + 4.55, y: 3.5, w: 2.2, h: 1.6, valign: "top", fontFace: BODY, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.35, margin: 0 });
  takeaway(s, "同構：玩具上看懂的每一條資料流，乘上倍率就是真實模型。這堂課逐 block 問「它在 GPU 上跑在哪」。");
  footer(s, PA);
})();

// ============================================================ 4 那台 GPU（單元總覽 + 互動①）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "02", "我們要對應的那台 GPU", COMP);
  s.addText("在逐 block 之前，先把 GPU 的零件擺出來——每個 block 會用到其中幾個。", { x: MX, y: 1.42, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 13.5, color: MUTE, margin: 0 });
  // 兩欄：運算單元 / 記憶體階層
  card(s, MX, 2.0, 5.85, 3.6, BG2, COMP);
  s.addText("運算單元（誰在算）", { x: MX + 0.3, y: 2.18, w: 5.2, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: COMP, margin: 0 });
  [
    ["Tensor Core", "整塊 tile 的矩陣乘加（MMA）→ 吃 GEMM", COMP],
    ["CUDA Core", "純量逐格算 → softmax / norm / 逐元素", MEM],
    ["SM / warp", "32 執行緒一組；大量 warp 藏記憶體延遲", INK],
  ].forEach(([k, v, c], i) => {
    const y = 2.68 + i * 0.92;
    s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.3, y: y + 0.05, w: 0.09, h: 0.62, fill: { color: c }, line: { type: "none" } });
    s.addText(k, { x: MX + 0.5, y, w: 5.1, h: 0.34, fontFace: MONO, fontSize: 13, bold: true, color: c, margin: 0 });
    s.addText(v, { x: MX + 0.5, y: 0.33 + y, w: 5.1, h: 0.38, fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
  });
  card(s, MX + 6.15, 2.0, 5.85, 3.6, BG2, MEM);
  s.addText("記憶體階層（資料住哪）", { x: MX + 6.45, y: 2.18, w: 5.2, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: MEM, margin: 0 });
  [
    ["HBM（global）", "~3.35 TB/s · 大而慢 · 權重 + KV cache 住這"],
    ["L2 cache", "~數十 MB · 全晶片共用"],
    ["Shared / L1", "~228 KB/SM · 程式可控（tiling 關鍵）"],
    ["Register", "最快 · tensor core 直接吃"],
  ].forEach(([k, v], i) => {
    const y = 2.66 + i * 0.72;
    s.addText(k, { x: MX + 6.45, y, w: 2.3, h: 0.6, valign: "middle", fontFace: MONO, fontSize: 12, bold: true, color: MEM, margin: 0 });
    s.addText(v, { x: MX + 8.75, y, w: 3.1, h: 0.6, valign: "middle", fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });
    if (i < 3) s.addShape(pres.shapes.LINE, { x: MX + 6.45, y: y + 0.66, w: 5.2, h: 0, line: { color: LINE, width: 0.75 } });
  });
  // 互動指引
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 5.85, w: 11.9, h: 0.62, rectRadius: 0.1, fill: { color: GOODTINT }, line: { color: GOOD, width: 1 } });
  s.addText([{ text: "🔍 互動環節①　", options: { bold: true, color: GOOD } }, { text: "切出去開 interactive/gpu_map.html：Cluster → Node → GPU → SM → 運算單元(CUDA/Tensor) 下鑽，回來接逐 block。", options: { color: INK } }], { x: MX + 0.3, y: 5.85, w: 11.3, h: 0.62, valign: "middle", fontFace: BODY, fontSize: 12.5, margin: 0 });
  footer(s, PA);
})();

// ============================================================ 5 Block 0 Embedding
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "03", "Block 0 — Embedding + 位置編碼", MEM);
  blockStepper(s, 0);
  // 左圖：token id → 查表 → 向量
  card(s, MX, 2.35, 7.6, 3.35, BG2);
  s.addText("token id → 查表（gather）→ 向量 [6]", { x: MX + 0.3, y: 2.55, w: 7, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0 });
  ["我", "愛", "喝", "咖", "啡", "…"].forEach((t, i) => {
    box(s, MX + 0.4 + i * 0.62, 3.15, 0.52, 0.52, t, BG3, INK, 13);
  });
  s.addText("查表", { x: MX + 0.4, y: 3.8, w: 3.5, h: 0.3, fontFace: MONO, fontSize: 10.5, color: FOOTC, margin: 0 });
  varrow(s, MX + 2.0, 3.75, 0.42, MEM, true);
  // embedding 表（大矩陣住 HBM）
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 4.4, y: 3.1, w: 3.0, h: 1.05, rectRadius: 0.06, fill: { color: MEMTINT }, line: { color: MEM, width: 1.5 } });
  s.addText("Embedding 表 [vocab × 6]", { x: MX + 4.4, y: 3.28, w: 3.0, h: 0.3, align: "center", fontFace: MONO, fontSize: 10.5, bold: true, color: MEM, margin: 0 });
  s.addText("住 HBM · 只讀取用到的幾列", { x: MX + 4.4, y: 3.62, w: 3.0, h: 0.4, align: "center", fontFace: BODY, fontSize: 10.5, color: MUTE, margin: 0 });
  s.addText("+ 位置編碼：把「第幾個字」的資訊加進向量 → 得到 X [6×6]（逐元素相加）", { x: MX + 0.3, y: 4.55, w: 7.0, h: 0.9, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  gpuMapCard(s, [
    ["主要運算", "查表 gather + 逐元素加"],
    ["用到單元", "CUDA core（不是 tensor core）"],
    ["資料位置", "embedding 表在 HBM"],
    ["搬運型態", "隨機讀取幾列 → 記憶體存取"],
  ], "memory-bound（運算極少）", MEM);
  takeaway(s, "第一步幾乎不算數：只是把字換成向量。GPU 這裡是在「搬」，不是在「算」。");
  footer(s, PA);
})();

// ============================================================ 6 Block 1 QKV 投影
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "04", "Block 1 — QKV 投影：第一個大矩陣乘", COMP);
  blockStepper(s, 1);
  card(s, MX, 2.35, 7.6, 3.35, BG2);
  s.addText("X [6×6] 各乘三個權重 → Q、K、V [6×6]", { x: MX + 0.3, y: 2.55, w: 7, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0 });
  // X · Wq/Wk/Wv
  matGlyph(s, MX + 0.55, 3.25, 6, 6, 0.16, MEM);
  s.addText("X", { x: MX + 0.55, y: 4.35, w: 1.15, h: 0.3, align: "center", fontFace: MONO, fontSize: 12, bold: true, color: MEM, margin: 0 });
  s.addText("×", { x: MX + 1.75, y: 3.55, w: 0.4, h: 0.4, align: "center", fontFace: MONO, fontSize: 20, color: FOOTC, margin: 0 });
  ["Wq", "Wk", "Wv"].forEach((wn, i) => {
    matGlyph(s, MX + 2.25 + i * 1.25, 3.25, 6, 6, 0.16, COMP);
    s.addText(wn, { x: MX + 2.25 + i * 1.25, y: 4.35, w: 1.15, h: 0.3, align: "center", fontFace: MONO, fontSize: 12, bold: true, color: COMP, margin: 0 });
  });
  s.addText("→ Q,K,V", { x: MX + 6.0, y: 3.55, w: 1.4, h: 0.4, valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: GOOD, margin: 0 });
  s.addText("這是標準 GEMM（矩陣×矩陣）。真實模型 d=12288 → 這幾個乘法是每層算力大戶。", { x: MX + 0.3, y: 4.75, w: 7.0, h: 0.8, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  gpuMapCard(s, [
    ["主要運算", "GEMM：X · W（矩陣乘）"],
    ["用到單元", "Tensor Core（MMA 一次一塊 tile）"],
    ["資料流動", "HBM → L2 → shared → register"],
    ["加速關鍵", "tiling：tile 留 shared 重複用"],
  ], "compute-bound（T 夠大時）", COMP);
  takeaway(s, "權重從 HBM 讀進來、切成 16×16 的 tile 餵進 tensor core——這就是 GPU「算得快」的主場。");
  footer(s, PA);
})();

// ============================================================ 7 Block 2 多頭切分
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "05", "Block 2 — 多頭切分：分叉就是平行軸", PURP);
  blockStepper(s, 2);
  card(s, MX, 2.35, 7.6, 3.35, BG2);
  s.addText("6 維「切」成 2 個 head × 3 維——兩個 head 結構相同、彼此零依賴", { x: MX + 0.3, y: 2.55, w: 7, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0 });
  // Q[6x6] 切成兩塊
  matGlyph(s, MX + 0.5, 3.3, 6, 6, 0.17, MEM);
  s.addText("Q,K,V [6×6]", { x: MX + 0.5, y: 4.45, w: 1.6, h: 0.3, align: "center", fontFace: MONO, fontSize: 11, color: MUTE, margin: 0 });
  s.addShape(pres.shapes.LINE, { x: MX + 2.35, y: 3.85, w: 0.5, h: 0, line: { color: PURP, width: 2.5, endArrowType: "triangle" } });
  // head1 / head2
  matGlyph(s, MX + 3.05, 3.3, 3, 6, 0.17, MEM);
  s.addText("head 1", { x: MX + 3.05, y: 4.45, w: 0.9, h: 0.3, align: "center", fontFace: MONO, fontSize: 11, bold: true, color: MEM, margin: 0 });
  matGlyph(s, MX + 4.35, 3.3, 3, 6, 0.17, COMP);
  s.addText("head 2", { x: MX + 4.35, y: 4.45, w: 0.9, h: 0.3, align: "center", fontFace: MONO, fontSize: 11, bold: true, color: COMP, margin: 0 });
  s.addText("→ 各自算 attention", { x: MX + 5.4, y: 3.7, w: 2.0, h: 0.6, valign: "middle", fontFace: BODY, fontSize: 11.5, color: GOOD, margin: 0 });
  s.addText("多送 GPU 一個平行軸（head 軸）：head 之間可同時算，對到不同 tile / 不同 warp（batched GEMM）。", { x: MX + 0.3, y: 4.85, w: 7.0, h: 0.8, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  gpuMapCard(s, [
    ["切出來的", "head 軸（玩具 2、真實 96+）"],
    ["平行對應", "batched GEMM / 不同 warp、tile"],
    ["為何重要", "GPU 靠「寬」——平行軸越多越好餵"],
    ["延伸", "GQA：多個 Q head 共用一組 KV"],
  ], "餵飽平行度 → 高利用率", COMP);
  takeaway(s, "Transformer 天生「寬」：head 軸、token 軸都能平行——這正是它為 GPU 而生的原因（第一堂 Part 5）。");
  footer(s, PA);
})();

// ============================================================ 8 Block 3 Attention
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "06", "Block 3 — Attention：prefill 與 decode 兩張臉", MEM);
  blockStepper(s, 3);
  card(s, MX, 2.35, 7.6, 3.35, BG2);
  s.addText("一個 head 的矩陣鏈：Q·Kᵀ → softmax → ·V", { x: MX + 0.3, y: 2.52, w: 7, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0 });
  const chain = [["Q·Kᵀ", COMP, "GEMM/GEMV"], ["softmax", MEM, "CUDA core"], ["·V", COMP, "GEMM/GEMV"]];
  chain.forEach(([t, c, sub], i) => {
    const x = MX + 0.4 + i * 2.35;
    obox(s, x, 3.05, 1.85, 0.6, t, c, c, 14);
    s.addText(sub, { x, y: 3.68, w: 1.85, h: 0.28, align: "center", fontFace: MONO, fontSize: 9.5, color: MUTE, margin: 0 });
    if (i < 2) s.addText("→", { x: x + 1.85, y: 3.05, w: 0.5, h: 0.6, align: "center", valign: "middle", fontFace: BODY, fontSize: 16, color: FOOTC, margin: 0 });
  });
  // prefill vs decode
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.4, y: 4.25, w: 3.4, h: 1.25, rectRadius: 0.06, fill: { color: GOODTINT }, line: { color: GOOD, width: 1 } });
  s.addText([{ text: "Prefill（吃整段）\n", options: { bold: true, color: GOOD } }, { text: "Q[6×3]·Kᵀ = GEMM · 整批算 · 高 AI → compute-bound", options: { color: INK } }], { x: MX + 0.55, y: 4.35, w: 3.1, h: 1.05, valign: "middle", fontFace: BODY, fontSize: 11, lineSpacingMultiple: 1.15, margin: 0 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 3.95, y: 4.25, w: 3.4, h: 1.25, rectRadius: 0.06, fill: { color: MEMTINT }, line: { color: MEM, width: 1 } });
  s.addText([{ text: "Decode（逐字吐）\n", options: { bold: true, color: MEM } }, { text: "q[1×3]·Kᵀ = GEMV · KV cache 從 HBM 重讀 · AI≈1 → memory-bound", options: { color: INK } }], { x: MX + 4.1, y: 4.35, w: 3.1, h: 1.05, valign: "middle", fontFace: BODY, fontSize: 11, lineSpacingMultiple: 1.15, margin: 0 });
  gpuMapCard(s, [
    ["Q·Kᵀ / ·V", "Tensor Core（矩陣乘）"],
    ["softmax", "CUDA core（逐元素/歸約）"],
    ["KV cache", "住 HBM（塞不進晶片）"],
    ["decode 瓶頸", "每步從 HBM 重串流 KV"],
  ], "prefill=compute / decode=memory", MEM);
  takeaway(s, "同一段 attention，prefill 吃算力、decode 吃頻寬——第一堂「<5% 利用率之謎」就發生在這個 block。");
  footer(s, PA);
})();

// ============================================================ 9 Block 4 Add & LayerNorm
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "07", "Block 4 — Add & LayerNorm：不上 tensor core 的那些", MEM);
  blockStepper(s, 4);
  card(s, MX, 2.35, 7.6, 3.35, BG2);
  s.addText("殘差相加（Add）+ 層正規化（LayerNorm）", { x: MX + 0.3, y: 2.55, w: 7, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0 });
  obox(s, MX + 0.5, 3.2, 1.7, 0.6, "輸入 X", MEM, MEM, 12);
  s.addText("+", { x: MX + 2.2, y: 3.2, w: 0.5, h: 0.6, align: "center", valign: "middle", fontFace: MONO, fontSize: 20, color: FOOTC, margin: 0 });
  obox(s, MX + 2.7, 3.2, 1.7, 0.6, "子層輸出", COMP, COMP, 12);
  s.addText("→", { x: MX + 4.4, y: 3.2, w: 0.5, h: 0.6, align: "center", valign: "middle", fontFace: BODY, fontSize: 16, color: FOOTC, margin: 0 });
  obox(s, MX + 4.9, 3.2, 2.2, 0.6, "LayerNorm", GOOD, GOOD, 12);
  s.addText("這些是逐元素相加、求平均與變異數（歸約）——運算量很少，但要把整份 activation 從 HBM 讀進來、寫回去。", { x: MX + 0.3, y: 4.05, w: 7.0, h: 0.9, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.5, y: 4.95, w: 6.6, h: 0.6, rectRadius: 0.06, fill: { color: COMPTINT }, line: { color: COMP, width: 1 } });
  s.addText([{ text: "工程實務：", options: { bold: true, color: COMP } }, { text: "常和前後運算「fuse（融合）」成一個 kernel，避免多跑一趟 HBM。", options: { color: INK } }], { x: MX + 0.7, y: 4.95, w: 6.2, h: 0.6, valign: "middle", fontFace: BODY, fontSize: 11.5, margin: 0 });
  gpuMapCard(s, [
    ["主要運算", "逐元素加 + 歸約（mean/var）"],
    ["用到單元", "CUDA core（非 tensor core）"],
    ["資料流動", "整份 activation 進出 HBM"],
    ["優化手段", "kernel fusion 減少 HBM 往返"],
  ], "memory-bound（算少、搬多）", MEM);
  takeaway(s, "Transformer 不是只有大矩陣乘。這些「小」運算是純 memory-bound，融合它們是省頻寬的日常功夫。");
  footer(s, PA);
})();

// ============================================================ 10 Block 5 FFN
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "08", "Block 5 — FFN：另一組大矩陣乘", COMP);
  blockStepper(s, 5);
  card(s, MX, 2.35, 7.6, 3.35, BG2);
  s.addText("每個 token 獨立過兩層：6 → 12 → 6（中間放大 4×，玩具用 2×）", { x: MX + 0.3, y: 2.55, w: 7, h: 0.35, fontFace: HEAD, fontSize: 13.5, bold: true, color: INK, margin: 0 });
  obox(s, MX + 0.5, 3.25, 1.5, 0.7, "X [6]", MEM, MEM, 12);
  arrow(s, MX + 2.0, 3.6, 0.7, COMP, "W1 (6→12)", true);
  obox(s, MX + 2.7, 3.25, 1.6, 0.7, "h [12]\nGELU", COMP, COMP, 11);
  arrow(s, MX + 4.3, 3.6, 0.7, COMP, "W2 (12→6)", true);
  obox(s, MX + 5.0, 3.25, 1.5, 0.7, "out [6]", GOOD, GOOD, 12);
  s.addText("兩個 GEMM，中間夾一個非線性（GELU）。真實模型 FFN 常佔一層 2/3 的參數與算力。token 之間零依賴 → 整批一起算。", { x: MX + 0.3, y: 4.25, w: 7.0, h: 1.1, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.35, margin: 0 });
  gpuMapCard(s, [
    ["主要運算", "兩個 GEMM（W1、W2）"],
    ["用到單元", "Tensor Core 主場"],
    ["GELU", "CUDA core（逐元素）"],
    ["平行軸", "token 軸天然平行"],
  ], "compute-bound（大 GEMM）", COMP);
  takeaway(s, "FFN 和 QKV 一樣是 tensor core 的主場。一層裡「大矩陣乘吃算力、其餘吃頻寬」的節奏就此定調。");
  footer(s, PA);
})();

// ============================================================ 11 彙整表（互動②）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "09", "一頁看完：block → GPU 單元 → 被誰卡住", MEM);
  const rows = [
    ["0", "Embed + 位置", "查表 gather + 加", "CUDA core", "HBM 讀幾列", "memory", MEM],
    ["1", "QKV 投影", "GEMM", "Tensor Core", "tile 留 shared", "compute", COMP],
    ["2", "多頭切分", "拆平行軸", "warp / batched", "—", "結構", PURP],
    ["3", "Attention", "QKᵀ/softmax/·V", "Tensor+CUDA", "KV cache 住 HBM", "pre=C / dec=M", MEM],
    ["4", "Add & Norm", "逐元素 + 歸約", "CUDA core", "activation 進出 HBM", "memory", MEM],
    ["5", "FFN", "兩個 GEMM", "Tensor Core", "tile 留 shared", "compute", COMP],
  ];
  // 表頭
  const cx = [MX, MX + 0.55, MX + 2.35, MX + 4.55, MX + 6.7, MX + 9.2], cw = [0.55, 1.8, 2.2, 2.15, 2.5, 2.7];
  const heads = ["#", "Block", "運算", "GPU 單元", "資料/搬運", "判定"];
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 1.85, w: 11.9, h: 0.5, fill: { color: BG3 }, line: { type: "none" } });
  heads.forEach((h, i) => s.addText(h, { x: cx[i], y: 1.85, w: cw[i], h: 0.5, valign: "middle", align: i === 0 ? "center" : "left", fontFace: HEAD, fontSize: 12.5, bold: true, color: INK, margin: 0.05 }));
  rows.forEach((r, i) => {
    const y = 2.35 + i * 0.6;
    if (i % 2) s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 11.9, h: 0.6, fill: { color: BG2 }, line: { type: "none" } });
    s.addText(r[0], { x: cx[0], y, w: cw[0], h: 0.6, align: "center", valign: "middle", fontFace: MONO, fontSize: 12, bold: true, color: r[6], margin: 0 });
    s.addText(r[1], { x: cx[1], y, w: cw[1], h: 0.6, valign: "middle", fontFace: BODY, fontSize: 12, bold: true, color: INK, margin: 0.05 });
    s.addText(r[2], { x: cx[2], y, w: cw[2], h: 0.6, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0.05 });
    s.addText(r[3], { x: cx[3], y, w: cw[3], h: 0.6, valign: "middle", fontFace: MONO, fontSize: 11, color: r[3].includes("Tensor") ? COMP : MEM, margin: 0.05 });
    s.addText(r[4], { x: cx[4], y, w: cw[4], h: 0.6, valign: "middle", fontFace: BODY, fontSize: 10.5, color: MUTE, margin: 0.05 });
    const bc = r[5].includes("compute") ? COMP : r[5].includes("結構") ? PURP : r[5].includes("/") ? MEM : MEM;
    s.addText(r[5], { x: cx[5], y, w: cw[5], h: 0.6, valign: "middle", fontFace: MONO, fontSize: 10.5, bold: true, color: bc, margin: 0.05 });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 5.95, w: 11.9, h: 0.62, rectRadius: 0.1, fill: { color: GOODTINT }, line: { color: GOOD, width: 1 } });
  s.addText([{ text: "🔬 互動環節②　", options: { bold: true, color: GOOD } }, { text: "開 interactive/transformer_map.html：全景 → Block → Attention → Head → 計算子(L2⟷HBM) → FlashAttention(線上 softmax) → 硬體，切 訓練/Prefill/Decode 看資料流。", options: { color: INK } }], { x: MX + 0.3, y: 5.95, w: 11.3, h: 0.62, valign: "middle", fontFace: BODY, fontSize: 12, margin: 0 });
  footer(s, PA);
})();

// ============================================================ 12 收束 A
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "10", "Part A 收束：一張卡就跑得完", MEM);
  // 一張卡裝三樣東西
  gpuGlyph(s, MX + 0.3, 2.4, 2.6, 2.9, "1 顆 GPU", MEM, "");
  s.addText("HBM 80–192 GB", { x: MX + 0.3, y: 4.55, w: 2.6, h: 0.3, align: "center", fontFace: MONO, fontSize: 11, color: MEM, margin: 0 });
  const items = [["模型權重", COMP], ["KV cache", MEM], ["中間 activation", GOOD]];
  items.forEach(([t, c], i) => {
    const y = 2.55 + i * 0.9;
    s.addShape(pres.shapes.LINE, { x: MX + 3.1, y: y + 0.3, w: 0.5, h: 0, line: { color: c, width: 2, endArrowType: "triangle" } });
    obox(s, MX + 3.65, y, 2.5, 0.62, t, c, c, 13);
  });
  s.addText("玩具 Transformer——甚至 7B 這種真實小模型——的權重、KV cache、activation 全部塞得進一顆 GPU 的 HBM。所有 block 的資料流都在同一顆晶片裡走完，沒有跨卡問題。", { x: MX + 6.5, y: 2.5, w: 6.0, h: 2.0, valign: "top", fontFace: BODY, fontSize: 14, color: MUTE, lineSpacingMultiple: 1.4, margin: 0 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 6.5, y: 4.55, w: 6.0, h: 0.95, rectRadius: 0.08, fill: { color: WARNTINT }, line: { color: WARN, width: 1.5 } });
  s.addText([{ text: "但是……　", options: { bold: true, color: WARN } }, { text: "70B、175B、甚至上兆參數的模型呢？一顆 HBM 放不下。這就是 Part B。", options: { color: INK } }], { x: MX + 6.75, y: 4.55, w: 5.5, h: 0.95, valign: "middle", fontFace: BODY, fontSize: 13.5, margin: 0 });
  takeaway(s, "單卡的世界：看 roofline、看記憶體階層就夠了。跨過一顆 GPU 的容量，遊戲規則就變了。", COMP);
  footer(s, PA);
})();

// ============================================================ 13 一張卡裝不下
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "11", "為什麼一張卡不夠：三筆帳都在漲", COMP);
  s.addText("以推論/訓練一個大模型為例，HBM 要同時裝下這幾樣——很快就爆掉 80–192 GB。", { x: MX, y: 1.45, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 13.5, color: MUTE, margin: 0 });
  const cards = [
    ["模型權重", "70B fp16 ≈ 140 GB\n175B ≈ 350 GB", "一張 H100(80GB)/H200(141GB) 就放不下 70B 權重", COMP],
    ["KV cache", "隨 序列長 × batch\n× 層數 線性長大", "長上下文 + 高併發 → 動輒數十~上百 GB", MEM],
    ["訓練狀態", "梯度 + Adam 狀態\n≈ 參數量的 3–4×", "訓練時遠比推論更吃容量（第一堂 Part 3）", GOOD],
  ];
  cards.forEach(([t, n, d, c], i) => {
    const x = MX + i * 4.05;
    card(s, x, 2.1, 3.75, 3.5, BG2, c);
    s.addText(t, { x: x + 0.25, y: 2.3, w: 3.25, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: c, margin: 0 });
    s.addText(n, { x: x + 0.25, y: 2.85, w: 3.25, h: 1.0, fontFace: MONO, fontSize: 14, bold: true, color: INK, lineSpacingMultiple: 1.2, margin: 0 });
    s.addShape(pres.shapes.LINE, { x: x + 0.25, y: 3.95, w: 3.25, h: 0, line: { color: LINE, width: 0.75 } });
    s.addText(d, { x: x + 0.25, y: 4.1, w: 3.25, h: 1.35, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.35, margin: 0 });
  });
  takeaway(s, "模型放不進一顆 HBM → 必須拆到多張卡。問題是：拆什麼、卡跟卡之間要搬什麼？");
  footer(s, PB);
})();

// ============================================================ 14 資料平行 DP
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "12", "資料平行 (Data Parallel)：最直覺的第一招", MEM);
  s.addText("每張卡放「一份完整模型」，把 batch 切開各算各的，再把梯度對齊。", { x: MX, y: 1.45, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 13.5, color: MUTE, margin: 0 });
  // 4 GPU each full model
  [0, 1, 2, 3].forEach((i) => {
    const x = MX + i * 2.15;
    gpuGlyph(s, x, 2.15, 1.85, 1.5, `GPU ${i}`, MEM, "整份模型");
    s.addText(`batch 第 ${i + 1}/4 份`, { x, y: 3.7, w: 1.85, h: 0.3, align: "center", fontFace: BODY, fontSize: 10.5, color: MUTE, margin: 0 });
  });
  // all-reduce bar
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 4.25, w: 8.4, h: 0.62, rectRadius: 0.1, fill: { color: MEMTINT }, line: { color: MEM, width: 1.5 } });
  s.addText("↕ 每步：梯度 all-reduce（把 4 張卡的梯度加總、再發回每張卡）", { x: MX, y: 4.25, w: 8.4, h: 0.62, align: "center", valign: "middle", fontFace: BODY, fontSize: 12.5, bold: true, color: MEM, margin: 0 });
  card(s, MX + 8.7, 2.15, 3.9, 2.72, BG2, GOOD);
  s.addText("DP 擅長什麼", { x: MX + 8.95, y: 2.32, w: 3.4, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: GOOD, margin: 0 });
  ["實作最簡單、最成熟", "訓練吞吐幾乎線性擴充", "推論可服務更多併發請求", "模型「放得下一張卡」時的首選"].forEach((t, i) => {
    s.addShape(pres.shapes.OVAL, { x: MX + 8.95, y: 2.85 + i * 0.47 + 0.05, w: 0.09, h: 0.09, fill: { color: GOOD }, line: { type: "none" } });
    s.addText(t, { x: MX + 9.2, y: 2.85 + i * 0.47, w: 3.2, h: 0.42, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
  });
  takeaway(s, "資料平行＝複製模型、切資料、同步梯度。切的是「資料」，不是「模型」——這正是它的天花板。");
  footer(s, PB);
})();

// ============================================================ 15 DP 的問題
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "13", "資料平行的四個問題（本堂重點）", WARN);
  const probs = [
    ["① 模型沒有變小", "每張卡仍要放整份模型 → 「一張卡裝不下」的模型，DP 根本救不了。", WARN],
    ["② 通訊隨規模長大", "梯度 all-reduce 每步要搬 ≈ 2× 參數量；模型越大、卡越多，同步成本越兇。", COMP],
    ["③ 記憶體重複浪費", "權重、梯度、optimizer 狀態每張卡各存一份 → ZeRO/FSDP 把它們切開，但換來更多通訊。", MEM],
    ["④ 推論延遲無解", "decode 是 memory-bound；DP 只增吞吐、對單一請求的延遲沒幫助，KV cache 還各卡獨立。", PURP],
  ];
  probs.forEach(([t, d, c], i) => {
    const x = MX + (i % 2) * 6.05, y = 1.95 + Math.floor(i / 2) * 1.95;
    card(s, x, y, 5.75, 1.75, BG2, c);
    s.addText(t, { x: x + 0.28, y: y + 0.2, w: 5.2, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: c, margin: 0 });
    s.addText(d, { x: x + 0.28, y: y + 0.68, w: 5.25, h: 0.95, valign: "top", fontFace: BODY, fontSize: 12.5, color: MUTE, lineSpacingMultiple: 1.35, margin: 0 });
  });
  takeaway(s, "①最致命：DP 切資料、不切模型。要裝下大模型、又想壓通訊，就得把「模型本身」拆開。", WARN);
  footer(s, PB);
})();

// ============================================================ 16 TP/PP/EP taxonomy
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "14", "切模型本身：張量 / 管線 / 專家平行", COMP);
  s.addText("既然 DP 不切模型，就換三種「切模型」的平行——各自切不同的軸，付不同的通訊代價。", { x: MX, y: 1.45, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 13.5, color: MUTE, margin: 0 });
  const cols = [
    ["Tensor Parallel", "TP · 張量平行", "切「層內」：一個矩陣乘橫著切到多卡", "每層 2 次 all-reduce", "通訊最重 → 必須待在 NVLink 域", COMP],
    ["Pipeline Parallel", "PP · 管線平行", "切「層間」：不同層放不同卡，像流水線", "點對點傳 activation（p2p）", "有 bubble（頭尾閒置）", MEM],
    ["Expert Parallel", "EP · 專家平行", "切「MoE 專家」：不同 expert 放不同卡", "all-to-all 路由 token", "省 FLOPs、不省通訊", PURP],
  ];
  cols.forEach(([en, zh, what, comm, note, c], i) => {
    const x = MX + i * 4.05;
    card(s, x, 2.05, 3.75, 3.55, BG2, c);
    s.addText(en, { x: x + 0.25, y: 2.22, w: 3.25, h: 0.35, fontFace: MONO, fontSize: 12.5, bold: true, color: c, margin: 0 });
    s.addText(zh, { x: x + 0.25, y: 2.56, w: 3.25, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0 });
    s.addText([{ text: "切什麼\n", options: { color: FOOTC, fontSize: 10 } }, { text: what, options: { color: MUTE } }], { x: x + 0.25, y: 3.05, w: 3.3, h: 0.95, valign: "top", fontFace: BODY, fontSize: 11.5, lineSpacingMultiple: 1.25, margin: 0 });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.25, y: 4.05, w: 3.3, h: 0.5, rectRadius: 0.05, fill: { color: BG3 }, line: { type: "none" } });
    s.addText(comm, { x: x + 0.25, y: 4.05, w: 3.3, h: 0.5, align: "center", valign: "middle", fontFace: BODY, fontSize: 11, bold: true, color: c, margin: 0 });
    s.addText(note, { x: x + 0.25, y: 4.7, w: 3.3, h: 0.8, valign: "top", fontFace: BODY, fontSize: 11, color: MUTE, lineSpacingMultiple: 1.25, margin: 0 });
  });
  takeaway(s, "實務是「3D/4D 混合」：DP + TP + PP(+ EP) 疊用。切得越細、卡間通訊越密——通訊就成了新瓶頸。");
  footer(s, PB);
})();

// ============================================================ 17 頻寬階梯 × 落位
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "15", "通訊變瓶頸：頻寬階梯決定誰住哪", MEM);
  s.addText("卡間搬資料的速度差一個數量級往下掉。哪種平行「通訊多」，就得住在「線快」的那一層。", { x: MX, y: 1.45, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 13.5, color: MUTE, margin: 0 });
  const ladder = [
    ["晶片內 HBM", "~3.35 TB/s", "（單卡基準）", INK, 6.6],
    ["NVLink 5 / NVSwitch", "~1.8 TB/s / GPU", "scale-up：TP、EP 住這裡", GOOD, 5.4],
    ["跨節點 InfiniBand / Spectrum-X", "~50–100 GB/s / GPU", "scale-out：DP、PP 可容忍", COMP, 2.6],
    ["PCIe Gen5", "~64 GB/s", "沒 NVLink 時的卡間路（會被拖死）", WARN, 1.7],
  ];
  ladder.forEach(([t, bw, use, c, w], i) => {
    const y = 2.05 + i * 0.92;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: w, h: 0.72, rectRadius: 0.06, fill: { color: c === INK ? BG3 : c === GOOD ? GOODTINT : c === COMP ? COMPTINT : WARNTINT }, line: { color: c === INK ? LINE : c, width: 1.5 } });
    s.addText(t, { x: MX + 0.25, y, w: w - 0.4, h: 0.72, valign: "middle", fontFace: HEAD, fontSize: 13.5, bold: true, color: c === INK ? INK : c, margin: 0 });
    s.addText(bw, { x: MX + w + 0.2, y, w: 2.4, h: 0.72, valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: c === INK ? INK : c, margin: 0 });
    s.addText(use, { x: MX + w + 2.6, y, w: 5.0, h: 0.72, valign: "middle", fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  });
  takeaway(s, "心法：把「通訊最兇」的平行（TP/EP）綁在最快的線（NVLink）；「通訊稀」的（DP/PP）才放去跨節點。", COMP);
  footer(s, PB);
})();

// ============================================================ 18 scale-up NVLink/NVSwitch/NVL72
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "16", "NVIDIA 解法① scale-up：把多顆綁成一顆大 GPU", GOOD);
  s.addText("NVLink + NVSwitch 用「記憶體語意」把整個機架的 GPU 連成一個高頻寬域——TP/EP 的家。", { x: MX, y: 1.45, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 13.5, color: MUTE, margin: 0 });
  const tech = [
    ["NVLink 5", "1.8 TB/s / GPU", "18 條 × 100 GB/s；≈ PCIe Gen5 的 14×", GOOD],
    ["NVSwitch（4 代）", "全連通 fabric", "72 顆非阻塞互連；可擴到 576 GPU / 1 PB/s", MEM],
    ["GB200 NVL72", "72 顆 = 一個域", "130 TB/s 聚合頻寬，當一顆「大 GPU」用", COMP],
  ];
  tech.forEach(([t, n, d, c], i) => {
    const y = 2.05 + i * 0.98;
    card(s, MX, y, 7.4, 0.85, BG2, c);
    s.addText(t, { x: MX + 0.25, y, w: 2.55, h: 0.85, valign: "middle", fontFace: HEAD, fontSize: 14.5, bold: true, color: c, margin: 0 });
    s.addText(n, { x: MX + 2.8, y, w: 2.1, h: 0.85, valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: INK, margin: 0 });
    s.addText(d, { x: MX + 4.95, y, w: 2.35, h: 0.85, valign: "middle", fontFace: BODY, fontSize: 10.8, color: MUTE, margin: 0 });
  });
  card(s, MX + 7.7, 2.05, 4.9, 2.91, PURPTINT, PURP);
  s.addText("2026 最新 · Rubin", { x: MX + 7.95, y: 2.22, w: 4.4, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: PURP, margin: 0 });
  s.addText([
    { text: "NVLink 6", options: { bold: true, color: INK, fontSize: 13 } },
    { text: "：3.6 TB/s / GPU（NVLink 5 的 2×）\n", options: { color: MUTE } },
    { text: "Vera Rubin NVL72", options: { bold: true, color: INK, fontSize: 13 } },
    { text: "：260 TB/s 聚合\n", options: { color: MUTE } },
    { text: "CES 2026 發表、H2 2026 出貨；NVL144/CPX 版把 prefill 拆出來做", options: { color: MUTE } },
  ], { x: MX + 7.95, y: 2.62, w: 4.45, h: 2.25, valign: "top", fontFace: BODY, fontSize: 12, lineSpacingMultiple: 1.3, margin: 0 });
  takeaway(s, "scale-up 的意義：一個 NVLink 域內，72 顆 GPU 的 HBM 像一大池——大模型的 TP/EP 就攤在這池裡跑。", GOOD);
  footer(s, PB);
})();

// ============================================================ 19 in-network SHARP + NCCL
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "17", "NVIDIA 解法② 把 all-reduce 搬進交換器：SHARP", COMP);
  card(s, MX, 2.0, 5.85, 3.4, BG2, MUTE);
  s.addText("傳統 all-reduce", { x: MX + 0.3, y: 2.18, w: 5.2, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: MUTE, margin: 0 });
  s.addText("每張卡互相收發、在 GPU 上做加總 → 佔 NVLink 頻寬、也佔 GPU 的 SM 去算加法。", { x: MX + 0.3, y: 2.6, w: 5.25, h: 0.85, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  [0, 1, 2, 3].forEach((i) => gpuGlyph(s, MX + 0.35 + i * 1.32, 3.55, 1.15, 0.95, `G${i}`, MUTE));
  s.addText("互相收發、GPU 自己加", { x: MX + 0.3, y: 4.6, w: 5.3, h: 0.35, align: "center", fontFace: BODY, fontSize: 10.5, color: FOOTC, margin: 0 });
  card(s, MX + 6.15, 2.0, 5.85, 3.4, BG2, COMP);
  s.addText("SHARP（in-network reduction）", { x: MX + 6.45, y: 2.18, w: 5.3, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: COMP, margin: 0 });
  s.addText("加總直接在 NVSwitch / IB 交換器的晶片裡做完 → 省 NVLink 頻寬、也把 GPU 的 SM 解放出來算模型。", { x: MX + 6.45, y: 2.6, w: 5.25, h: 0.85, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  [0, 1, 2, 3].forEach((i) => gpuGlyph(s, MX + 6.5 + i * 1.32, 3.9, 1.15, 0.62, `G${i}`, COMP));
  box(s, MX + 8.0, 3.55, 2.15, 0.32, "NVSwitch：算好加總", COMP, BG, 10);
  s.addText("交換器算、發回結果", { x: MX + 6.45, y: 4.6, w: 5.3, h: 0.35, align: "center", fontFace: BODY, fontSize: 10.5, color: COMP, margin: 0 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 5.6, w: 11.9, h: 0.32, rectRadius: 0.05, fill: { color: BG3 }, line: { type: "none" } });
  s.addText([{ text: "NCCL　", options: { bold: true, color: MEM } }, { text: "是這一切的軟體層：all-reduce / all-gather / reduce-scatter / all-to-all 的實作，2.27 起 NVLink 與 IB 都能吃 SHARP。", options: { color: MUTE } }], { x: MX + 0.25, y: 5.6, w: 11.5, h: 0.32, valign: "middle", fontFace: BODY, fontSize: 11.5, margin: 0 });
  takeaway(s, "梯度同步（DP）、層內同步（TP）都靠 all-reduce——把它 offload 進交換器，通訊瓶頸就鬆一大截。", COMP);
  footer(s, PB);
})();

// ============================================================ 20 scale-out IB/Spectrum-X/GPUDirect
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "18", "NVIDIA 解法③ scale-out：跨節點的網路", MEM);
  s.addText("一個機架塞不下時，用網路把多台節點串成 AI factory——這裡走「訊息語意」（RDMA）。", { x: MX, y: 1.45, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 13.5, color: MUTE, margin: 0 });
  const tech = [
    ["InfiniBand（Quantum）", "NDR 400 / XDR 800 Gb/s", "HPC 老本行，原生支援 SHARP", MEM],
    ["Spectrum-X 乙太", "為 AI 調過的以太網", "adaptive routing + 壅塞控制，跑 RoCEv2", COMP],
    ["GPUDirect RDMA", "遠端 NIC 直達 HBM", "繞過 CPU，跨節點也能直接讀寫對方 GPU", GOOD],
  ];
  tech.forEach(([t, n, d, c], i) => {
    const y = 2.1 + i * 1.05;
    card(s, MX, y, 8.0, 0.92, BG2, c);
    s.addText(t, { x: MX + 0.25, y, w: 3.1, h: 0.92, valign: "middle", fontFace: HEAD, fontSize: 14, bold: true, color: c, margin: 0 });
    s.addText(n, { x: MX + 3.35, y, w: 2.5, h: 0.92, valign: "middle", fontFace: MONO, fontSize: 12, bold: true, color: INK, margin: 0 });
    s.addText(d, { x: MX + 5.85, y, w: 1.95, h: 0.92, valign: "middle", fontFace: BODY, fontSize: 10.5, color: MUTE, margin: 0 });
  });
  card(s, MX + 8.3, 2.1, 4.3, 3.27, BG3, MUTE);
  s.addText("scale-up vs scale-out", { x: MX + 8.55, y: 2.28, w: 3.8, h: 0.35, fontFace: HEAD, fontSize: 13.5, bold: true, color: INK, margin: 0 });
  s.addText([
    { text: "scale-up（NVLink）\n", options: { bold: true, color: GOOD } },
    { text: "記憶體語意、~TB/s、機架內 → 綁 TP/EP\n\n", options: { color: MUTE } },
    { text: "scale-out（IB/乙太）\n", options: { bold: true, color: COMP } },
    { text: "訊息語意、~百 Gb/s、跨節點 → 綁 DP/PP", options: { color: MUTE } },
  ], { x: MX + 8.55, y: 2.7, w: 3.85, h: 2.5, valign: "top", fontFace: BODY, fontSize: 11.5, lineSpacingMultiple: 1.3, margin: 0 });
  takeaway(s, "兩層網路差一個數量級：所以「通訊密的平行留機架內、稀的才跨節點」不是選擇，是被頻寬逼的。");
  footer(s, PB);
})();

// ============================================================ 21 落到 LLM
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "19", "組起來：一個大模型怎麼跑在 NVL72 上", GOOD);
  s.addText("把前面的技術疊起來，看一個「一張卡裝不下」的模型實際怎麼被攤開。", { x: MX, y: 1.45, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 13.5, color: MUTE, margin: 0 });
  const steps = [
    ["① 一個 NVLink 域 = 一顆大 GPU", "GB200 NVL72 把 72 顆連成 130 TB/s 的域 → 大模型的權重攤在這一池 HBM 裡放得下。", GOOD],
    ["② 域內做 TP + EP", "層內矩陣切到多卡、MoE 專家散到多卡，每層的 all-reduce / all-to-all 跑在 NVLink，並用 SHARP 加速。", COMP],
    ["③ 跨域做 DP + PP", "多個 NVL72 之間用 InfiniBand / Spectrum-X 做資料平行的梯度同步、管線的 activation 傳遞。", MEM],
    ["④ NCCL 統籌", "上層一律用 NCCL 呼叫集合通訊，自動選 NVLink 還是網路、要不要走 SHARP。", PURP],
  ];
  steps.forEach(([t, d, c], i) => {
    const y = 1.95 + i * 0.98;
    s.addShape(pres.shapes.RECTANGLE, { x: MX, y: y + 0.05, w: 0.1, h: 0.78, fill: { color: c }, line: { type: "none" } });
    s.addText(t, { x: MX + 0.3, y, w: 4.6, h: 0.88, valign: "middle", fontFace: HEAD, fontSize: 13.5, bold: true, color: c, margin: 0 });
    s.addText(d, { x: MX + 5.0, y, w: 7.5, h: 0.88, valign: "middle", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.25, margin: 0 });
  });
  takeaway(s, "真實訓練＝DP × TP × PP × EP 的混合，每一軸配一種通訊、一條線——互連技術就是讓這套疊得起來。", GOOD);
  footer(s, PB);
})();

// ============================================================ 22 CMX 2026
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "20", "2026 新招 CMX：把 KV cache 搬出 HBM", MEM);
  s.addText("呼應第一堂：decode 是 memory-bound，KV cache 又吃 HBM 容量。CMX 給它一層專屬儲存。", { x: MX, y: 1.45, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 13.5, color: MUTE, margin: 0 });
  const tiers = [["GPU HBM", "最快、最貴、最小", COMP], ["CPU DRAM", "中間層", MEM], ["NVMe flash (CMX)", "大、便宜、放 KV", GOOD]];
  tiers.forEach(([t, d, c], i) => {
    const y = 2.1 + i * 0.85;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 5.6 - i * 0.5, h: 0.66, rectRadius: 0.06, fill: { color: c === COMP ? COMPTINT : c === MEM ? MEMTINT : GOODTINT }, line: { color: c, width: 1.5 } });
    s.addText(t, { x: MX + 0.25, y, w: 3.0, h: 0.66, valign: "middle", fontFace: HEAD, fontSize: 13.5, bold: true, color: c, margin: 0 });
    s.addText(d, { x: MX + 2.9, y, w: 2.5, h: 0.66, valign: "middle", fontFace: BODY, fontSize: 10.5, color: MUTE, margin: 0 });
  });
  varrow(s, MX + 6.0, 2.3, 2.0, MEM, true);
  s.addText("BlueField-4\n+ Spectrum-X\n排 I/O", { x: MX + 6.2, y: 2.55, w: 1.6, h: 1.4, valign: "middle", fontFace: BODY, fontSize: 11, color: MEM, lineSpacingMultiple: 1.25, margin: 0 });
  card(s, MX + 8.0, 2.1, 4.6, 2.96, BG2, GOOD);
  s.addText("CMX 帶來什麼", { x: MX + 8.25, y: 2.28, w: 4.1, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: GOOD, margin: 0 });
  [["~5×", "token 吞吐"], ["~4×", "能源效率"], ["~2×", "資料載入"]].forEach(([n, d], i) => {
    const y = 2.72 + i * 0.62;
    s.addText(n, { x: MX + 8.25, y, w: 1.3, h: 0.55, valign: "middle", fontFace: MONO, fontSize: 20, bold: true, color: GOOD, margin: 0 });
    s.addText(d, { x: MX + 9.55, y, w: 2.9, h: 0.55, valign: "middle", fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  });
  s.addText("Dynamo + NIXL 統籌 prefill/decode/KV，並支援 prefill–decode 拆分與前綴重用。", { x: MX + 8.25, y: 4.55, w: 4.15, h: 0.5, valign: "top", fontFace: BODY, fontSize: 10.5, color: FOOTC, lineSpacingMultiple: 1.2, margin: 0 });
  takeaway(s, "互連技術不只服務訓練：把 KV cache 卸到網路掛載的 flash，直接打 decode 的頻寬/容量牆。");
  footer(s, PB);
})();

// ============================================================ 23 互動環節（parallelism_map）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "21", "互動環節③：多卡平行地圖", GOOD);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: 1.9, w: 11.9, h: 1.0, rectRadius: 0.1, fill: { color: GOODTINT }, line: { color: GOOD, width: 1.5 } });
  s.addText([{ text: "🕸️ 切出去開　", options: { bold: true, color: GOOD } }, { text: "interactive/parallelism_map.html", options: { bold: true, color: INK, fontFace: MONO } }, { text: "　——把同一個玩具 Transformer 攤到多張卡上。", options: { color: INK } }], { x: MX + 0.3, y: 1.9, w: 11.3, h: 1.0, valign: "middle", fontFace: BODY, fontSize: 14, margin: 0 });
  const layers = [
    ["1 單卡", "整份模型在一顆 HBM → 沒有通訊", MEM],
    ["2 裝不下", "模型長大 → 權重/KV/狀態溢出一顆卡", WARN],
    ["3 資料平行 DP", "複製模型、切 batch、梯度 all-reduce（看四個問題）", MEM],
    ["4 張量平行 TP", "把「一層」的 Wq/Wk/Wv/Wo+W1/W2 每個矩陣切 4 片、每層 all-reduce", COMP],
    ["5 管線平行 PP", "整層分段（不切一層內部）、p2p 傳 activation、看 bubble", GOOD],
    ["6 互連硬體", "各平行走 NVLink/NVSwitch 還是 IB/Spectrum-X（scale-up vs scale-out）", PURP],
  ];
  layers.forEach(([t, d, c], i) => {
    const y = 3.1 + (i % 3) * 0.85, x = MX + Math.floor(i / 3) * 6.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.75, h: 0.72, rectRadius: 0.06, fill: { color: BG2 }, line: { color: c, width: 1 } });
    s.addText(t, { x: x + 0.2, y, w: 1.7, h: 0.72, valign: "middle", fontFace: HEAD, fontSize: 12.5, bold: true, color: c, margin: 0 });
    s.addText(d, { x: x + 1.95, y, w: 3.7, h: 0.72, valign: "middle", fontFace: BODY, fontSize: 10.8, color: MUTE, margin: 0 });
  });
  s.addText("操作：數字鍵 1–6 跳層 · Esc 回上層　（TP / PP 兩層會畫出「一層」的權重矩陣怎麼被切）", { x: MX, y: 5.7, w: 11.9, h: 0.35, align: "center", fontFace: MONO, fontSize: 11, color: FOOTC, margin: 0 });
  takeaway(s, "抽象的「切哪個軸、走哪條線」變成看得到、玩得到的東西——這是本堂的落地。", GOOD);
  footer(s, PB);
})();

// ============================================================ 24 收束
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "22", "帶走三句話", COMP);
  const lines = [
    ["1", "一張卡：逐 block 看單元", "Transformer 每個 block 落在 tensor core（大 GEMM）或 CUDA core（softmax/norm/查表）；大矩陣吃算力、其餘吃頻寬。", MEM],
    ["2", "多卡：DP 切資料、不切模型", "資料平行救不了「裝不下」的模型，還有通訊/重複/延遲四問題 → 得靠 TP/PP/EP 切模型本身。", COMP],
    ["3", "互連決定切法的可行性", "通訊階梯逼你把 TP/EP 綁 NVLink 域、DP/PP 才跨節點；NVSwitch+SHARP、Rubin/NVLink6、CMX 都在鬆這道牆。", GOOD],
  ];
  lines.forEach(([n, t, d, c], i) => {
    const y = 1.95 + i * 1.35;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 0.85, h: 0.85, rectRadius: 0.12, fill: { color: c }, line: { type: "none" } });
    s.addText(n, { x: MX, y, w: 0.85, h: 0.85, align: "center", valign: "middle", fontFace: MONO, fontSize: 30, bold: true, color: BG, margin: 0 });
    s.addText(t, { x: MX + 1.1, y, w: 11.0, h: 0.5, fontFace: HEAD, fontSize: 18, bold: true, color: c, margin: 0 });
    s.addText(d, { x: MX + 1.1, y: y + 0.52, w: 11.2, h: 0.75, valign: "top", fontFace: BODY, fontSize: 13, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  });
  s.addText("下一步：把 demo 在多卡 GPU 上實跑，量 all-reduce 頻寬與 TP 的通訊佔比。", { x: MX, y: 6.15, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 12.5, color: FOOTC, margin: 0 });
  footer(s, PB);
})();

pres.writeFile({ fileName: "../class2_transformer_gpu.pptx" }).then((f) => console.log("✅ 產生：" + f + "（" + PAGE + " 頁）")).catch((e) => console.error(e));
