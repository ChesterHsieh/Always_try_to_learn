// S5 — 平行運算與軟硬體共同演化：從 CNN 到 Transformer 到混合架構
// 產生 ../s5_parallelism_codesign.pptx。沿用 S1–S4 的深色「矽晶」主題。
const pptxgen = require("pptxgenjs");

const BG = "0E1726", BG2 = "16233A", BG3 = "1C2E4A";
const INK = "EAF1FB", MUTE = "8FA6C4", LINE = "2A3D5C", FOOTC = "5C7299";
const MEM = "38BDF8", COMP = "F59E0B", WARN = "FB7185", GOOD = "34D399";
const MEMTINT = "10455F", COMPTINT = "4A3410", WARNTINT = "4A2433";
const HEAD = "PingFang TC", BODY = "PingFang TC", MONO = "Menlo";

const W = 13.33, H = 7.5, MX = 0.7, TITLE_Y = 0.62, FOOT_Y = 7.05, TOTAL = 13;
const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 });

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "GPU 記憶體與資料搬遷讀書會";
pres.title = "S5 — 平行運算與軟硬體共同演化";

const base = (s) => { s.background = { color: BG }; };
function runningHeader(s) {
  s.addText("GPU 記憶體與資料搬遷讀書會 · S5", { x: W - 5.2, y: 0.3, w: 4.5, h: 0.3, align: "right", fontFace: BODY, fontSize: 10, color: MUTE, margin: 0 });
}
function footer(s, n) {
  s.addText("平行運算與軟硬體共同演化", { x: MX, y: FOOT_Y, w: 8, h: 0.3, fontFace: BODY, fontSize: 9, color: FOOTC, margin: 0 });
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
function laneGrid(s, x, y, cols, rows, cell, gap, color) {
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++)
    s.addShape(pres.shapes.RECTANGLE, { x: x + c * (cell + gap), y: y + r * (cell + gap), w: cell, h: cell, fill: { color }, line: { type: "none" } });
}

// 1 — 標題
(() => {
  const s = pres.addSlide(); base(s);
  // 右側 glyph：雙向箭頭迴圈（硬體 ⇄ 模型）
  card(s, 9.0, 2.0, 3.5, 1.0, MEMTINT);
  s.addText("硬體", { x: 9.0, y: 2.0, w: 3.5, h: 1.0, align: "center", valign: "middle", fontFace: HEAD, fontSize: 20, bold: true, color: MEM, margin: 0 });
  card(s, 9.0, 3.9, 3.5, 1.0, COMPTINT);
  s.addText("模型", { x: 9.0, y: 3.9, w: 3.5, h: 1.0, align: "center", valign: "middle", fontFace: HEAD, fontSize: 20, bold: true, color: COMP, margin: 0 });
  s.addShape(pres.shapes.LINE, { x: 9.6, y: 3.05, w: 0, h: 0.8, line: { color: MEM, width: 2.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.LINE, { x: 11.9, y: 3.05, w: 0, h: 0.8, flipV: true, line: { color: COMP, width: 2.5, endArrowType: "triangle" } });
  s.addText("hardware lottery <-> tensor core", { x: 8.7, y: 5.1, w: 4.2, h: 0.3, align: "center", fontFace: MONO, fontSize: 10, color: MUTE, margin: 0 });

  s.addText("GPU 記憶體與資料搬遷讀書會  ·  S5 / 番外進階場", { x: MX, y: 1.7, w: 9, h: 0.4, fontFace: BODY, fontSize: 15, color: MEM, bold: true, charSpacing: 1, margin: 0 });
  s.addText([{ text: "平行運算與軟硬體共同演化", options: { breakLine: true } }, { text: "從 CNN 到 Transformer 到混合架構", options: {} }],
    { x: MX, y: 2.45, w: 8.3, h: 2.0, fontFace: HEAD, fontSize: 40, bold: true, color: INK, lineSpacingMultiple: 1.06, margin: 0 });
  s.addText("GPU 不是「快」，是「寬」——而過去 15 年的模型史，就是一部遷就與改造硬體的歷史。", { x: MX, y: 4.8, w: 8.2, h: 0.6, fontFace: BODY, fontSize: 17, color: MUTE, margin: 0 });
  s.addText("用 S1–S4 的工具（roofline、記憶體階層、prefill/decode）讀懂模型演化。", { x: MX, y: 5.55, w: 8.2, h: 0.4, fontFace: BODY, fontSize: 13, color: FOOTC, margin: 0 });
})();

// 2 — 熱身：GPU 單執行緒比 CPU 慢
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "01", "熱身：GPU 的單執行緒，比你的筆電還慢", WARN);

  const cw = 5.7, ch = 3.5;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 1.95, w: cw, h: 0.12, fill: { color: COMP }, line: { type: "none" } });
  s.addText("CPU 核心：為單執行緒而生", { x: MX + 0.35, y: 2.25, w: cw - 0.7, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: COMP, margin: 0 });
  s.addText([
    { text: "~4–5 GHz、亂序執行、分支預測", options: { bullet: true, breakLine: true } },
    { text: "大 cache 把延遲藏在硬體裡", options: { bullet: true, breakLine: true } },
    { text: "電晶體花在「讓一條 thread 快」", options: { bullet: true } },
  ], { x: MX + 0.35, y: 2.85, w: cw - 0.7, h: 2.3, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 8, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: 1.95, w: cw, h: 0.12, fill: { color: MEM }, line: { type: "none" } });
  s.addText("GPU 核心：單條 thread 又慢又笨", { x: x2 + 0.35, y: 2.25, w: cw - 0.7, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: MEM, margin: 0 });
  s.addText([
    { text: "~1.5–2 GHz、循序執行、沒有分支預測", options: { bullet: true, breakLine: true } },
    { text: "靠「大量 warp 互相掩護」藏延遲（S2）", options: { bullet: true, breakLine: true } },
    { text: "電晶體全部換成更多運算單元", options: { bullet: true } },
  ], { x: x2 + 0.35, y: 2.85, w: cw - 0.7, h: 2.3, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 8, margin: 0 });

  s.addText("GPU 不是快，是寬。沒有平行度的工作丟上 GPU = 用上萬個車道跑一台車。", { x: MX, y: 5.95, w: 11.9, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: WARN, margin: 0 });
  footer(s, 2);
})();

// 3 — 吞吐機器數字感 + Amdahl
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "02", "餵飽一張卡要多少平行度？Amdahl 給天花板", MEM);

  card(s, MX, 1.95, 5.9, 3.7, BG2);
  s.addText("H100 SXM 的「寬」", { x: MX + 0.35, y: 2.2, w: 5.2, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: MEM, margin: 0 });
  laneGrid(s, MX + 0.45, 2.75, 12, 4, 0.3, 0.1, MEM);
  s.addText([
    { text: "132 SM × 128 lane ≈ 16,896 條 lane + 528 tensor core", options: { breakLine: true } },
    { text: "每條 lane 還要數十個 warp 待命藏延遲（S2）", options: { breakLine: true } },
    { text: "→ 同時要有 10⁵ 量級的 thread 在飛", options: { color: MEM, bold: true } },
  ], { x: MX + 0.35, y: 4.45, w: 5.3, h: 1.1, fontFace: BODY, fontSize: 13, color: INK, lineSpacingMultiple: 1.25, margin: 0 });

  const x2 = MX + 5.9 + 0.5;
  card(s, x2, 1.95, 5.5, 3.7, BG2);
  s.addText("Amdahl's Law", { x: x2 + 0.35, y: 2.2, w: 4.8, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: COMP, margin: 0 });
  s.addText("加速上限 = 1 / ((1−p) + p/N)", { x: x2 + 0.35, y: 2.7, w: 4.8, h: 0.45, fontFace: MONO, fontSize: 16, bold: true, color: INK, fill: { color: "0A1322" }, valign: "middle", margin: 8 });
  s.addText([
    { text: "p = 可平行比例，N = lane 數", options: { breakLine: true, color: MUTE } },
    { text: "p = 95%、N → ∞：最多也只有 20×", options: { breakLine: true, color: WARN, bold: true } },
    { text: "N 夠大之後，瓶頸只剩序列的 (1−p)", options: {} },
  ], { x: x2 + 0.35, y: 3.35, w: 4.8, h: 1.6, fontFace: BODY, fontSize: 14.5, color: INK, lineSpacingMultiple: 1.3, paraSpaceAfter: 8, margin: 0 });

  s.addText("模型裡任何「序列相依」的部分，都是 p 的天花板——本場的鑰匙句。", { x: MX, y: 5.95, w: 11.9, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: COMP, margin: 0 });
  footer(s, 3);
})();

// 4 — 平行軸
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "03", "平行度不是硬體給的，是模型「暴露」出來的", COMP);

  const axes = [
    ["batch", "樣本之間天然獨立", MEM],
    ["pixel / 空間", "卷積對每個位置獨立", MEM],
    ["channel / head", "特徵維度切開算", MEM],
    ["token / 時間步", "RNN：序列相依 ✗　attention：獨立 ✓", COMP],
    ["layer (pipeline)", "層間相依，只能流水線", WARN],
  ];
  axes.forEach((a, i) => {
    const y = 1.95 + i * 0.78;
    card(s, MX, y, 3.2, 0.62, BG3);
    s.addText(a[0], { x: MX + 0.25, y, w: 2.9, h: 0.62, valign: "middle", fontFace: MONO, fontSize: 14, bold: true, color: a[2], margin: 0 });
    s.addText(a[1], { x: MX + 3.5, y, w: 8.4, h: 0.62, valign: "middle", fontFace: BODY, fontSize: 14.5, color: INK, margin: 0 });
  });

  s.addText("模型設計 = 決定把計算依賴圖畫成「深的鏈」還是「寬的樹」。同一個任務，RNN 在 token 軸暴露 B×H，attention 暴露 B×T×H。", { x: MX, y: 6.0, w: 11.9, h: 0.6, fontFace: BODY, fontSize: 15, color: MEM, lineSpacingMultiple: 1.2, margin: 0 });
  footer(s, 4);
})();

// 5 — 硬體→模型（上）：CNN 與 hardware lottery
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "04", "硬體 → 模型（上）：CNN 等了 23 年的不是想法", MEM);

  // 時間軸
  const ty = 2.3;
  s.addShape(pres.shapes.LINE, { x: MX + 0.3, y: ty + 0.5, w: 10.8, h: 0, line: { color: LINE, width: 2 } });
  const pts = [
    [MX + 0.6, "1989", "LeCun ConvNet", "想法已存在，CPU 訓不大", MUTE],
    [MX + 5.0, "2012", "AlexNet ×2 GTX 580", "遊戲卡訓一週，ImageNet 拉開 10pp", COMP],
    [MX + 9.4, "之後", "VGG / ResNet…", "捲積=GEMM，吃滿 GPU", MEM],
  ];
  pts.forEach((p) => {
    s.addShape(pres.shapes.OVAL, { x: p[0], y: ty + 0.38, w: 0.24, h: 0.24, fill: { color: p[4] }, line: { color: BG, width: 1 } });
    s.addText(p[1], { x: p[0] - 0.8, y: ty - 0.15, w: 1.9, h: 0.35, align: "center", fontFace: MONO, fontSize: 14, bold: true, color: p[4], margin: 0 });
    s.addText(p[2], { x: p[0] - 1.5, y: ty + 0.8, w: 3.3, h: 0.35, align: "center", fontFace: HEAD, fontSize: 13.5, bold: true, color: INK, margin: 0 });
    s.addText(p[3], { x: p[0] - 1.5, y: ty + 1.15, w: 3.3, h: 0.6, align: "center", fontFace: BODY, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.1, margin: 0 });
  });

  card(s, MX, 4.4, 11.9, 1.35, BG2);
  s.addText([
    { text: "為什麼 CNN 跟 GPU 一拍即合：", options: { bold: true, color: MEM } },
    { text: "捲積經 im2col 攤平就是大 GEMM；pixel / channel / batch 全平行；權重重複用 → AI 高 → compute-bound。", options: { color: INK } },
  ], { x: MX + 0.35, y: 4.4, w: 11.2, h: 1.35, valign: "middle", fontFace: BODY, fontSize: 15, lineSpacingMultiple: 1.25, margin: 0 });

  s.addText("Hardware lottery（Sara Hooker）：不是最好的想法贏，是最合當代硬體的想法贏。", { x: MX, y: 6.0, w: 11.9, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: COMP, margin: 0 });
  footer(s, 5);
})();

// 6 — 硬體→模型（下）：Transformer 為平行而生
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "05", "硬體 → 模型（下）：Transformer 為平行而生", COMP);

  const cw = 5.7, ch = 3.4;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addText("RNN 訓練：T 步的鏈", { x: MX + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: WARN, margin: 0 });
  // 鏈：4 個方塊串列
  for (let i = 0; i < 4; i++) {
    s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.45 + i * 1.25, y: 2.85, w: 0.8, h: 0.6, fill: { color: WARNTINT }, line: { color: WARN, width: 1 } });
    s.addText(`h${i}`, { x: MX + 0.45 + i * 1.25, y: 2.85, w: 0.8, h: 0.6, align: "center", valign: "middle", fontFace: MONO, fontSize: 12, color: WARN, margin: 0 });
    if (i < 3) s.addShape(pres.shapes.LINE, { x: MX + 1.27 + i * 1.25, y: 3.15, w: 0.4, h: 0, line: { color: WARN, width: 2, endArrowType: "triangle" } });
  }
  s.addText("h_t 依賴 h_{t−1} → token 軸只能一步步來；暴露平行度 B×H。", { x: MX + 0.35, y: 3.75, w: cw - 0.7, h: 1.3, fontFace: BODY, fontSize: 14, color: INK, lineSpacingMultiple: 1.25, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addText("Attention：整段一次 matmul", { x: x2 + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: GOOD, margin: 0 });
  laneGrid(s, x2 + 0.5, 2.85, 8, 2, 0.42, 0.14, GOOD);
  s.addText("整段序列攤成 (B·T)×H 的大 GEMM → token 軸從鏈變寬；暴露平行度 B×T×H。", { x: x2 + 0.35, y: 4.1, w: cw - 0.7, h: 1.0, fontFace: BODY, fontSize: 14, color: INK, lineSpacingMultiple: 1.25, margin: 0 });

  s.addText([
    { text: "論文自己說的（2017）：RNN 的序列本質 “precludes parallelization”；Transformer “allowing significantly more parallelization”。", options: { breakLine: true, color: MUTE, italic: true } },
    { text: "Transformer 贏不是因為 FLOPs 少（O(T²) 其實更多），是把計算重排成 GPU 吃得下的形狀。", options: { color: COMP, bold: true } },
  ], { x: MX, y: 5.6, w: 11.9, h: 1.0, fontFace: BODY, fontSize: 14.5, lineSpacingMultiple: 1.25, margin: 0 });
  footer(s, 6);
})();

// 7 — 模型→硬體
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "06", "模型 → 硬體：箭頭反過來", MEM);

  const rows = [
    ["Tensor core（Volta, 2017）", "DL 負載九成是 GEMM → 做專用矩陣乘加單元"],
    ["TPU systolic array", "整顆晶片就是一台矩陣乘法機"],
    ["H100「Transformer Engine」", "硬體功能直接用模型命名；fp8 為 transformer 訓練而做"],
    ["H200：HBM 加到 141 GB", "不是算力不夠，是 KV cache（S3）吃容量+頻寬"],
    ["精度 fp32→fp16→bf16→fp8→fp4", "模型端證明訓得動 ⇄ 硬體端做出來，一步步往返"],
  ];
  rows.forEach((r, i) => {
    const y = 1.95 + i * 0.82;
    card(s, MX, y, 4.6, 0.66, MEMTINT);
    s.addText(r[0], { x: MX + 0.25, y, w: 4.2, h: 0.66, valign: "middle", fontFace: HEAD, fontSize: 13.5, bold: true, color: MEM, margin: 0 });
    s.addText(r[1], { x: MX + 4.95, y, w: 7.5, h: 0.66, valign: "middle", fontFace: BODY, fontSize: 14, color: INK, margin: 0 });
  });

  s.addText("這是一個迴圈，不是單向因果：硬體決定哪些模型贏，贏的模型再回頭改造硬體。", { x: MX, y: 6.25, w: 11.9, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: COMP, margin: 0 });
  footer(s, 7);
})();

// 8 — Case study CNN：FLOPs ≠ 速度
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "07", "Case study CNN：FLOPs ≠ 速度", COMP);

  const cw = 5.7, ch = 3.6;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addText("MobileNet：depthwise separable", { x: MX + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 16.5, bold: true, color: WARN, margin: 0 });
  s.addText([
    { text: "紙上 FLOPs ÷ 8–9（9C² → 9C + C²）", options: { bullet: true, breakLine: true } },
    { text: "但 depthwise 每 byte 只做幾次運算 → AI 個位數、memory-bound", options: { bullet: true, breakLine: true } },
    { text: "channel 軸被拆散 → 餵不滿 tensor core", options: { bullet: true, breakLine: true } },
    { text: "它的目標硬體本來就是手機 CPU", options: { bullet: true, color: MUTE } },
  ], { x: MX + 0.35, y: 2.7, w: cw - 0.7, h: 2.6, fontFace: BODY, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 7, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addText("ConvNeXt（2022）：反向操作", { x: x2 + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 16.5, bold: true, color: GOOD, margin: 0 });
  s.addText([
    { text: "用 transformer 時代配方重設計 ResNet", options: { bullet: true, breakLine: true } },
    { text: "敢用 7×7 大 kernel：FLOPs 多但平行度好、AI 高", options: { bullet: true, breakLine: true } },
    { text: "GPU 上「多而齊」常勝過「少而碎」", options: { bullet: true, color: GOOD, bold: true } },
  ], { x: x2 + 0.35, y: 2.7, w: cw - 0.7, h: 2.2, fontFace: BODY, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 7, margin: 0 });

  s.addText("教訓：紙上 FLOPs 是 CPU 思維；GPU 上要問 bytes 與平行度（demo 直接量這件事）。", { x: MX, y: 5.95, w: 11.9, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: COMP, margin: 0 });
  footer(s, 8);
})();

// 9 — Case study Transformer：FlashAttention / GQA
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "08", "Case study Transformer：演算法遷就記憶體階層", MEM);

  const cw = 5.7, ch = 3.7;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 1.95, w: cw, h: 0.12, fill: { color: MEM }, line: { type: "none" } });
  s.addText("FlashAttention（2022）", { x: MX + 0.35, y: 2.25, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: MEM, margin: 0 });
  s.addText([
    { text: "數學完全不變（exact attention）", options: { bullet: true, breakLine: true } },
    { text: "tiling + 線上 softmax → T×T 矩陣從不落地 HBM（S2 的招）", options: { bullet: true, breakLine: true } },
    { text: "同樣 FLOPs、bytes 大減 → 快 2–4×、記憶體 O(T²)→O(T)", options: { bullet: true, color: MEM, bold: true } },
    { text: "改的不是數學，是資料的「住址」", options: { bullet: true, color: MUTE } },
  ], { x: MX + 0.35, y: 2.75, w: cw - 0.7, h: 2.7, fontFace: BODY, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 7, margin: 0 });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: x2, y: 1.95, w: cw, h: 0.12, fill: { color: COMP }, line: { type: "none" } });
  s.addText("MQA / GQA", { x: x2 + 0.35, y: 2.25, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: COMP, margin: 0 });
  s.addText([
    { text: "S3：decode 每 token 要搬整份 KV cache", options: { bullet: true, breakLine: true } },
    { text: "乾脆改模型：多個 Q head 共用一組 KV head", options: { bullet: true, breakLine: true } },
    { text: "Llama 2 70B：64 Q head 共用 8 組 KV → KV bytes ÷8", options: { bullet: true, color: COMP, bold: true } },
    { text: "為了頻寬連架構都改，接受輕微品質代價", options: { bullet: true, color: MUTE } },
  ], { x: x2 + 0.35, y: 2.75, w: cw - 0.7, h: 2.7, fontFace: BODY, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.2, paraSpaceAfter: 7, margin: 0 });

  footer(s, 9);
})();

// 10 — Case study 混合架構
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "09", "Case study 混合架構：一個模型、兩種硬體人格", GOOD);

  card(s, MX, 1.95, 11.9, 1.9, BG2);
  s.addText("Mamba / SSM（2023）：雙目標共同設計", { x: MX + 0.35, y: 2.15, w: 11.2, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: GOOD, margin: 0 });
  s.addText([
    { text: "訓練時：parallel scan 攤平 T 軸 → 平行、compute-bound、吃滿 GPU　", options: { color: INK } },
    { text: "｜　", options: { color: FOOTC } },
    { text: "推論時：退回遞迴、每 token O(1) 狀態 → 沒有越長越肥的 KV cache", options: { color: INK } },
  ], { x: MX + 0.35, y: 2.65, w: 11.2, h: 0.9, fontFace: BODY, fontSize: 14.5, lineSpacingMultiple: 1.25, margin: 0 });

  card(s, MX, 4.1, 11.9, 1.5, BG3);
  s.addText([
    { text: "但固定狀態 = 有損記憶，精確長程檢索不如 attention → 2024 起主流是混血：", options: { breakLine: true, color: INK } },
    { text: "Jamba / Griffin（attention ⨯ SSM 交錯）；ASR 圈更早：Conformer = 卷積（局部）+ attention（全局），正是 S3 的 ASR encoder 主流。", options: { color: MEM } },
  ], { x: MX + 0.35, y: 4.1, w: 11.2, h: 1.5, valign: "middle", fontFace: BODY, fontSize: 14.5, lineSpacingMultiple: 1.3, margin: 0 });

  s.addText("架構設計越來越像：在硬體約束下解最佳化問題。", { x: MX, y: 6.0, w: 11.9, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: COMP, margin: 0 });
  footer(s, 10);
})();

// 11 — 收束框架：三個硬體問題
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "10", "設計／選模型前，先問三個硬體問題", COMP);

  const qs = [
    ["①", "平行度在哪個軸？有多寬？", "餵不餵得飽上萬條 lane（B / T / pixel / channel / head）", MEM],
    ["②", "每搬一個 byte 做幾次運算？", "AI 落在 roofline 哪一區（S1）", COMP],
    ["③", "序列相依的鏈有多長？", "Amdahl 的 (1−p)：訓練看 T 軸、推論看 decode 步數", WARN],
  ];
  qs.forEach((q, i) => {
    const y = 1.95 + i * 1.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 0.7, h: 0.7, rectRadius: 0.1, fill: { color: q[3] }, line: { type: "none" } });
    s.addText(q[0], { x: MX, y, w: 0.7, h: 0.7, align: "center", valign: "middle", fontFace: HEAD, fontSize: 22, bold: true, color: BG, margin: 0 });
    s.addText(q[1], { x: MX + 1.0, y: y - 0.02, w: 11.0, h: 0.45, fontFace: HEAD, fontSize: 18, bold: true, color: INK, margin: 0 });
    s.addText(q[2], { x: MX + 1.0, y: y + 0.4, w: 11.0, h: 0.35, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  });

  card(s, MX, 5.25, 11.9, 1.0, BG2);
  s.addText([
    { text: "用三問掃模型史：", options: { bold: true, color: COMP } },
    { text: "CNN（✓✓✓）→ RNN（✗鏈長）→ Transformer（訓練✓✓✓；decode ✗鏈長✗AI低）→ Flash/GQA（修 bytes）→ Mamba/混合（修鏈）", options: { color: INK } },
  ], { x: MX + 0.35, y: 5.25, w: 11.2, h: 1.0, valign: "middle", fontFace: BODY, fontSize: 13.5, lineSpacingMultiple: 1.2, margin: 0 });
  footer(s, 11);
})();

// 12 — Demo
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "11", "動手：flops_vs_parallelism demo", MEM);

  const cw = 5.7, ch = 3.9;
  card(s, MX, 1.95, cw, ch, BG2);
  s.addText("實驗 A：序列相依 vs 全平行", { x: MX + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: MEM, margin: 0 });
  const ta = [
    [{ text: "model", options: { color: MUTE, bold: true } }, { text: "GFLOPs", options: { color: MUTE, bold: true, align: "right" } }, { text: "ms", options: { color: MUTE, bold: true, align: "right" } }, { text: "TFLOPS", options: { color: MUTE, bold: true, align: "right" } }],
    [{ text: "LSTM" }, { text: "17", options: { align: "right" } }, { text: "47.7", options: { align: "right", color: WARN } }, { text: "0.36", options: { align: "right" } }],
    [{ text: "Transformer" }, { text: "30", options: { align: "right" } }, { text: "24.9", options: { align: "right", color: MEM } }, { text: "1.21", options: { align: "right" } }],
  ];
  s.addTable(ta, { x: MX + 0.35, y: 2.7, w: 5.0, colW: [1.9, 1.1, 1.0, 1.0], rowH: 0.5, fontFace: MONO, fontSize: 12, color: INK, valign: "middle", fill: { color: "0A1322" }, border: { type: "solid", color: LINE, pt: 1 } });
  s.addText("FLOPs 多 1.75×，反而快 1.9×（Apple M2 實測，CUDA 差距更大）", { x: MX + 0.35, y: 4.45, w: 5.0, h: 0.6, fontFace: BODY, fontSize: 12, color: MEM, lineSpacingMultiple: 1.15, margin: 0 });
  s.addText("$ python run.py", { x: MX + 0.35, y: 5.15, w: 5.0, h: 0.45, fontFace: MONO, fontSize: 12, color: MEM, fill: { color: "0A1322" }, valign: "middle", margin: 8 });

  const x2 = MX + cw + 0.5;
  card(s, x2, 1.95, cw, ch, BG2);
  s.addText("實驗 B：FLOPs ≠ 速度", { x: x2 + 0.35, y: 2.2, w: cw - 0.7, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: COMP, margin: 0 });
  const tb = [
    [{ text: "conv", options: { color: MUTE, bold: true } }, { text: "GFLOPs", options: { color: MUTE, bold: true, align: "right" } }, { text: "ms", options: { color: MUTE, bold: true, align: "right" } }, { text: "TFLOPS", options: { color: MUTE, bold: true, align: "right" } }],
    [{ text: "dense 3×3" }, { text: "118", options: { align: "right" } }, { text: "40.8", options: { align: "right" } }, { text: "2.90", options: { align: "right", color: GOOD } }],
    [{ text: "depthwise sep" }, { text: "14", options: { align: "right" } }, { text: "11.0", options: { align: "right" } }, { text: "1.23", options: { align: "right", color: WARN } }],
  ];
  s.addTable(tb, { x: x2 + 0.35, y: 2.7, w: 5.0, colW: [1.9, 1.1, 1.0, 1.0], rowH: 0.5, fontFace: MONO, fontSize: 12, color: INK, valign: "middle", fill: { color: "0A1322" }, border: { type: "solid", color: LINE, pt: 1 } });
  s.addText("FLOPs ÷8.7，時間只 ÷3.7；達成 TFLOPS 掉 2.4×——省下的 FLOPs 被頻寬吃掉", { x: x2 + 0.35, y: 4.45, w: 5.0, h: 0.6, fontFace: BODY, fontSize: 12, color: COMP, lineSpacingMultiple: 1.15, margin: 0 });

  s.addText("兩個實驗合起來 = 「FLOPs 不是速度」的量化版。", { x: MX, y: 6.05, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 15, color: MEM, margin: 0 });
  footer(s, 12);
})();

// 13 — 收束
(() => {
  const s = pres.addSlide(); base(s);
  s.addText("本場帶走三件事", { x: MX, y: 0.8, w: 11.9, h: 0.7, fontFace: HEAD, fontSize: 32, bold: true, color: INK, margin: 0 });
  const items = [
    { n: "1", t: "GPU 不是快，是寬——平行度就是一切。", d: "單執行緒比 CPU 慢；序列相依的鏈（Amdahl 的 1−p）是加速天花板。", c: MEM },
    { n: "2", t: "模型與硬體是雙向共同演化的迴圈。", d: "硬體決定哪些想法贏（hardware lottery）；贏的模型回頭改造硬體（tensor core、Transformer Engine）。", c: COMP },
    { n: "3", t: "看新架構先問三題：平行軸？AI？鏈多長？", d: "CNN → Transformer → Flash/GQA → Mamba/混合，就是輪流在這三題上補洞。", c: GOOD },
  ];
  items.forEach((it, i) => {
    const y = 1.9 + i * 1.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 0.7, h: 0.7, rectRadius: 0.1, fill: { color: it.c }, line: { type: "none" } });
    s.addText(it.n, { x: MX, y, w: 0.7, h: 0.7, align: "center", valign: "middle", fontFace: MONO, fontSize: 26, bold: true, color: BG, margin: 0 });
    s.addText(it.t, { x: MX + 1.0, y: y - 0.05, w: 11.0, h: 0.5, fontFace: HEAD, fontSize: 19, bold: true, color: INK, margin: 0 });
    s.addText(it.d, { x: MX + 1.0, y: y + 0.42, w: 11.0, h: 0.4, fontFace: BODY, fontSize: 13.5, color: MUTE, margin: 0 });
  });
  card(s, MX, 5.6, 11.9, 1.1, BG2);
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 5.6, w: 0.12, h: 1.1, fill: { color: GOOD }, line: { type: "none" } });
  s.addText([{ text: "系列回顧　", options: { color: GOOD, bold: true } }, { text: "S1–S4 給了「資料搬遷」的顯微鏡，S5 用它讀懂模型演化史。下次看到新架構，先問那三個硬體問題。", options: { color: INK } }],
    { x: MX + 0.4, y: 5.6, w: 11.3, h: 1.1, valign: "middle", fontFace: BODY, fontSize: 15, margin: 0 });
})();

pres.writeFile({ fileName: "../s5_parallelism_codesign.pptx" }).then((f) => console.log("written:", f));
