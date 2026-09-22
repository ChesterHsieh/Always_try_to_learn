// 第四堂課（最終章）— 從模型到機櫃：一個字，穿過一整排機櫃
// 產生 ../class4_models_to_racks.html。合併原第四（多機）、第五（中國開源模型）、第六（機櫃）三堂。
//
// 前半 = 先認識要搬上機櫃的模型：decode 與 MoE 的基本動作，再用五個旋鈕過一遍中國開源模型的思路。
// 後半 = 跟著一個請求穿過 SGLang 96×H100（prefill 4 台 EP32、decode 9 台 EP72），
//        按「多常搬 × 一次多大」把資料分四種，合回來指出真瓶頸＝一個 scale-up 域裝得下幾張卡。
// 主軸 = 拆散變便宜的前提：最常搬的資料，走最快的路。
// 逐頁事實原子與數字來源見 ../../notes/class4_models_to_racks.md
const pptxgen = require("./pptx-html");

const BG = "0E1726", BG2 = "16233A", BG3 = "1C2E4A";
const INK = "EAF1FB", MUTE = "8FA6C4", LINE = "2A3D5C", FOOTC = "5C7299";
const MEM = "38BDF8", COMP = "F59E0B", WARN = "FB7185", GOOD = "34D399", PURP = "A78BFA";
const MEMTINT = "10455F", COMPTINT = "4A3410", WARNTINT = "4A2433", GOODTINT = "123D31", PURPTINT = "2A2150";
const HEAD = "PingFang TC", BODY = "PingFang TC", MONO = "Menlo";

const W = 13.33, H = 7.5, MX = 0.7, TITLE_Y = 0.62, FOOT_Y = 7.05, TOTAL = 28;
const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 });

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "GPU 記憶體與資料搬遷讀書會";
pres.title = "第四堂課 · 從模型到機櫃";

let PAGE = 0;
const base = (s) => { s.background = { color: BG }; PAGE += 1; };
function runningHeader(s) {
  s.addText("讀書會 · 第四堂課（最終章）· 從模型到機櫃", { x: W - 5.9, y: 0.3, w: 5.2, h: 0.3, align: "right", fontFace: BODY, fontSize: 10, color: MUTE, margin: 0 });
}
function footer(s, part) {
  s.addText(part, { x: MX, y: FOOT_Y, w: 9.5, h: 0.3, fontFace: BODY, fontSize: 9, color: FOOTC, margin: 0 });
  s.addText(`${PAGE} / ${TOTAL}`, { x: W - 1.6, y: FOOT_Y, w: 0.9, h: 0.3, align: "right", fontFace: MONO, fontSize: 9, color: FOOTC, margin: 0 });
}
function header(s, num, title, accent) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: TITLE_Y, w: 0.62, h: 0.62, rectRadius: 0.08, fill: { color: accent }, line: { type: "none" }, shadow: shadow() });
  s.addText(num, { x: MX, y: TITLE_Y, w: 0.62, h: 0.62, align: "center", valign: "middle", fontFace: MONO, fontSize: 20, bold: true, color: BG, margin: 0 });
  s.addText(title, { x: MX + 0.85, y: TITLE_Y, w: W - MX - 0.85 - 0.5, h: 0.62, valign: "middle", fontFace: HEAD, fontSize: 23, bold: true, color: INK, margin: 0 });
}
function lede(s, text, y) {
  s.addText(text, { x: MX, y: y || 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
}
function card(s, x, y, w, h, fill, lineColor) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: lineColor || LINE, width: 1 }, shadow: shadow() });
}
function pill(s, x, y, w, h, text, edge, fill, txt, fs) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: h / 2, fill: { color: fill || BG2 }, line: { color: edge, width: 1 } });
  s.addText(text, { x, y, w, h, align: "center", valign: "middle", fontFace: BODY, fontSize: fs || 11, bold: true, color: txt || edge, margin: 0 });
}
function takeaway(s, text, color) {
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 6.02, w: 0.09, h: 0.62, fill: { color: color || MEM }, line: { type: "none" } });
  s.addText(text, { x: MX + 0.22, y: 6.0, w: 11.9, h: 0.66, fontFace: HEAD, fontSize: 14, bold: true, color: color || MEM, valign: "middle", margin: 0 });
}
function tableGrid(s, x, y, w, cols, rows, accent, fs, rh) {
  const RH = rh || 0.42, hh = 0.42;
  let cx = x;
  cols.forEach((c) => {
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y, w: c.w, h: hh, fill: { color: BG3 }, line: { color: LINE, width: 0.8 } });
    s.addText(c.t, { x: cx + 0.08, y, w: c.w - 0.16, h: hh, valign: "middle", fontFace: HEAD, fontSize: (fs || 11) + 0.5, bold: true, color: INK, margin: 0 });
    cx += c.w;
  });
  rows.forEach((r, ri) => {
    cx = x;
    r.forEach((cell, ci) => {
      s.addShape(pres.shapes.RECTANGLE, { x: cx, y: y + hh + ri * RH, w: cols[ci].w, h: RH, fill: { color: ri % 2 ? BG2 : BG }, line: { color: LINE, width: 0.8 } });
      s.addText(cell, { x: cx + 0.08, y: y + hh + ri * RH, w: cols[ci].w - 0.16, h: RH, valign: "middle", fontFace: ci === 0 ? HEAD : BODY, fontSize: fs || 11, bold: ci === 0, color: ci === 0 ? accent : MUTE, margin: 0 });
      cx += cols[ci].w;
    });
  });
}
// 水平長條：rows = [標籤, 值, 值文字, 顏色?]
function hbar(s, x, y, w, rows, accent, opt) {
  const o = opt || {};
  const rh = o.rh || 0.52, lw = o.lw || 4.2, vw = o.vw || 1.2;
  const max = o.max || Math.max(...rows.map((r) => r[1]));
  const track = w - lw - vw - 0.3;
  rows.forEach(([lab, v, vt, c], i) => {
    const yy = y + i * rh, col = c || accent;
    s.addText(lab, { x, y: yy, w: lw, h: rh - 0.06, align: "right", valign: "middle", fontFace: BODY, fontSize: o.fs || 11.5, color: MUTE, margin: 0 });
    s.addShape(pres.shapes.RECTANGLE, { x: x + lw + 0.15, y: yy + 0.07, w: track, h: rh - 0.22, fill: { color: BG }, line: { color: LINE, width: 0.8 } });
    s.addShape(pres.shapes.RECTANGLE, { x: x + lw + 0.15, y: yy + 0.07, w: Math.max(0.05, track * v / max), h: rh - 0.22, fill: { color: col }, line: { type: "none" } });
    s.addText(vt, { x: x + lw + 0.25 + track, y: yy, w: vw, h: rh - 0.06, valign: "middle", fontFace: MONO, fontSize: 11, bold: true, color: col, margin: 0 });
  });
}
// 堆疊長條：segs = [值, 顏色, 標籤]
function stackBar(s, x, y, w, h, segs, max, tot) {
  let cx = x;
  segs.forEach(([v, c, lab]) => {
    const bw = w * v / max;
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y, w: bw, h, fill: { color: c }, line: { type: "none" } });
    if (lab && bw > 0.8) s.addText(lab, { x: cx, y, w: bw, h, align: "center", valign: "middle", fontFace: MONO, fontSize: 10.5, bold: true, color: BG, margin: 0 });
    cx += bw;
  });
  if (tot) s.addText(tot, { x: cx + 0.12, y, w: 1.3, h, valign: "middle", fontFace: MONO, fontSize: 12, bold: true, color: INK, margin: 0 });
}
// 旅程進度條（第 16–23 頁）
const STATIONS = ["① router", "② prefill", "③ KV 交接", "④ decode 一步"];
function stationStepper(s, active, y) {
  const yy = y || 1.3, w = 2.87, gap = 0.14, h = 0.44;
  STATIONS.forEach((t, i) => {
    const on = i === active;
    const x = MX + i * (w + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: yy, w, h, rectRadius: 0.08, fill: { color: on ? MEM : BG2 }, line: { color: on ? MEM : LINE, width: 1 } });
    s.addText(t, { x, y: yy, w, h, align: "center", valign: "middle", fontFace: BODY, fontSize: 12, bold: on, color: on ? BG : MUTE, margin: 0 });
  });
}
function toolHint(s, text, y) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 11.9, h: 0.46, rectRadius: 0.08, fill: { color: PURPTINT }, line: { color: PURP, width: 1 } });
  s.addText("🎛  " + text, { x: MX + 0.2, y, w: 11.5, h: 0.46, valign: "middle", fontFace: BODY, fontSize: 11.5, color: PURP, margin: 0 });
}

const KNOBS = ["① 壓 KV", "② 少算", "③ 少看", "④ 一次多產", "⑤ 降精度"];
function knobStepper(s, active) {
  const y = 1.4, x0 = MX, w = 2.24, gap = 0.14, h = 0.46;
  KNOBS.forEach((b, i) => {
    const on = i === active || active === -1;
    const x = x0 + i * (w + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: on ? PURP : BG2 }, line: { color: on ? PURP : LINE, width: 1 } });
    s.addText(b, { x, y, w, h, align: "center", valign: "middle", fontFace: BODY, fontSize: 12, bold: on, color: on ? BG : MUTE, margin: 0 });
  });
}

const P0 = "讀書會 · 第四堂課（最終章）";
const P1 = "前半 · 先認識要搬上機櫃的模型";
const PA = "後半 · 拆法";
const PB = "後半 · 跟著一個請求走";
const PC = "後半 · 合回來：真瓶頸";
const PE = "收尾";

// ============================================================ 1 封面＋鉤子
(() => {
  const s = pres.addSlide(); base(s);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.16, h: H, fill: { color: MEM }, line: { type: "none" } });
  s.addText("第四堂課 · 最終章 · 從模型到機櫃", { x: MX + 0.3, y: 0.95, w: 9, h: 0.42, fontFace: MONO, fontSize: 15, color: MEM, margin: 0 });
  s.addText("你看到的每一個字，", { x: MX + 0.3, y: 1.45, w: 11.5, h: 0.8, fontFace: HEAD, fontSize: 38, bold: true, color: INK, margin: 0 });
  s.addText("背後有 72 張卡同時踏了一步", { x: MX + 0.3, y: 2.2, w: 11.5, h: 0.8, fontFace: HEAD, fontSize: 38, bold: true, color: INK, margin: 0 });
  s.addText("DeepSeek 的模型設計 × SGLang / vLLM 的多機服務 × NVIDIA 的機櫃", { x: MX + 0.3, y: 3.1, w: 11.5, h: 0.45, fontFace: HEAD, fontSize: 18, color: MUTE, margin: 0 });
  [["278 台", "DeepSeek 尖峰用來服務 V3/R1 的 8 卡機（平均 226.75 台）", COMP],
  ["12 台 · 96 張 H100", "SGLang 的開源復現：prefill 4 台（EP32）、decode 9 台（EP72）", MEM],
  ["116 次", "每產一個字的全員交換：58 層 MoE × dispatch + combine", PURP]]
    .forEach(([n, d, c], i) => {
      const x = MX + 0.3 + i * 3.83;
      card(s, x, 3.8, 3.65, 1.5, BG2, c);
      s.addText(n, { x: x + 0.22, y: 3.92, w: 3.2, h: 0.5, valign: "middle", fontFace: MONO, fontSize: 19, bold: true, color: c, margin: 0 });
      s.addText(d, { x: x + 0.22, y: 4.44, w: 3.25, h: 0.75, valign: "top", fontFace: BODY, fontSize: 11, color: MUTE, lineSpacingMultiple: 1.25, margin: 0 });
    });
  card(s, MX + 0.3, 5.55, 11.3, 0.85, BG2, COMP);
  s.addText([
    { text: "通訊明明變多了，為什麼拆散反而便宜？", options: { bold: true, color: COMP } },
    { text: "　前半先認識這個模型（decode、MoE、MLA），後半跟著一個請求穿過機櫃算帳。", options: { color: INK } },
  ], { x: MX + 0.55, y: 5.55, w: 10.8, h: 0.85, valign: "middle", fontFace: HEAD, fontSize: 14.5, margin: 0 });
  footer(s, P0);
})();

// ============================================================ 2 一個字怎麼生出來
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "01", "一個字是怎麼生出來的：prefill 一次吃完 prompt，decode 一次只吐一個字", MEM);
  lede(s, "第一堂的謎題換個角度重看：大語言模型回答你，其實是兩個性格完全不同的階段。");
  const flow = [["Prompt（整段）", FOOTC], ["Prefill", COMP], ["字 1", INK], ["Decode", MEM], ["字 2", INK], ["Decode", MEM], ["字 3 …", INK]];
  flow.forEach(([t, c], i) => {
    const x = MX + i * 1.72;
    pill(s, x, 1.85, 1.5, 0.48, t, c, c === INK ? BG3 : BG2, c, 11.5);
    if (i < flow.length - 1) s.addText("▶", { x: x + 1.52, y: 1.9, w: 0.2, h: 0.38, fontFace: BODY, fontSize: 11, color: FOOTC, margin: 0 });
  });
  s.addText("自回歸：字 2 要等字 1 出來才能開始算——decode 無法在時間軸上平行", { x: MX, y: 2.38, w: 11.9, h: 0.3, fontFace: BODY, fontSize: 10.5, color: FOOTC, margin: 0 });

  [["Prefill（一次）", ["整段 prompt 的 T 個 token 同時進模型", "大矩陣乘法，算術強度高 → compute-bound", "副產品：每層的 K、V 存起來 ＝ KV cache"], COMP, 0],
  ["Decode（每個字一次）", ["每步只餵上一個字，產出下一個字", "讀一次整份權重 + 這條序列的全部 KV，只算 1 個 token", "AI ≈ 1 → memory-bound：時間花在搬，不在算"], MEM, 1]]
    .forEach(([t, arr, c, k]) => {
      const x = MX + k * 6.05;
      card(s, x, 2.8, 5.85, 1.75, BG2, c);
      s.addText(t, { x: x + 0.22, y: 2.88, w: 5.4, h: 0.36, fontFace: HEAD, fontSize: 15, bold: true, color: c, margin: 0 });
      arr.forEach((d, i) => s.addText("· " + d, { x: x + 0.22, y: 3.3 + i * 0.37, w: 5.45, h: 0.34, valign: "middle", fontFace: BODY, fontSize: 11.5, color: i === 2 ? INK : MUTE, margin: 0 }));
    });

  s.addText("一層 Transformer 裡只做兩件事——後面所有技巧都是在改其中一件", { x: MX, y: 4.68, w: 11.9, h: 0.32, fontFace: HEAD, fontSize: 13, bold: true, color: INK, margin: 0 });
  [["Attention：回頭看過去所有字", "要讀 KV cache（隨對話變長而長大）→ 旋鈕① 壓 KV、③ 少看", PURP],
  ["FFN：每個字各自過一個大 MLP", "要讀權重（佔參數大半）→ 旋鈕② MoE 少算", COMP]]
    .forEach(([t, d, c], k) => {
      const x = MX + k * 6.05;
      s.addShape(pres.shapes.RECTANGLE, { x, y: 5.08, w: 5.85, h: 0.78, fill: { color: BG2 }, line: { color: c, width: 1 } });
      s.addText(t, { x: x + 0.2, y: 5.1, w: 5.5, h: 0.36, valign: "middle", fontFace: HEAD, fontSize: 12.5, bold: true, color: c, margin: 0 });
      s.addText(d, { x: x + 0.2, y: 5.45, w: 5.5, h: 0.36, valign: "middle", fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });
    });
  takeaway(s, "整堂課只追一個問題：decode 每吐一個字，要搬多少 bytes、從哪裡搬到哪裡？", MEM);
  footer(s, P1);
})();

// ============================================================ 3 五個旋鈕
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "02", "中國開源模型的思路：五個旋鈕，讓每個字少搬一點", PURP);
  knobStepper(s, -1);
  s.addText("第三堂是框架從外面調（把 batch 撐大＝動 roofline 的分子）；這些模型從裡面改：直接讓每產一個字要搬的 bytes 變少（動分母）。",
    { x: MX, y: 2.0, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  tableGrid(s, MX, 2.45, 11.9, [
    { t: "旋鈕", w: 2.1 }, { t: "打擊的瓶頸", w: 4.0 }, { t: "代表技術", w: 5.8 },
  ], [
    ["① 壓 KV", "decode 讀 KV 的頻寬 + 容量", "MHA → GQA → MLA（低秩 latent）→ Gated MLA"],
    ["② 少算", "每 token 的 FLOPs 與權重讀取", "MoE 稀疏化：細粒度專家、共享專家、極高稀疏比"],
    ["③ 少看", "長 context 的 O(n²) 與 KV 線性增長", "稀疏注意力（DSA/CSA/MSA）、線性注意力（Lightning/GDN/KDA）、混合層"],
    ["④ 一次多產", "單請求延遲（memory-bound 天花板）", "MTP 多 token 預測 ＋ 投機解碼"],
    ["⑤ 降精度", "搬的位元組數", "FP8 訓練、MXFP4 權重 / MXFP8 activation 的 QAT"],
  ], PURP, 11.5);
  takeaway(s, "出口管制下算力受限，效率不是加分項，是生存條件——所以成本從第一天就寫進架構裡。", PURP);
  footer(s, P1);
})();

// ============================================================ 4 旋鈕① 壓 KV
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "03", "旋鈕①：壓 KV —— MHA → GQA → MLA", MEM);
  knobStepper(s, 0);
  s.addText("KV cache 每 token 的大小，直接決定 decode 每步要搬多少位元組、以及同一顆 HBM 放得下幾條。",
    { x: MX, y: 2.05, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  const evo = [
    ["MHA", "每個 head 各存 K/V", "Llama 式 32 heads：512 KB/token", WARN],
    ["GQA", "多個 query head 共用一組 K/V", "Llama-3-8B（8 KV heads）：128 KB/token（÷4）", COMP],
    ["MLA", "K/V 投影成低秩 latent 再存，用時解回", "DeepSeek-V3：≈ 70 KB/token（同規模 MHA 推算 ~4 MB）", GOOD],
  ];
  evo.forEach(([t, d, n, c], i) => {
    const x = MX + i * 4.03;
    card(s, x, 2.5, 3.83, 2.2, BG2, c);
    s.addText(t, { x: x + 0.2, y: 2.62, w: 3.4, h: 0.42, fontFace: HEAD, fontSize: 20, bold: true, color: c, margin: 0 });
    s.addText(d, { x: x + 0.2, y: 3.1, w: 3.45, h: 0.7, valign: "top", fontFace: BODY, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
    s.addShape(pres.shapes.RECTANGLE, { x: x + 0.2, y: 3.88, w: 3.45, h: 0.65, fill: { color: c === WARN ? WARNTINT : c === COMP ? COMPTINT : GOODTINT }, line: { color: c, width: 1 } });
    s.addText(n, { x: x + 0.25, y: 3.88, w: 3.35, h: 0.65, align: "center", valign: "middle", fontFace: MONO, fontSize: 10, bold: true, color: c, margin: 0 });
    if (i < 2) s.addText("▶", { x: x + 3.85, y: 3.4, w: 0.2, h: 0.4, fontFace: BODY, fontSize: 14, color: FOOTC, margin: 0 });
  });
  card(s, MX, 4.95, 11.9, 0.9, BG2, MEM);
  s.addText("DeepSeek-V2 論文自陳：MLA 讓 KV cache 相對 MHA 減少 93.3%。Kimi K3 進一步用 Gated MLA，GLM-5 也採用 MLA——這個旋鈕已經是共識。",
    { x: MX + 0.28, y: 4.95, w: 11.3, h: 0.9, valign: "middle", fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  takeaway(s, "MLA 是為了 decode 的 HBM 頻寬而發明的——後半段第 19 頁會看到它順手買到的另一件事。", MEM);
  footer(s, P1);
})();

// ============================================================ 5 MoE 是什麼
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "04", "MoE：把一個大 FFN 換成 256 個小 FFN，每個字只挑 8 個走", COMP);
  lede(s, "Mixture of Experts 只改一層裡的 FFN。attention 照舊，FFN 從「一個大的」變成「一群小的＋一個挑人的 gate」。");

  card(s, MX, 1.85, 3.55, 2.75, BG2, FOOTC);
  s.addText("Dense FFN", { x: MX + 0.22, y: 1.95, w: 3.1, h: 0.36, fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.5, y: 2.45, w: 2.55, h: 1.1, fill: { color: BG3 }, line: { color: FOOTC, width: 1 } });
  s.addText("一個大 MLP", { x: MX + 0.5, y: 2.45, w: 2.55, h: 1.1, align: "center", valign: "middle", fontFace: MONO, fontSize: 12, color: MUTE, margin: 0 });
  s.addText("每個字都走完整個 FFN\n→ 每個字讀全部 FFN 權重", { x: MX + 0.22, y: 3.7, w: 3.1, h: 0.8, valign: "top", fontFace: BODY, fontSize: 11, color: MUTE, lineSpacingMultiple: 1.25, margin: 0 });
  s.addText("▶", { x: MX + 3.6, y: 2.95, w: 0.25, h: 0.4, fontFace: BODY, fontSize: 16, color: FOOTC, margin: 0 });

  const X = 4.6, Wd = 8.0;
  card(s, X, 1.85, Wd, 2.75, BG2, COMP);
  s.addText("MoE 層（DeepSeek-V3）", { x: X + 0.22, y: 1.95, w: 4, h: 0.36, fontFace: HEAD, fontSize: 15, bold: true, color: COMP, margin: 0 });
  [["① gate 替 256 個專家打分", PURP], ["② 取分數最高的 8 個", PURP], ["③ 8 個專家各算一份", COMP], ["④ 加權相加 ＋ shared", MEM]]
    .forEach(([t, c], i) => pill(s, X + 0.22 + i * 1.93, 2.4, 1.85, 0.42, t, c, BG3, c, 9.5));
  const sel = new Set([5, 37, 70, 101, 142, 177, 203, 250]);
  for (let i = 0; i < 256; i++) {
    const gx = X + 0.25 + (i % 32) * 0.185, gy = 3.0 + Math.floor(i / 32) * 0.16;
    const on = sel.has(i);
    s.addShape(pres.shapes.RECTANGLE, { x: gx, y: gy, w: 0.15, h: 0.12, fill: { color: on ? COMP : BG3 }, line: { color: on ? COMP : LINE, width: 0.5 } });
  }
  s.addShape(pres.shapes.RECTANGLE, { x: X + 6.3, y: 3.0, w: 1.45, h: 1.24, fill: { color: MEMTINT }, line: { color: MEM, width: 1 } });
  s.addText("shared\n專家\n每字必經", { x: X + 6.3, y: 3.0, w: 1.45, h: 1.24, align: "center", valign: "middle", fontFace: BODY, fontSize: 10.5, bold: true, color: MEM, margin: 0 });
  s.addText("256 個 routed 專家，這個字點亮了 8 個（橘色）", { x: X + 0.25, y: 4.3, w: 6, h: 0.26, fontFace: BODY, fontSize: 10, color: FOOTC, margin: 0 });

  card(s, MX, 4.75, 11.9, 1.1, BG2, PURP);
  s.addText("專家太多，一張卡放不下 → 分散到很多卡上（EP，專家平行）", { x: MX + 0.25, y: 4.8, w: 11.4, h: 0.36, valign: "middle", fontFace: HEAD, fontSize: 13, bold: true, color: PURP, margin: 0 });
  s.addText([
    { text: "dispatch", options: { bold: true, color: COMP } },
    { text: "：把這個字送到它選中的專家所在的卡　→　專家計算　→　", options: { color: MUTE } },
    { text: "combine", options: { bold: true, color: MEM } },
    { text: "：結果送回原卡。每一層 MoE 都要來回一次，所有卡互相交換（all-to-all）。", options: { color: MUTE } },
  ], { x: MX + 0.25, y: 5.18, w: 11.4, h: 0.6, valign: "middle", fontFace: BODY, fontSize: 11.5, margin: 0 });
  takeaway(s, "MoE 省的是每個字的計算與權重讀取；它不省容量，還多出通訊——這兩筆帳要到機櫃上才結。", COMP);
  footer(s, P1);
})();

// ============================================================ 6 旋鈕② 少算
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "05", "旋鈕②：少算 —— MoE 稀疏度一路往上推", COMP);
  knobStepper(s, 1);
  s.addText("決定 decode 速度的是「活躍參數 + KV」，不是總參數。所以趨勢是：總參數大幅變大、活躍參數只小幅變大。",
    { x: MX, y: 2.05, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  tableGrid(s, MX, 2.45, 11.9, [
    { t: "模型", w: 2.6 }, { t: "總參數 / 活躍", w: 2.6 }, { t: "專家配置", w: 3.2 }, { t: "活躍比例", w: 3.5 },
  ], [
    ["DeepSeek-V3", "671B / 37B", "256 routed + 1 shared，選 8", "5.5%　細粒度專家 + 共享專家"],
    ["Kimi K2", "1T / 32B", "384 專家，每 token 選 8", "3.2%"],
    ["Kimi K3", "2.8T / 104B", "896 專家，每 token 選 16", "3.7%　Stable LatentMoE"],
    ["Qwen 3.5", "397B / 17B", "極高稀疏", "4.3%"],
    ["GLM-5", "744B / 40B", "—", "5.4%"],
  ], COMP, 11.5);
  card(s, MX, 5.0, 5.8, 0.95, BG2, GOOD);
  s.addText("細粒度專家：切更小、選更多 → 組合數變多，表達力上升", { x: MX + 0.25, y: 5.0, w: 5.3, h: 0.95, valign: "middle", fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  card(s, 7.0, 5.0, 5.6, 0.95, BG2, PURP);
  s.addText("共享專家：每 token 必經，承接共通知識 → 讓 routed 專家專心學差異", { x: 7.25, y: 5.0, w: 5.1, h: 0.95, valign: "middle", fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  takeaway(s, "⚠️ 但 MoE 在單卡上不省容量（專家都得在 HBM）——要連容量也省，得靠後半段的大規模 EP（第 22 頁）。", COMP);
  footer(s, P1);
})();

// ============================================================ 7 旋鈕③ 少看
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "06", "旋鈕③：少看 —— 稀疏 vs 線性，兩條不同的路", GOOD);
  knobStepper(s, 2);
  s.addText("長 context 讓 attention 的 O(n²) 與 KV 的線性增長同時失控。兩條路的差別非常重要。",
    { x: MX, y: 2.05, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  card(s, MX, 2.5, 5.8, 2.9, BG2, GOOD);
  s.addText("稀疏注意力", { x: MX + 0.28, y: 2.62, w: 5.2, h: 0.4, fontFace: HEAD, fontSize: 19, bold: true, color: GOOD, margin: 0 });
  s.addText("保留完整 KV，但每個 query 只看 top-k 個位置", { x: MX + 0.28, y: 3.08, w: 5.2, h: 0.36, fontFace: BODY, fontSize: 12, color: INK, margin: 0 });
  ["DeepSeek DSA / CSA + HCA", "MiniMax MSA", "GLM-5 的 DSA 式稀疏"].forEach((t, i) =>
    s.addText("· " + t, { x: MX + 0.28, y: 3.5 + i * 0.34, w: 5.2, h: 0.32, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 }));
  pill(s, MX + 0.28, 4.62, 5.25, 0.55, "KV 還在 → prefix caching / 投機解碼還能用", GOOD, GOODTINT, GOOD, 11.5);

  card(s, 7.0, 2.5, 5.6, 2.9, BG2, PURP);
  s.addText("線性注意力", { x: 7.28, y: 2.62, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 19, bold: true, color: PURP, margin: 0 });
  s.addText("不存 KV，改成一個固定大小的遞迴狀態", { x: 7.28, y: 3.08, w: 5.0, h: 0.36, fontFace: BODY, fontSize: 12, color: INK, margin: 0 });
  ["MiniMax Lightning Attention（M1）", "Qwen Gated DeltaNet（3:1 混合）", "Kimi KDA（69 層線性 + 24 層 full）"].forEach((t, i) =>
    s.addText("· " + t, { x: 7.28, y: 3.5 + i * 0.34, w: 5.0, h: 0.32, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 }));
  pill(s, 7.28, 4.62, 5.05, 0.55, "KV 沒了 → 第三堂建好的三個系統全要重做", WARN, WARNTINT, WARN, 11.5);

  takeaway(s, "2026 的共識不是「線性取代 full attention」，而是「混合 + 稀疏」——下一頁看 MiniMax 為什麼這樣說。", GOOD);
  footer(s, P1);
})();

// ============================================================ 8 MiniMax 反例
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "07", "最有價值的反例：MiniMax M1 → M2 → M3", WARN);
  [["M1 · 2025-06", "Lightning Attention 混合（線性為主）+ CISPO RL", MEM],
  ["M2 · 2025-10", "退回 full attention，並公開說明為什麼", WARN],
  ["M3 · 2026", "改走 MSA 稀疏注意力：宣稱 1M ctx 下 prefill 9×、decode 15× 快於 M2", GOOD]]
    .forEach(([t, d, c], i) => {
      const x = MX + i * 4.03;
      card(s, x, 1.45, 3.85, 1.3, BG2, c);
      s.addText(t, { x: x + 0.18, y: 1.55, w: 3.5, h: 0.32, fontFace: MONO, fontSize: 12, bold: true, color: c, margin: 0 });
      s.addText(d, { x: x + 0.18, y: 1.9, w: 3.5, h: 0.75, valign: "top", fontFace: BODY, fontSize: 10.8, color: MUTE, lineSpacingMultiple: 1.25, margin: 0 });
      if (i < 2) s.addText("▶", { x: x + 3.87, y: 1.95, w: 0.16, h: 0.3, fontFace: BODY, fontSize: 13, color: FOOTC, margin: 0 });
    });
  s.addText("M2 為什麼退回？三個理由（LMSYS 部落格 + MiniMax 官方文件）", { x: MX, y: 2.92, w: 11.9, h: 0.38, fontFace: HEAD, fontSize: 15, bold: true, color: WARN, margin: 0 });
  [["① 評測會騙人", "混合注意力在 MMLU / LongBench 上看起來沒問題，放大後才發現多跳推理明顯退化；要在困難任務上得到顯著訊號，所需算力是天文數字。", WARN],
  ["② 理論 FLOPs ≠ 牆上時間", "線性注意力的實作本身就是 memory-bound，連訓練都吃不滿算力——第一堂 roofline 的教訓：省下的是紙上的 FLOPs。", COMP],
  ["③ 打壞第三堂的三個系統", "線性狀態對精度敏感、不能像 KV 切片複用 → KV 量化、prefix caching、投機解碼全部要重做。", PURP]]
    .forEach(([t, d, c], i) => {
      const y = 3.4 + i * 0.85;
      card(s, MX, y, 11.9, 0.75, BG2, c);
      s.addText(t, { x: MX + 0.22, y, w: 3.0, h: 0.75, valign: "middle", fontFace: HEAD, fontSize: 13.5, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 3.25, y, w: 8.45, h: 0.75, valign: "middle", fontFace: BODY, fontSize: 11, color: MUTE, lineSpacingMultiple: 1.2, margin: 0 });
    });
  takeaway(s, "「理論複雜度更低」離「生產環境更快」隔著三層：kernel 效率、評測有效性、生態相容性。", WARN);
  footer(s, P1);
})();

// ============================================================ 9 旋鈕④⑤
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "08", "旋鈕④ 一次多產、旋鈕⑤ 降精度", COMP);
  card(s, MX, 1.45, 11.9, 2.0, BG2, COMP);
  s.addText("④ MTP（Multi-Token Prediction）", { x: MX + 0.28, y: 1.58, w: 6, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: COMP, margin: 0 });
  s.addText("訓練時多預測幾步當額外訊號（更密的監督），推論時那些 head 直接當投機解碼的 draft。",
    { x: MX + 0.28, y: 2.02, w: 11.3, h: 0.36, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.28, y: 2.45, w: 11.3, h: 0.45, fill: { color: COMPTINT }, line: { color: COMP, width: 1 } });
  s.addText("驗證 k 個草稿 token：權重讀取 = 1 次（不變），FLOPs = k 倍　→　AI 從 1 變成 k", { x: MX + 0.28, y: 2.45, w: 11.3, h: 0.45, align: "center", valign: "middle", fontFace: MONO, fontSize: 12.5, bold: true, color: COMP, margin: 0 });
  s.addText("DeepSeek-V3 報告第二 token 接受率 ~85–90%；Qwen3-Next 也內建 MTP——訓練時就把 draft 模型長在自己身上。",
    { x: MX + 0.28, y: 2.95, w: 11.3, h: 0.42, valign: "middle", fontFace: BODY, fontSize: 11.8, color: MUTE, margin: 0 });

  card(s, MX, 3.6, 11.9, 2.25, BG2, MEM);
  s.addText("⑤ 降精度：從「部署後處理」變成「訓練的一部分」", { x: MX + 0.28, y: 3.73, w: 8, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: MEM, margin: 0 });
  tableGrid(s, MX + 0.28, 4.18, 11.3, [
    { t: "階段", w: 2.6 }, { t: "做法", w: 4.2 }, { t: "意義", w: 4.5 },
  ], [
    ["以前", "訓練用 bf16 → 社群事後量化成 GGUF/AWQ", "品質掉多少看運氣"],
    ["DeepSeek-V3", "FP8 訓練（首個大規模開源前沿模型）", "細粒度 scaling + 高精度累加解決數值問題"],
    ["Kimi K3", "MXFP4 權重 / MXFP8 activation，從 SFT 起 QAT", "出廠就是 4-bit，直接對齊 Blackwell FP4"],
  ], MEM, 10.5);
  footer(s, P1);
})();

// ============================================================ 10 全景對照
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "09", "全景：五個實驗室 × 五個旋鈕", PURP);
  tableGrid(s, MX, 1.45, 11.9, [
    { t: "", w: 2.1 }, { t: "① 壓 KV", w: 1.8 }, { t: "② 少算（MoE）", w: 2.5 }, { t: "③ 少看（attention）", w: 3.0 }, { t: "④ 一次多產", w: 1.4 }, { t: "⑤ 降精度", w: 1.1 },
  ], [
    ["DeepSeek", "MLA（−93.3%）", "671B-A37B 細粒度+共享", "DSA → CSA+HCA（1M：FLOPs 27%、KV 10%）", "MTP 85–90%", "FP8 訓練"],
    ["Kimi", "MLA → Gated MLA", "1T-A32B → 2.8T-A104B", "KDA 線性 ×69 + full ×24", "—", "MXFP4 QAT"],
    ["MiniMax", "—", "MoE", "Lightning → 退回 full → MSA 稀疏", "—", "—"],
    ["Qwen", "GQA", "397B-A17B 極高稀疏", "Gated DeltaNet : full ＝ 3:1", "MTP", "—"],
    ["GLM", "MLA", "744B-A40B", "DSA 式稀疏", "—", "—"],
  ], PURP, 10);
  s.addText("註：以各家技術報告／官方部落格公開數字為準；2026 上半年版本迭代極快，開講前請對一次官方頁面。",
    { x: MX, y: 4.6, w: 11.9, h: 0.35, fontFace: BODY, fontSize: 10.5, color: FOOTC, margin: 0 });
  card(s, MX, 5.05, 11.9, 1.35, BG2, PURP);
  s.addText("讀這張表的方法", { x: MX + 0.28, y: 5.14, w: 4, h: 0.32, fontFace: HEAD, fontSize: 13.5, bold: true, color: PURP, margin: 0 });
  s.addText("每一格都在回答同一個問題：「怎麼讓每產一個 token，少搬一點位元組？」——壓 KV 少搬 KV、MoE 少搬權重、稀疏注意力少搬歷史、MTP 把搬的成本攤給更多 token、量化讓每個數字更小。這正是第一堂 roofline 的分母。",
    { x: MX + 0.28, y: 5.5, w: 11.3, h: 0.8, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  footer(s, P1);
})();

// ============================================================ 11 開源零件
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "10", "為什麼他們連 kernel 都開源？", MEM);
  s.addText("這是前半與後半的接縫：這四個零件，等一下會在機櫃上一一出現。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  card(s, MX, 1.85, 11.9, 1.3, BG2, WARN);
  s.addText("困境：新架構如果沒有 kernel，就沒有人跑得動", { x: MX + 0.28, y: 1.97, w: 8, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: WARN, margin: 0 });
  s.addText("MLA 不是標準 attention，vLLM / SGLang 原本的 FlashAttention kernel 直接用不了。MoE 的 all-to-all、FP8 GEMM、專家負載均衡也都一樣——開源模型權重卻沒有配套 kernel，等於發布了一台沒有輪子的車。",
    { x: MX + 0.28, y: 2.35, w: 11.3, h: 0.7, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  const parts = [
    ["FlashMLA", "MLA 的 decode kernel", "讓 ① 壓 KV 真的跑得快", MEM],
    ["DeepEP", "MoE dispatch / combine 的通訊庫", "讓 ② 少算 在多機可行（第 17、20 頁）", COMP],
    ["DeepGEMM", "FP8 GEMM", "讓 ⑤ 降精度 吃到 tensor core（第 20 頁）", GOOD],
    ["EPLB", "專家平行負載均衡器", "熱門專家做副本（第 22–23 頁）", PURP],
  ];
  parts.forEach(([t, d, why, c], i) => {
    const y = 3.35 + i * 0.66;
    s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 11.9, h: 0.58, fill: { color: BG2 }, line: { color: c, width: 1 } });
    s.addText(t, { x: MX + 0.22, y, w: 2.0, h: 0.58, valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: c, margin: 0 });
    s.addText(d, { x: MX + 2.35, y, w: 4.2, h: 0.58, valign: "middle", fontFace: BODY, fontSize: 11.5, color: INK, margin: 0 });
    s.addText(why, { x: MX + 6.7, y, w: 5.0, h: 0.58, align: "right", valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
  });
  takeaway(s, "開源 kernel 是讓自家架構進入 vLLM / SGLang 生態的手段——模型與框架是共生的，不是上下游。", MEM);
  footer(s, P1);
})();

// ============================================================ 12 接縫：要搬上機櫃的模型
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "11", "等一下要搬上機櫃的，就是這個模型：DeepSeek-V3", MEM);
  lede(s, "前半段的每個設計決定，到了機櫃上都會變成一筆搬運帳。");
  tableGrid(s, MX, 1.8, 11.9, [
    { t: "規格", w: 2.2 }, { t: "數字", w: 3.6 }, { t: "前半段哪一頁", w: 2.1 }, { t: "到了機櫃上變成…", w: 4.0 },
  ], [
    ["層數", "61 層：前 3 層 dense、58 層 MoE", "第 5 頁", "每個字 58 × 2 ＝ 116 次交換"],
    ["專家", "256 routed + 1 shared，每字選 8", "第 5–6 頁", "decode 72 張卡，每卡只放 4 個"],
    ["參數", "671B 總 / 37B 活躍；hidden 7168", "旋鈕②", "權重 688.6 GB，一台 8×H100 放不下"],
    ["注意力", "MLA，≈ 70 KB / token", "旋鈕①", "一份 5K token 的 KV 跨機只要 7 ms"],
    ["精度", "FP8 訓練與權重", "旋鈕⑤", "dispatch 用 FP8、combine 用 BF16"],
  ], MEM, 11.5, 0.55);
  card(s, MX, 5.05, 11.9, 0.8, BG2, COMP);
  s.addText([
    { text: "671B 一台機器放不下，只能拆。", options: { bold: true, color: COMP } },
    { text: "但拆得越散、卡之間的交換越多——照常識應該越慢越貴。實測卻相反。", options: { color: MUTE } },
  ], { x: MX + 0.28, y: 5.05, w: 11.3, h: 0.8, valign: "middle", fontFace: BODY, fontSize: 13, margin: 0 });
  takeaway(s, "後半段：跟著一個請求穿過 12 台機器，看這些規格各自在哪一站付帳。", MEM);
  footer(s, P1);
})();

// ============================================================ 13 拆散反而更便宜
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "12", "把 DeepSeek 拆到越多卡上，每張卡反而產出越多 token", COMP);
  lede(s, "常識：「拆越散、通訊越多，一定越慢越貴。」但三組獨立的實測都指向相反方向。");
  hbar(s, MX, 1.85, 11.9, [
    ["基準：不拆（TP16 / EP8）", 1, "1×", FOOTC],
    ["SGLang：PD 分離 + EP72 vs TP16（decode）", 5.2, "5.2×", MEM],
    ["TRT-LLM：EP16/32 vs EP4/8（含 MTP）", 6.17, "6.17×", COMP],
    ["InferenceX：GB200 EP16 vs B200 EP8（同顆晶片）", 4.4, "4.4×", GOOD],
  ], MEM, { rh: 0.62, lw: 4.6, max: 6.5 });
  card(s, MX, 4.45, 11.9, 1.35, BG2, WARN);
  s.addText("⚠️ 三組條件不同，不能互相換算", { x: MX + 0.28, y: 4.55, w: 6, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: WARN, margin: 0 });
  s.addText("模型、精度、互動性目標、是否用 MTP 都不一樣（SGLang 是 R1 FP8、InferenceX 是 R1 FP4 @125 tok/s/user）。看的是方向，不是倍數本身。",
    { x: MX + 0.28, y: 4.95, w: 11.3, h: 0.75, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  takeaway(s, "拆散（EP 變大）→ 每卡產出變多。接下來要拆解：為什麼成立、代價是什麼、什麼時候不成立。", COMP);
  footer(s, PA);
})();

// ============================================================ 14 服務長什麼樣
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "13", "一個「模型服務」不是一台機器，是一個 router、兩個池子、三張網", MEM);
  lede(s, "本堂主角：SGLang 在 12 台 × 8 張 H100 上的開源復現（prefill 與 decode 在同一叢集上分開量測）。");

  card(s, MX, 1.85, 1.75, 1.1, BG2, INK);
  s.addText("Router", { x: MX, y: 1.95, w: 1.75, h: 0.35, align: "center", fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0 });
  s.addText("KV-aware\n選一組 P/D", { x: MX, y: 2.3, w: 1.75, h: 0.6, align: "center", valign: "top", fontFace: BODY, fontSize: 10.5, color: MUTE, margin: 0 });

  card(s, 2.75, 1.75, 3.4, 2.5, BG2, COMP);
  s.addText("Prefill 池 · EP32", { x: 2.95, y: 1.85, w: 3.0, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: COMP, margin: 0 });
  for (let i = 0; i < 4; i++) {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 2.95, y: 2.3 + i * 0.46, w: 3.0, h: 0.38, rectRadius: 0.06, fill: { color: COMPTINT }, line: { color: COMP, width: 1 } });
    s.addText(`node P${i + 1} · 8 × H100`, { x: 2.95, y: 2.3 + i * 0.46, w: 3.0, h: 0.38, align: "center", valign: "middle", fontFace: MONO, fontSize: 10.5, color: COMP, margin: 0 });
  }
  card(s, 6.5, 1.75, 6.1, 2.5, BG2, MEM);
  s.addText("Decode 池 · EP72", { x: 6.7, y: 1.85, w: 4.0, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: MEM, margin: 0 });
  for (let i = 0; i < 9; i++) {
    const x = 6.7 + (i % 3) * 1.93, y = 2.3 + Math.floor(i / 3) * 0.6;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 1.82, h: 0.5, rectRadius: 0.06, fill: { color: MEMTINT }, line: { color: MEM, width: 1 } });
    s.addText(`node D${i + 1}\n8 × H100`, { x, y, w: 1.82, h: 0.5, align: "center", valign: "middle", fontFace: MONO, fontSize: 9, color: MEM, margin: 0 });
  }
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 4.4, w: 11.9, h: 0.5, fill: { color: MEMTINT }, line: { color: MEM, width: 1 } });
  s.addText("算力網：跨機 RDMA · 每卡一張 400G ≈ 50 GB/s（每方向）", { x: MX, y: 4.4, w: 11.9, h: 0.5, align: "center", valign: "middle", fontFace: BODY, fontSize: 12, bold: true, color: MEM, margin: 0 });

  [["機內 NVLink 4", "每方向 450 GB/s · 8 卡一個 scale-up 域", COMP],
  ["跨機 RDMA 400G", "每方向 ≈ 50 GB/s · 慢一個數量級", MEM],
  ["前端乙太", "使用者 ↔ router：延遲 ms 級", FOOTC]]
    .forEach(([t, d, c], i) => {
      const x = MX + i * 4.03;
      card(s, x, 5.05, 3.85, 0.8, BG2, c);
      s.addText(t, { x: x + 0.18, y: 5.12, w: 3.5, h: 0.32, fontFace: HEAD, fontSize: 12.5, bold: true, color: c, margin: 0 });
      s.addText(d, { x: x + 0.18, y: 5.44, w: 3.5, h: 0.34, fontFace: BODY, fontSize: 10.5, color: MUTE, margin: 0 });
    });
  takeaway(s, "DeepSeek-V3 權重 688.6 GB——一台 8×80 GB 的 H100 連權重都放不下。跨機不是選項，是前提。", MEM);
  footer(s, PA);
})();

// ============================================================ 15 四種資料 × 四條路
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "14", "機櫃群＝多了三層的記憶體階層：每種資料走它付得起的路", PURP);
  lede(s, "第一堂：暫存器 → L1 → L2 → HBM → PCIe → SSD，每往外一層慢一個數量級。機櫃群只是再往外加三層。");
  [["scale-up 域", "450–900 GB/s", COMP], ["跨機 RDMA", "≈ 50 GB/s", MEM], ["前端乙太", "ms 級", FOOTC], ["儲存網", "啟動時才用", PURP]]
    .forEach(([t, n, c], i) => {
      const x = MX + i * 3.03;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.78, w: 2.7, h: 0.52, rectRadius: 0.08, fill: { color: BG2 }, line: { color: c, width: 1 } });
      s.addText(`${t}　${n}`, { x, y: 1.78, w: 2.7, h: 0.52, align: "center", valign: "middle", fontFace: BODY, fontSize: 11.5, bold: true, color: c, margin: 0 });
      if (i < 3) s.addText("▶", { x: x + 2.72, y: 1.86, w: 0.3, h: 0.35, fontFace: BODY, fontSize: 13, color: FOOTC, margin: 0 });
    });
  tableGrid(s, MX, 2.55, 11.9, [
    { t: "資料", w: 2.5 }, { t: "一次多大", w: 3.0 }, { t: "多常搬", w: 3.2 }, { t: "付得起的路", w: 3.2 },
  ], [
    ["① 請求本身", "KB 級", "每請求一次 + 每字回傳一次", "前端乙太就夠"],
    ["② KV 交接", "≈ 351 MB（MLA BF16 × 4,989 token）", "每請求一次", "跨機 RDMA"],
    ["③ MoE 交換", "每卡每層 ≈ 14 MB + 28 MB", "每字 116 次、72 張卡同步", "只有 scale-up 域或特化 RDMA"],
    ["④ 權重", "688.6 GB；單專家 FP8 ≈ 42 MiB", "幾乎不搬（啟動、EPLB 重排）", "儲存網 / RDMA"],
  ], PURP, 11.5, 0.62);
  takeaway(s, "推論引擎的全部工作，就是讓每一種資料只走它付得起的那一段路。這張表是本堂的地圖。", PURP);
  footer(s, PA);
})();

// ============================================================ 16 第一站 router
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "15", "進門那一跳只搬幾 KB，卻決定了一半的 prefill 要不要算", MEM);
  stationStepper(s, 0);
  [["SGLang", "sglang_router --pd-disaggregation", "挑一組 prefill／decode，塞入 bootstrap_host / port / room（隨機 63-bit ID），同一個請求同時 POST 給兩邊；串流從 decode 回來", MEM],
  ["vLLM / llm-d", "Gateway → Endpoint Picker（EPP）", "KV-aware 排程挑 pod，再由 sidecar 串起 prefill → NIXL → decode", COMP],
  ["NVIDIA Dynamo", "Smart Router", "KV-aware；Dynamo 1.0（2026-03-16）整合 SGLang / vLLM / llm-d / LMCache", PURP]]
    .forEach(([t, sub, d, c], i) => {
      const y = 2.0 + i * 1.25;
      card(s, MX, y, 11.9, 1.1, BG2, c);
      s.addText(t, { x: MX + 0.25, y: y + 0.1, w: 2.6, h: 0.42, valign: "middle", fontFace: HEAD, fontSize: 15.5, bold: true, color: c, margin: 0 });
      s.addText(sub, { x: MX + 2.9, y: y + 0.1, w: 8.7, h: 0.42, valign: "middle", fontFace: MONO, fontSize: 10.5, color: FOOTC, margin: 0 });
      s.addText(d, { x: MX + 0.25, y: y + 0.5, w: 11.3, h: 0.5, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
    });
  card(s, MX, 5.8, 11.9, 0.65, BG2, GOOD);
  s.addText([
    { text: "DeepSeek 線上 56.3% 的輸入 token 命中快取", options: { bold: true, color: GOOD } },
    { text: "（24 小時 608B 輸入裡有 342B）——router 把請求送到沒有快取的機器，這一半就得重算。", options: { color: MUTE } },
  ], { x: MX + 0.28, y: 5.8, w: 11.3, h: 0.65, valign: "middle", fontFace: BODY, fontSize: 12.5, margin: 0 });
  s.addText("兩難：越追快取命中，請求越容易擠在少數幾台機器上——局部性 vs 負載均衡", { x: MX, y: 6.6, w: 11.9, h: 0.3, fontFace: BODY, fontSize: 10.5, color: FOOTC, margin: 0 });
  footer(s, PB);
})();

// ============================================================ 17 第二站 prefill
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "16", "Prefill 是整批的大塊通訊，跨 InfiniBand 也吃得消", COMP);
  stationStepper(s, 1);
  lede(s, "prefill 吃整段 prompt → 大 GEMM → compute-bound（第一堂 roofline）；通訊是「一批 token 一次送」。", 1.9);
  tableGrid(s, MX, 2.3, 5.8, [
    { t: "輸入長度", w: 2.2 }, { t: "prefill 吞吐（每 node）", w: 3.6 },
  ], [
    ["1K token", "57,674 tok/s"],
    ["2K token", "54,543 tok/s"],
    ["4K token", "50,302 tok/s"],
  ], COMP, 11.5);
  s.addText("SGLang prefill 4 台（EP32）實測", { x: MX, y: 4.08, w: 5.8, h: 0.3, fontFace: BODY, fontSize: 10.5, color: FOOTC, margin: 0 });

  [["DeepEP normal 模式", "機內 NVLink 轉發 153 GB/s｜跨機 RDMA 43–58 GB/s。不支援 CUDA Graph——prefill 不需要，它一批就跑很久", COMP],
  ["Two-batch overlap", "一批切成兩個 micro-batch：一個在通訊、另一個在算 → prefill 吞吐 +27–35%", GOOD],
  ["每卡 9 個 routed 專家", "288 個專家槽（256 + 32 冗餘）÷ 32 卡", MEM]]
    .forEach(([t, d, c], i) => {
      const y = 2.3 + i * 1.12;
      card(s, 6.8, y, 5.8, 0.98, BG2, c);
      s.addText(t, { x: 7.0, y: y + 0.06, w: 5.4, h: 0.34, valign: "middle", fontFace: HEAD, fontSize: 13, bold: true, color: c, margin: 0 });
      s.addText(d, { x: 7.0, y: y + 0.38, w: 5.45, h: 0.55, valign: "top", fontFace: BODY, fontSize: 10.8, color: MUTE, lineSpacingMultiple: 1.2, margin: 0 });
    });
  takeaway(s, "大塊、低頻的通訊，跨機頻寬就吃得下——所以 prefill 放哪台機器，不是延遲的主要來源。", COMP);
  footer(s, PB);
})();

// ============================================================ 18 第三站 KV 交接流程
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "17", "KV 交接不是「傳過去」而已：SGLang 先預留再推，vLLM 先算完再拉", MEM);
  stationStepper(s, 2);
  const sg = ["router 同時送給 prefill 與 decode（同一個 bootstrap_room）",
    "decode 查 prefill 端的 bootstrap server、握手",
    "decode 先把 KV 槽位配好（PreallocQueue）",
    "prefill 算 forward（Waiting → Inflight）",
    "KV 經 RDMA 推到預留好的槽位（Mooncake / NIXL）",
    "decode 跳過 prefill forward，直接組 batch",
    "串流從 decode 回傳；逾時預設 300 s"];
  const vl = ["proxy／sidecar 先送 prefill（只產第一個 token）",
    "prefill 算 forward，KV 留在自己 GPU 上不釋放",
    "prefill 回傳 KV 的位址（block ids、engine id、host/port）",
    "proxy 把位址附在請求上，再轉給 decode",
    "首次接觸經 ZMQ side channel 交換 NIXL metadata",
    "decode 用單邊 RDMA read 直接「拉」走 KV",
    "decode 串流回傳；prefill 讀完或逾時才釋放 block"];
  [["SGLang · 推（push）", sg, MEM], ["vLLM + NIXL · 拉（pull）", vl, COMP]].forEach(([t, arr, c], k) => {
    const x = MX + k * 6.05;
    card(s, x, 1.95, 5.85, 3.5, BG2, c);
    s.addText(t, { x: x + 0.22, y: 2.05, w: 5.4, h: 0.36, fontFace: HEAD, fontSize: 15, bold: true, color: c, margin: 0 });
    arr.forEach((d, i) => {
      const y = 2.5 + i * 0.44;
      s.addShape(pres.shapes.OVAL, { x: x + 0.24, y: y + 0.04, w: 0.28, h: 0.28, fill: { color: i === 4 && k === 0 || i === 5 && k === 1 ? c : BG3 }, line: { color: c, width: 0.8 } });
      s.addText(String(i + 1), { x: x + 0.24, y: y + 0.04, w: 0.28, h: 0.28, align: "center", valign: "middle", fontFace: MONO, fontSize: 9, bold: true, color: i === 4 && k === 0 || i === 5 && k === 1 ? BG : c, margin: 0 });
      s.addText(d, { x: x + 0.6, y, w: 5.1, h: 0.4, valign: "middle", fontFace: BODY, fontSize: 10.8, color: (i === 4 && k === 0) || (i === 5 && k === 1) ? INK : MUTE, margin: 0 });
    });
  });
  toolHint(s, "互動教具第 2 層可逐步播放這兩條泳道（SGLang / vLLM 可切換），並看每一步搬多少、多久。", 5.5);
  takeaway(s, "兩家做的都是「KV 所有權轉移」（prefill 的 HBM → decode 的 HBM），差別只在誰先動——推的那一方先算，拉的那一方先等。", MEM);
  footer(s, PB);
})();

// ============================================================ 19 KV 要搬多久
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "18", "一份 5K token 的 KV 跨機只要 7 ms——MLA 在前半段就替這一跳付了帳", MEM);
  stationStepper(s, 2);
  lede(s, "以 DeepSeek 官方統計的平均 KV 長度 4,989 token、跨機 400G（實測 49.5 GB/s）計算。橫軸為對數刻度。", 1.9);
  const lg = (ms) => (Math.log10(ms) + 1) / 5;   // 0.1 ms → 10 s
  hbar(s, MX, 2.3, 11.9, [
    ["MHA（同骨架假想：128 頭 × 128 維）　20 GB", lg(403), "403 ms", WARN],
    ["GQA（Llama-3-70B：8 KV 頭 × 80 層）　1.6 GB", lg(33), "33 ms", COMP],
    ["MLA · BF16（DeepSeek-V3）　351 MB", lg(7.1), "7.1 ms", GOOD],
    ["MLA · FP8　175 MB", lg(3.5), "3.5 ms", GOOD],
  ], MEM, { rh: 0.62, lw: 5.0, max: 1 });
  [[0.1, "0.1 ms"], [1, "1 ms"], [10, "10 ms"], [100, "100 ms"], [1000, "1 s"], [10000, "10 s"]].forEach(([v, t]) => {
    const cx = MX + 5.0 + 0.15 + 5.4 * lg(v);
    s.addShape(pres.shapes.LINE, { x: cx, y: 4.78, w: 0, h: 0.1, line: { color: LINE, width: 1 } });
    s.addText(t, { x: cx - 0.5, y: 4.86, w: 1.0, h: 0.26, align: "center", fontFace: MONO, fontSize: 8.5, color: FOOTC, margin: 0 });
  });
  card(s, MX, 5.2, 5.8, 0.75, BG2, GOOD);
  s.addText("對照：TTFT 實測 2–5 秒（含排隊）→ 7 ms 在 TTFT 裡是雜訊", { x: MX + 0.22, y: 5.2, w: 5.4, h: 0.75, valign: "middle", fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  card(s, 6.8, 5.2, 5.8, 0.75, BG2, WARN);
  s.addText("換成 MHA：光交接就要 0.4 秒，比一整個 decode 步（≈ 92 ms）還久", { x: 7.0, y: 5.2, w: 5.4, h: 0.75, valign: "middle", fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  takeaway(s, "MLA 是為 decode 的 HBM 頻寬發明的，卻順手買到另一件事：prefill 與 decode 可以放在不同機櫃。", MEM);
  footer(s, PB);
})();

// ============================================================ 20 decode 一步
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "19", "Decode 一步：attention 各算各的，MoE 大家一起交換", PURP);
  stationStepper(s, 3);
  const steps = [["① Attention", "每卡只算自己那批\nKV 不重複", MEM], ["② Gate", "每個 token\n選 8 個專家", PURP],
  ["③ Dispatch", "送去專家所在的卡\nFP8、每張卡只送一次", COMP], ["④ Experts", "grouped GEMM\n（DeepGEMM）", COMP],
  ["⑤ Combine", "結果送回原卡\nBF16（位元組 ×2）", MEM], ["⑥ × 58 層", "→ LM head → 取樣\n→ 串流回使用者", PURP]];
  steps.forEach(([t, d, c], i) => {
    const x = MX + i * 2.0;
    card(s, x, 1.95, 1.85, 1.45, BG2, c);
    s.addText(t, { x: x + 0.1, y: 2.05, w: 1.65, h: 0.34, align: "center", fontFace: HEAD, fontSize: 12.5, bold: true, color: c, margin: 0 });
    s.addText(d, { x: x + 0.1, y: 2.42, w: 1.65, h: 0.9, align: "center", valign: "top", fontFace: BODY, fontSize: 10, color: MUTE, lineSpacingMultiple: 1.2, margin: 0 });
    if (i < 5) s.addText("▶", { x: x + 1.87, y: 2.5, w: 0.16, h: 0.3, fontFace: BODY, fontSize: 12, color: FOOTC, margin: 0 });
  });
  [["每卡只放 4 個 routed 專家", "288 個專家槽 ÷ 72 卡，另加 1 個 shared", COMP],
  ["一個 token 平均送往 7.5 張卡", "選 8 個專家，但同一張目的卡只送一次", PURP],
  ["DeepEP low-latency 模式", "純 RDMA（NVSHMEM + IBGDA：GPU 直接敲網卡門鈴、不經 CPU）＋固定 buffer → 可被 CUDA Graph 錄下來（第三堂④）", MEM]]
    .forEach(([t, d, c], i) => {
      const y = 3.6 + i * 0.78;
      s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 11.9, h: 0.68, fill: { color: BG2 }, line: { color: c, width: 1 } });
      s.addText(t, { x: MX + 0.22, y, w: 3.9, h: 0.68, valign: "middle", fontFace: HEAD, fontSize: 13, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 4.2, y, w: 7.4, h: 0.68, valign: "middle", fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });
    });
  toolHint(s, "互動教具第 3 層：拉 EP 大小、切 dispatch 精度，看這個 token 的 8 個專家落在哪幾張卡、幾成要出機器。", 6.05);
  footer(s, PB);
})();

// ============================================================ 21 通訊的時間帳
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "20", "不做重疊時，通訊佔掉 decode 一步約四成", MEM);
  lede(s, "實測一步 92 ms（22,282 tok/s ÷ 8 卡 ÷ 256 序列）；模型算出每層通訊 0.86 ms × 58 層 ≈ 50 ms。");
  const MAXMS = 140;
  s.addText("不開 two-batch overlap", { x: MX, y: 1.85, w: 3.4, h: 0.4, valign: "middle", align: "right", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
  stackBar(s, MX + 3.6, 1.85, 7.2, 0.4, [[75, COMP, "計算 75 ms"], [50, MEM, "通訊 50 ms"]], MAXMS, "124 ms");
  s.addText("開 two-batch overlap（實測設定）", { x: MX, y: 2.4, w: 3.4, h: 0.4, valign: "middle", align: "right", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
  stackBar(s, MX + 3.6, 2.4, 7.2, 0.4, [[75, COMP, "計算 75 ms（與通訊重疊）"], [17, MEM, "17 ms"]], MAXMS, "92 ms");
  s.addText("← 通訊 50 ms 裡約 33 ms 被藏進計算，露在外面的只剩 17 ms", { x: MX + 3.6, y: 2.85, w: 7.5, h: 0.3, fontFace: BODY, fontSize: 10.5, color: FOOTC, margin: 0 });

  tableGrid(s, MX, 3.3, 5.8, [
    { t: "EP 大小", w: 1.5 }, { t: "dispatch", w: 1.3 }, { t: "combine", w: 1.4 }, { t: "× 58 層", w: 1.6 },
  ], [
    ["8", "77 µs", "114 µs", "~11 ms"],
    ["32", "155 µs", "273 µs", "~25 ms"],
    ["128", "192 µs", "369 µs", "~33 ms"],
    ["256", "194 µs", "360 µs", "~32 ms"],
  ], MEM, 10.5, 0.38);
  s.addText("DeepEP low-latency 實測（H800 + CX7 400G、每批 128 token、FP8）", { x: MX, y: 5.32, w: 5.8, h: 0.3, fontFace: BODY, fontSize: 10, color: FOOTC, margin: 0 });

  s.addText("引擎在做的三件事，全是「把等待藏起來」", { x: 6.8, y: 3.3, w: 5.8, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: GOOD, margin: 0 });
  [["Two-batch overlap", "一個 micro-batch 通訊時，另一個在算 → decode +35%", GOOD],
  ["通訊降精度", "dispatch FP8 → NVFP4：all-to-all 流量再減半", PURP],
  ["固定 buffer + CUDA Graph", "每步不重新配置、不回 CPU（第三堂④）", MEM]]
    .forEach(([t, d, c], i) => {
      const y = 3.72 + i * 0.62;
      s.addShape(pres.shapes.RECTANGLE, { x: 6.8, y, w: 5.8, h: 0.54, fill: { color: BG2 }, line: { color: c, width: 1 } });
      s.addText(t, { x: 7.0, y, w: 2.3, h: 0.54, valign: "middle", fontFace: HEAD, fontSize: 11.5, bold: true, color: c, margin: 0 });
      s.addText(d, { x: 9.35, y, w: 3.1, h: 0.54, valign: "middle", fontFace: BODY, fontSize: 9.8, color: MUTE, margin: 0 });
    });
  takeaway(s, "引擎的工作不是讓通訊變快，是把它藏進計算裡——藏不掉的部分，才需要換硬體。", MEM);
  footer(s, PB);
})();

// ============================================================ 22 為什麼拆散便宜
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "21", "拆散之所以便宜：每卡只放 4 個專家，省下的 HBM 拿去開大 batch", GOOD);
  lede(s, "同樣 72 張 H100（每張 80 GB），兩種切法下單卡記憶體的用途完全不同。");
  [["TP16（72 張切成 4.5 組）", [[43, FOOTC, "權重 43 GB"], [37, MEM, "KV 37 GB"]], "MLA 的 latent 無法按 head 切 → 16 張卡各存同一份 KV", WARN],
  ["DP attention + EP72", [[29, FOOTC, "權重 29 GB"], [51, MEM, "KV 51 GB"]], "每張卡只存自己那批請求的 KV，完全不重複", GOOD]]
    .forEach(([t, segs, note, c], i) => {
      const y = 1.85 + i * 1.35;
      s.addText(t, { x: MX, y, w: 3.4, h: 0.45, align: "right", valign: "middle", fontFace: HEAD, fontSize: 13, bold: true, color: c, margin: 0 });
      stackBar(s, MX + 3.6, y, 7.6, 0.45, segs, 80, "80 GB");
      s.addText(note, { x: MX + 3.6, y: y + 0.5, w: 8.4, h: 0.35, fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });
    });
  card(s, MX, 4.55, 5.8, 1.25, BG2, GOOD);
  s.addText("72 張卡能放的「不重複」KV", { x: MX + 0.25, y: 4.65, w: 5.3, h: 0.32, fontFace: HEAD, fontSize: 13, bold: true, color: GOOD, margin: 0 });
  s.addText([{ text: "TP16 ≈ 170 GB", options: { color: WARN, bold: true } }, { text: "　→　", options: { color: MUTE } }, { text: "EP72 ≈ 3.6 TB", options: { color: GOOD, bold: true } }],
    { x: MX + 0.25, y: 5.0, w: 5.3, h: 0.45, valign: "middle", fontFace: MONO, fontSize: 15, margin: 0 });
  s.addText("推算，只看量級（未扣 activation 與通訊 buffer）", { x: MX + 0.25, y: 5.42, w: 5.3, h: 0.3, fontFace: BODY, fontSize: 10, color: FOOTC, margin: 0 });
  [["每卡 256 條序列", "batch 大 → AI ≈ B → 逼近 H100 的 ridge point ≈ 296（第三堂地基）", MEM],
  ["EPLB 讓熱門專家做副本", "decode 2.54×、prefill 1.49×", PURP],
  ["結果：decode vs TP16 為 5.2×", "這就是第 13 頁那條長條的來源", GOOD]]
    .forEach(([t, d, c], i) => {
      const y = 4.55 + i * 0.44;
      s.addText("· " + t, { x: 6.8, y, w: 3.3, h: 0.4, valign: "middle", fontFace: HEAD, fontSize: 11.5, bold: true, color: c, margin: 0 });
      s.addText(d, { x: 10.0, y, w: 2.6, h: 0.4, valign: "middle", fontFace: BODY, fontSize: 9.8, color: MUTE, margin: 0 });
    });
  takeaway(s, "拆散買到的不是算力，是 HBM：權重佔得少 → KV 放得多 → batch 開得大 → memory-bound 的 decode 才划算。", GOOD);
  footer(s, PB);
})();

// ============================================================ 23 代價：節拍器
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "22", "代價是 72 張卡綁成一個節拍器：最慢那張決定速度", WARN);
  lede(s, "DP attention + EP 下，同一個 decode 單元的所有 rank 必須同步進入每一層的 all-to-all。");
  for (let i = 0; i < 72; i++) {
    const x = MX + (i % 18) * 0.49, y = 1.85 + Math.floor(i / 18) * 0.42;
    const slow = i === 29;
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.4, h: 0.33, fill: { color: slow ? WARNTINT : MEMTINT }, line: { color: slow ? WARN : MEM, width: slow ? 1.5 : 0.6 } });
  }
  s.addText("← 這一張卡慢\n（專家熱點／掉卡）\n另外 71 張全部在等", { x: MX + 9.0, y: 2.0, w: 3.2, h: 1.0, valign: "middle", fontFace: BODY, fontSize: 11, bold: true, color: WARN, margin: 0 });
  [["最慢那張決定整體 ITL", "沒請求的卡也得跑空批次陪跑——集合通訊的本質（分散式系統的 straggler 問題）", COMP],
  ["掉一張卡，整個 72 卡單元停擺", "爆炸半徑隨 EP 變大，容錯只能以整個單元為單位", WARN],
  ["EPLB 重排＝權重唯一會搬家的時候", "TRT-LLM 實測單一 FP4 專家 24 MiB，最多重排 348 GiB MoE 權重", PURP]]
    .forEach(([t, d, c], i) => {
      const y = 3.55 + i * 0.8;
      card(s, MX, y, 11.9, 0.72, BG2, c);
      s.addText(t, { x: MX + 0.25, y: y + 0.04, w: 11.3, h: 0.36, valign: "middle", fontFace: HEAD, fontSize: 14, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 0.25, y: y + 0.38, w: 11.3, h: 0.34, valign: "middle", fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });
    });
  takeaway(s, "拆散換來大 batch，代價是 72 張卡變成一支軍樂隊——節拍能踩多快，取決於傳令兵跑多快。", WARN);
  footer(s, PB);
})();

// ============================================================ 24 互動環節
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "23", "互動環節：同一個請求放到三種硬體上各走一次", PURP);
  card(s, MX, 1.4, 11.9, 0.95, BG2, PURP);
  s.addText("interactive/rack_journey_map.html", { x: MX + 0.3, y: 1.4, w: 6.5, h: 0.95, valign: "middle", fontFace: MONO, fontSize: 17, bold: true, color: PURP, margin: 0 });
  [["1–4", "切層"], ["H B G", "切硬體"], ["空白鍵", "播放／暫停"]].forEach(([k, d], i) => {
    pill(s, 7.6 + i * 1.7, 1.62, 1.55, 0.5, `${k}　${d}`, PURP, BG3, PURP, 10);
  });
  tableGrid(s, MX, 2.5, 11.9, [
    { t: "步", w: 0.7 }, { t: "教具操作", w: 4.6 }, { t: "要讓聽眾看到的現象", w: 6.6 },
  ], [
    ["1", "第 1 層全景，停在 H100，點「③ MoE 交換」", "decode 池的交換九成要出機器、走 RDMA"],
    ["2", "按 G 切 GB200 NVL72", "同一種資料，整段路縮回機櫃內"],
    ["3", "第 2 層，attention 切 MHA → MLA", "400 ms → 7 ms；在 TTFT 的 2–5 s 裡變成雜訊"],
    ["4", "第 3 層，EP 從 8 拉到 144（H100）", "EP8 權重放不下；EP 越大、跨機比例越高、每卡權重越少"],
    ["5", "第 4 層時間帳，H100 → B200 → GB200", "NVLink 快一倍，通訊一毫秒都沒省；域從 8 變 72，通訊 50 → 9 ms"],
  ], PURP, 11.5, 0.58);
  takeaway(s, "三種硬體跑同一個請求，router、prefill、KV 交接幾乎不動——只有 decode 的交換那一站變了。", PURP);
  footer(s, PC);
})();

// ============================================================ 25 合回來：真瓶頸
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "24", "四種資料只有一種付不起跨機的路", COMP);
  lede(s, "回到第 15 頁的地圖，逐格結帳：");
  [["① 請求本身", "前端乙太就夠", "✔ 付得起", GOOD], ["② KV 交接", "MLA 讓它 7 ms，跨機櫃也行", "✔ 付得起", GOOD],
  ["④ 權重", "幾乎不搬", "✔ 付得起", GOOD], ["③ MoE 交換", "每字 116 次、每層都要全員同步", "✘ 只有它付不起慢車道", WARN]]
    .forEach(([t, d, v, c], i) => {
      const y = 1.75 + i * 0.55;
      s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 11.9, h: 0.48, fill: { color: c === WARN ? WARNTINT : BG2 }, line: { color: c, width: 1 } });
      s.addText(t, { x: MX + 0.22, y, w: 2.2, h: 0.48, valign: "middle", fontFace: HEAD, fontSize: 13, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 2.5, y, w: 5.6, h: 0.48, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
      s.addText(v, { x: MX + 8.2, y, w: 3.5, h: 0.48, valign: "middle", fontFace: HEAD, fontSize: 12.5, bold: true, color: c, margin: 0 });
    });
  tableGrid(s, MX, 4.05, 11.9, [
    { t: "部署（EP72、均勻路由）", w: 4.3 }, { t: "落在同一個 scale-up 域", w: 2.9 }, { t: "走跨機 RDMA", w: 2.2 }, { t: "每步通訊（模型）", w: 2.5 },
  ], [
    ["H100 8 卡機 × 9 台（主角）", "(8−1) ÷ (72−1) ≈ 10%", "≈ 90%", "≈ 50 ms"],
    ["B200 8 卡機 × 9 台", "同上（NVLink 快一倍也沒用）", "≈ 90%", "≈ 50 ms"],
    ["GB200 NVL72（72 卡一域）", "100%", "0%", "≈ 9 ms"],
  ], COMP, 11, 0.4);
  s.addText("NVLink 5 每方向 900 GB/s，是 400G 網卡（50 GB/s）的 18 倍。（NVIDIA 行銷寫「36×」，是拿雙向合計比單向。）",
    { x: MX, y: 5.68, w: 11.9, h: 0.3, fontFace: BODY, fontSize: 10.5, color: FOOTC, margin: 0 });
  takeaway(s, "MoE 推論選硬體，第一個要看的不是 FLOPs、也不是單卡 HBM，而是「一個 scale-up 域裝得下幾張卡」。", COMP);
  footer(s, PC);
})();

// ============================================================ 26 證據
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "25", "同一顆 GPU、同樣的 NVLink 頻寬，只把域從 8 擴到 72", GOOD);
  lede(s, "SemiAnalysis InferenceX：DeepSeek-R1 FP4、1K/1K、Dynamo + TRT-LLM、含 MTP、125 tok/s/user。");
  tableGrid(s, MX, 1.8, 11.9, [
    { t: "", w: 2.5 }, { t: "GPU", w: 1.5 }, { t: "每卡 NVLink", w: 2.0 }, { t: "域", w: 0.9 }, { t: "拓樸", w: 3.2 }, { t: "tok/s／GPU", w: 1.8 },
  ], [
    ["B200（HGX 8 卡）", "Blackwell", "1.8 TB/s（雙向）", "8", "4 prefill + 40 decode（EP8）", "941"],
    ["GB200 NVL72", "Blackwell", "1.8 TB/s（雙向）", "72", "8 prefill（TP8）+ 16 decode（EP16）", "4,130"],
  ], GOOD, 11.5, 0.5);
  hbar(s, MX, 3.3, 11.9, [
    ["B200 · 每卡 tok/s", 941, "941", FOOTC],
    ["GB200 NVL72 · 每卡 tok/s", 4130, "4,130", GOOD],
  ], GOOD, { rh: 0.6, lw: 4.2 });
  hbar(s, MX, 4.55, 11.9, [
    ["B200 · 每 1M token 成本", 0.576, "$0.576", WARN],
    ["GB200 NVL72 · 每 1M token 成本", 0.149, "$0.149", GOOD],
  ], GOOD, { rh: 0.6, lw: 4.2, max: 0.6 });
  s.addText("旁證：LMSYS 在 GB200 NVL72 上（FP8 attention + NVFP4 MoE）量到 decode 13,386 tok/s／GPU，相對 H100 為 4.8×——但這組同時換了 GPU 世代，只當旁證。",
    { x: MX, y: 5.78, w: 11.9, h: 0.3, fontFace: BODY, fontSize: 10.5, color: FOOTC, margin: 0 });
  takeaway(s, "每卡 4.4×、成本 ÷3.9。教具模型只解釋了通訊那一段，其餘來自域變大後 batch 也能開更大。", GOOD);
  footer(s, PC);
})();

// ============================================================ 27 旅程 × 全系列
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "26", "這趟旅程的每一站，都能在前面找到它的那一頁", MEM);
  tableGrid(s, MX, 1.4, 11.9, [
    { t: "旅程的一站", w: 3.6 }, { t: "決定它快慢的東西", w: 5.3 }, { t: "在哪一堂", w: 3.0 },
  ], [
    ["router 選誰", "cache-aware、局部性 vs 均衡", "本堂第 16 頁"],
    ["prefill 跨機", "compute-bound、大 GEMM", "第一堂 roofline"],
    ["KV 交接 7 ms", "MLA 把 KV 壓到 70 KB／token", "本堂旋鈕①（第 4 頁）"],
    ["KV 放得進 HBM", "分頁 KV、continuous batching", "第三堂 ②"],
    ["decode 為什麼要大 batch", "memory-bound、AI ≈ B", "第一堂、第三堂地基"],
    ["116 次 all-to-all 走哪條線", "scale-up 域 vs RDMA 的頻寬階梯", "第二堂 Part B"],
    ["每步不回 CPU", "CUDA Graph、固定 buffer", "第三堂 ④"],
    ["專家熱點、掉卡", "EPLB、爆炸半徑", "本堂第 22–23 頁"],
    ["每字少搬幾個 byte", "MoE 稀疏、FP8 / NVFP4", "本堂旋鈕②⑤（第 6、9 頁）"],
  ], MEM, 11, 0.44);
  takeaway(s, "第一堂那張記憶體階層表，在最後一堂長成了一座資料中心。", MEM);
  footer(s, PE);
})();

// ============================================================ 28 帶走三句話
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "27", "帶走三句話", MEM);
  [["1", "架構決策就是硬體帳單",
    "MLA 為了 decode 的 HBM 頻寬而生，順手讓 KV 交接只要 7 ms；MoE 讓每個字少算，卻換來每字 116 次的全員交換。模型怎麼設計，決定了機櫃上要搬什麼。", PURP],
  ["2", "72 張卡同步踏一步划算，是因為最常搬的東西走了最快的路",
    "每字 116 次的 MoE 交換被 DeepEP 壓到每層不到 1 ms；一次性的 KV 交接小到可以跨機櫃。四種資料，四條路——這就是機櫃群版的記憶體階層。", MEM],
  ["3", "拆散的收益在 HBM，代價在節拍——所以選硬體先看域的大小",
    "每卡只放 4 個專家、跑 256 條序列，decode 比 TP16 快 5.2×；但付不起慢車道的交換，只能靠更大的 scale-up 域：同一顆 Blackwell，域從 8 到 72，每卡 4.4×。", GOOD]]
    .forEach(([n, t, d, c], i) => {
      const y = 1.6 + i * 1.55;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 0.85, h: 0.85, rectRadius: 0.12, fill: { color: c }, line: { type: "none" } });
      s.addText(n, { x: MX, y, w: 0.85, h: 0.85, align: "center", valign: "middle", fontFace: MONO, fontSize: 30, bold: true, color: BG, margin: 0 });
      s.addText(t, { x: MX + 1.1, y, w: 11.0, h: 0.5, valign: "middle", fontFace: HEAD, fontSize: 17, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 1.1, y: y + 0.52, w: 11.2, h: 0.95, valign: "top", fontFace: BODY, fontSize: 12.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
    });
  s.addText("全系列收束：第一堂硬體的尺 → 第二堂單卡到多卡 → 第三堂單機引擎 → 第四堂從模型架構到一整排機櫃，跟著一個字走完全程。",
    { x: MX, y: 6.35, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 12, color: FOOTC, margin: 0 });
  footer(s, PE);
})();

pres.writeFile({ fileName: "../class4_models_to_racks.html" }).then((f) => console.log("✅ 產生：" + f + "（" + PAGE + " 頁）")).catch((e) => console.error(e));
