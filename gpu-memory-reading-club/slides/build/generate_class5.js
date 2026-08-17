// 第五堂課 — 中國開源模型：把推論成本寫進架構本身
// 產生 ../class5_china_models.pptx。沿用系列的深色「矽晶」主題。
//
// 主幹 = 五個旋鈕（壓 KV / 少算 / 少看 / 一次多產 / 降精度），
//        每一家實驗室只是在這五個旋鈕上轉了不同組合。
// 承接第三/四堂：框架從「外面」調（排程與記憶體管理），模型從「裡面」改（架構本身）——打的是同一個敵人。
const pptxgen = require("pptxgenjs");

const BG = "0E1726", BG2 = "16233A", BG3 = "1C2E4A";
const INK = "EAF1FB", MUTE = "8FA6C4", LINE = "2A3D5C", FOOTC = "5C7299";
const MEM = "38BDF8", COMP = "F59E0B", WARN = "FB7185", GOOD = "34D399", PURP = "A78BFA";
const MEMTINT = "10455F", COMPTINT = "4A3410", WARNTINT = "4A2433", GOODTINT = "123D31", PURPTINT = "2A2150";
const HEAD = "PingFang TC", BODY = "PingFang TC", MONO = "Menlo";

const W = 13.33, H = 7.5, MX = 0.7, TITLE_Y = 0.62, FOOT_Y = 7.05, TOTAL = 16;
const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 });

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "GPU 記憶體與資料搬遷讀書會";
pres.title = "第五堂課 · 中國開源模型的效率工程";

let PAGE = 0;
const base = (s) => { s.background = { color: BG }; PAGE += 1; };
function runningHeader(s) {
  s.addText("讀書會 · 第五堂課 · 中國開源模型", { x: W - 5.6, y: 0.3, w: 4.9, h: 0.3, align: "right", fontFace: BODY, fontSize: 10, color: MUTE, margin: 0 });
}
function footer(s, part) {
  s.addText(part, { x: MX, y: FOOT_Y, w: 9.5, h: 0.3, fontFace: BODY, fontSize: 9, color: FOOTC, margin: 0 });
  s.addText(`${PAGE} / ${TOTAL}`, { x: W - 1.6, y: FOOT_Y, w: 0.9, h: 0.3, align: "right", fontFace: MONO, fontSize: 9, color: FOOTC, margin: 0 });
}
function header(s, num, title, accent) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: TITLE_Y, w: 0.62, h: 0.62, rectRadius: 0.08, fill: { color: accent }, line: { type: "none" }, shadow: shadow() });
  s.addText(num, { x: MX, y: TITLE_Y, w: 0.62, h: 0.62, align: "center", valign: "middle", fontFace: MONO, fontSize: 20, bold: true, color: BG, margin: 0 });
  s.addText(title, { x: MX + 0.85, y: TITLE_Y, w: W - MX - 0.85 - 0.5, h: 0.62, valign: "middle", fontFace: HEAD, fontSize: 24, bold: true, color: INK, margin: 0 });
}
function card(s, x, y, w, h, fill, lineColor) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: lineColor || LINE, width: 1 }, shadow: shadow() });
}
function pill(s, x, y, w, h, text, edge, fill, txt, fs) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: h / 2, fill: { color: fill || BG2 }, line: { color: edge, width: 1 } });
  s.addText(text, { x, y, w, h, align: "center", valign: "middle", fontFace: BODY, fontSize: fs || 11, bold: true, color: txt || edge, margin: 0 });
}
function arrow(s, x, y, w, color, label) {
  s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width: 2.5, endArrowType: "triangle" } });
  if (label) s.addText(label, { x: x - 0.5, y: y + 0.07, w: w + 1.0, h: 0.3, align: "center", fontFace: MONO, fontSize: 10, color, margin: 0 });
}
function takeaway(s, text, color) {
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 6.02, w: 0.09, h: 0.62, fill: { color: color || MEM }, line: { type: "none" } });
  s.addText(text, { x: MX + 0.22, y: 6.0, w: 11.9, h: 0.66, fontFace: HEAD, fontSize: 14.5, bold: true, color: color || MEM, valign: "middle", margin: 0 });
}
function tableGrid(s, x, y, w, cols, rows, accent, fs) {
  const rh = 0.42, hh = 0.42;
  let cx = x;
  cols.forEach((c) => {
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y, w: c.w, h: hh, fill: { color: BG3 }, line: { color: LINE, width: 0.8 } });
    s.addText(c.t, { x: cx + 0.08, y, w: c.w - 0.16, h: hh, valign: "middle", fontFace: HEAD, fontSize: (fs || 11) + 0.5, bold: true, color: INK, margin: 0 });
    cx += c.w;
  });
  rows.forEach((r, ri) => {
    cx = x;
    r.forEach((cell, ci) => {
      s.addShape(pres.shapes.RECTANGLE, { x: cx, y: y + hh + ri * rh, w: cols[ci].w, h: rh, fill: { color: ri % 2 ? BG2 : BG }, line: { color: LINE, width: 0.8 } });
      s.addText(cell, { x: cx + 0.08, y: y + hh + ri * rh, w: cols[ci].w - 0.16, h: rh, valign: "middle", fontFace: ci === 0 ? HEAD : BODY, fontSize: fs || 11, bold: ci === 0, color: ci === 0 ? accent : MUTE, margin: 0 });
      cx += cols[ci].w;
    });
  });
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

const P0 = "讀書會 · 第五堂課";
const PK = "五個旋鈕";
const PL = "五個實驗室";

// ============================================================ 1 標題
(() => {
  const s = pres.addSlide(); base(s);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.16, h: H, fill: { color: PURP }, line: { type: "none" } });
  s.addText("第五堂課", { x: MX + 0.3, y: 1.55, w: 8, h: 0.45, fontFace: MONO, fontSize: 15, color: PURP, margin: 0 });
  s.addText("把推論成本寫進架構本身", { x: MX + 0.3, y: 2.05, w: 10.5, h: 0.85, fontFace: HEAD, fontSize: 40, bold: true, color: INK, margin: 0 });
  s.addText("中國開源模型做了哪些優化？", { x: MX + 0.3, y: 2.95, w: 10.5, h: 0.7, fontFace: HEAD, fontSize: 28, bold: true, color: MUTE, margin: 0 });
  s.addText("DeepSeek · Kimi · MiniMax · Qwen · GLM", { x: MX + 0.3, y: 3.8, w: 11.3, h: 0.4, fontFace: MONO, fontSize: 15, color: PURP, margin: 0 });
  card(s, MX + 0.3, 4.5, 11.3, 1.15, BG2, MEM);
  s.addText([
    { text: "第三、四堂是「框架從外面調」", options: { bold: true, color: MEM } },
    { text: "（排程、記憶體管理、路由）——一個模型權重都沒改。這一堂是", options: { color: INK } },
    { text: "「模型從裡面改」", options: { bold: true, color: PURP } },
    { text: "。兩邊打的是同一個敵人：decode 的 memory-bound 與 KV cache。", options: { color: INK } },
  ], { x: MX + 0.55, y: 4.5, w: 10.8, h: 1.15, valign: "middle", fontFace: HEAD, fontSize: 15, lineSpacingMultiple: 1.35, margin: 0 });
  footer(s, P0);
})();

// ============================================================ 2 命題
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "01", "命題：效率不是加分項，是生存條件", PURP);
  s.addText("出口管制下算力受限，逼出了這批實驗室的三個共同特徵。", { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  [["1", "架構層就為推論成本設計", "不是「先訓練完再想怎麼服務」。MLA 這種注意力機制的發明動機，直接就是 HBM 頻寬帳單——這是最乾淨的架構 ⇄ 硬體 co-design 案例。", PURP],
  ["2", "系統零件跟著開源", "FlashMLA / DeepEP / DeepGEMM / EPLB / 3FS…… 新架構必須被 vLLM、SGLang 支援才有人用，開源 kernel 是讓自家架構進入生態的手段。", MEM],
  ["3", "技術報告寫得很細，含失敗嘗試", "MiniMax 那篇「為什麼 M2 退回 full attention」尤其可貴——負面結果比正面結果更難得，也更有教學價值。", COMP]]
    .forEach(([n, t, d, c], i) => {
      const y = 1.85 + i * 1.5;
      card(s, MX, y, 11.9, 1.32, BG2, c);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.22, y: y + 0.35, w: 0.62, h: 0.62, rectRadius: 0.1, fill: { color: c }, line: { type: "none" } });
      s.addText(n, { x: MX + 0.22, y: y + 0.35, w: 0.62, h: 0.62, align: "center", valign: "middle", fontFace: MONO, fontSize: 22, bold: true, color: BG, margin: 0 });
      s.addText(t, { x: MX + 1.05, y: y + 0.14, w: 10.5, h: 0.42, valign: "middle", fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0 });
      s.addText(d, { x: MX + 1.05, y: y + 0.58, w: 10.5, h: 0.66, valign: "top", fontFace: BODY, fontSize: 11.8, color: MUTE, lineSpacingMultiple: 1.25, margin: 0 });
    });
  footer(s, P0);
})();

// ============================================================ 3 同一個敵人
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "02", "承接前兩堂：同一個敵人，兩個方向", MEM);
  s.addText("回到第三堂的那把尺：AI = FLOPs ÷ Bytes。decode 的 AI ≈ 1，所以 GPU 大半閒置。", { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  card(s, MX, 1.85, 5.8, 2.6, BG2, MEM);
  s.addText("框架的做法：動分子", { x: MX + 0.28, y: 1.98, w: 5.2, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: MEM, margin: 0 });
  s.addText("把 batch 撐大撐滿 → 同一次權重讀取被更多 token 分攤 → AI ≈ B", { x: MX + 0.28, y: 2.45, w: 5.2, h: 0.6, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  ["分頁 KV / RadixAttention", "continuous batching", "chunked prefill / PD 分離", "cache-aware router"].forEach((t, i) =>
    s.addText("· " + t, { x: MX + 0.28, y: 3.1 + i * 0.32, w: 5.2, h: 0.3, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MEM, margin: 0 }));

  card(s, 7.0, 1.85, 5.6, 2.6, BG2, PURP);
  s.addText("模型的做法：動分母", { x: 7.28, y: 1.98, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: PURP, margin: 0 });
  s.addText("直接讓「每產一個 token 要搬的位元組數」變少", { x: 7.28, y: 2.45, w: 5.0, h: 0.6, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  ["壓 KV（MLA）", "少算（MoE 稀疏）", "少看（稀疏/線性注意力）", "降精度（FP8 / MXFP4）"].forEach((t, i) =>
    s.addText("· " + t, { x: 7.28, y: 3.1 + i * 0.32, w: 5.0, h: 0.3, valign: "middle", fontFace: BODY, fontSize: 11.5, color: PURP, margin: 0 }));

  card(s, MX, 4.7, 11.9, 1.1, BG2, GOOD);
  s.addText("兩邊在 roofline 上做的是同一件事：把那個點從左邊的記憶體牆推向右邊。差別只在一個從外面調、一個從裡面改——而模型端改的東西，框架端必須支援才跑得起來（所以他們把 kernel 也開源了）。",
    { x: MX + 0.28, y: 4.7, w: 11.3, h: 1.1, valign: "middle", fontFace: BODY, fontSize: 13, color: MUTE, lineSpacingMultiple: 1.35, margin: 0 });
  footer(s, P0);
})();

// ============================================================ 4 五個旋鈕
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "03", "五個旋鈕：後面每一家都只是轉了不同組合", PURP);
  knobStepper(s, -1);
  tableGrid(s, MX, 2.05, 11.9, [
    { t: "旋鈕", w: 2.1 }, { t: "打擊的瓶頸", w: 4.0 }, { t: "代表技術", w: 5.8 },
  ], [
    ["① 壓 KV", "decode 讀 KV 的頻寬 + 容量", "MHA → GQA → MLA（低秩 latent）→ Gated MLA"],
    ["② 少算", "每 token 的 FLOPs 與權重讀取", "MoE 稀疏化：細粒度專家、共享專家、極高稀疏比"],
    ["③ 少看", "長 context 的 O(n²) 與 KV 線性增長", "稀疏注意力（DSA/CSA/MSA）、線性注意力（Lightning/GDN/KDA）、混合層"],
    ["④ 一次多產", "單請求延遲（memory-bound 天花板）", "MTP 多 token 預測 ＋ 投機解碼"],
    ["⑤ 降精度", "搬的位元組數", "FP8 訓練、MXFP4 權重 / MXFP8 activation 的 QAT"],
  ], PURP, 11.5);
  takeaway(s, "每一個旋鈕都在回答同一個問題：怎麼讓每產一個 token，少搬一點位元組？", PURP);
  footer(s, PK);
})();

// ============================================================ 5 旋鈕① 壓 KV
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "04", "旋鈕①：壓 KV —— MHA → GQA → MLA", MEM);
  knobStepper(s, 0);
  s.addText("KV cache 每 token 的大小，直接決定 decode 每步要搬多少位元組、以及同一顆 HBM 放得下幾條。",
    { x: MX, y: 2.05, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  const evo = [
    ["MHA", "每個 head 各存 K/V", "Llama 式 32 heads：512 KB/token", WARN],
    ["GQA", "多個 query head 共用一組 K/V", "Llama-3-8B（8 KV heads）：128 KB/token（÷4）", COMP],
    ["MLA", "K/V 投影成低秩 latent 再存，用時解回", "DeepSeek-V3：≈ 70 KB/token（同規模 MHA 推算 ~3.8 MB）", GOOD],
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
  s.addText("DeepSeek-V2 論文自陳：MLA 讓 KV cache 相對 MHA 減少 93.3%。Kimi K3 進一步用 Gated MLA。GLM-5 也採用 MLA——這個旋鈕已經是共識。",
    { x: MX + 0.28, y: 4.95, w: 11.3, h: 0.9, valign: "middle", fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  takeaway(s, "最該記的一句：MLA 是「為了 decode 的 HBM 頻寬」而發明的注意力機制——架構決策就是硬體帳單。", MEM);
  footer(s, PK);
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
  takeaway(s, "⚠️ 但 MoE 在單卡上不省容量（專家都得在 HBM）——要連容量也省，得靠第四堂的大規模 EP。", COMP);
  footer(s, PK);
})();

// ============================================================ 7 旋鈕③ 少看（分類）
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
  pill(s, 7.28, 4.62, 5.05, 0.55, "KV 沒了 → 三個生產系統全部要重做", WARN, WARNTINT, WARN, 11.5);

  takeaway(s, "2026 的共識不是「線性取代 full attention」，而是「混合 + 稀疏」——下一頁看 MiniMax 為什麼這樣說。", GOOD);
  footer(s, PK);
})();

// ============================================================ 8 MiniMax 反例（1）
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
  s.addText("「No Free Lunch」講了什麼（LMSYS 部落格 + MiniMax 官方文件）", { x: MX, y: 2.95, w: 11.9, h: 0.38, fontFace: HEAD, fontSize: 16, bold: true, color: WARN, margin: 0 });
  [["① 評測會騙人", "混合注意力在 MMLU / BBH / LongBench 上看起來沒問題，放大後才發現多跳推理明顯退化——把長文件裡散落的線索串起來的能力壞掉了。而要在困難任務上得到統計顯著訊號，所需算力是天文數字。", WARN],
  ["② 理論 FLOPs ≠ wall-clock", "線性注意力的實作本身就是 memory-bound，即使在訓練時也吃不滿算力——完全是第一堂 roofline 的教訓：省下的是紙上的 FLOPs，不是牆上的時間。", COMP]]
    .forEach(([t, d, c], i) => {
      const y = 3.45 + i * 1.25;
      card(s, MX, y, 11.9, 1.1, BG2, c);
      s.addText(t, { x: MX + 0.25, y: y + 0.1, w: 4.0, h: 0.4, valign: "middle", fontFace: HEAD, fontSize: 15, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 0.25, y: y + 0.48, w: 11.3, h: 0.55, valign: "top", fontFace: BODY, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.25, margin: 0 });
    });
  takeaway(s, "「省算力的方法」，需要巨量算力才驗證得了——這個弔詭是效率研究最大的結構性障礙。", WARN);
  footer(s, PL);
})();

// ============================================================ 9 MiniMax 反例（2）生態
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "08", "③ 卡在三個生產系統上（回扣第三堂）", WARN);
  s.addText("這一頁是第三堂與第五堂的接縫：模型端改架構，會直接打壞框架端已經建好的東西。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  [["KV cache", "第三堂問題②", "線性注意力的狀態對數值精度遠比 full attention 敏感 → 低精度存不了，量化那個旋鈕跟著失效", MEM],
  ["Prefix caching", "第三堂問題②", "線性狀態不像 KV 可以直接切片複用 → RadixAttention 的整套價值歸零", PURP],
  ["投機解碼", "第三堂天花板①", "在線性注意力骨幹上「仍是未解問題」——單請求延遲的唯一解法沒了", COMP]]
    .forEach(([t, ref, d, c], i) => {
      const y = 1.85 + i * 1.15;
      card(s, MX, y, 11.9, 1.0, BG2, c);
      s.addText(t, { x: MX + 0.25, y: y + 0.08, w: 3.2, h: 0.42, valign: "middle", fontFace: HEAD, fontSize: 16, bold: true, color: c, margin: 0 });
      s.addText(ref, { x: MX + 3.5, y: y + 0.08, w: 2.2, h: 0.42, valign: "middle", fontFace: MONO, fontSize: 10.5, color: FOOTC, margin: 0 });
      s.addText(d, { x: MX + 0.25, y: y + 0.5, w: 11.3, h: 0.45, valign: "middle", fontFace: BODY, fontSize: 11.8, color: MUTE, margin: 0 });
    });
  card(s, MX, 5.4, 11.9, 0.5, BG2, MUTE);
  s.addText("他們也試過滑動窗口混合，調過比例、RoPE 設定、層內/層間配置、sink token —— 在 agent 任務與複雜長文評測上一致地很差。",
    { x: MX + 0.28, y: 5.4, w: 11.3, h: 0.5, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
  takeaway(s, "「理論複雜度更低」離「生產環境更快」隔著三層：kernel 效率、評測有效性、生態相容性。", WARN);
  footer(s, PL);
})();

// ============================================================ 10 旋鈕④⑤
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "09", "旋鈕④ 一次多產、旋鈕⑤ 降精度", COMP);
  card(s, MX, 1.45, 11.9, 2.0, BG2, COMP);
  s.addText("④ MTP（Multi-Token Prediction）", { x: MX + 0.28, y: 1.58, w: 6, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: COMP, margin: 0 });
  s.addText("訓練時多預測幾步當額外訊號（更密的監督），推論時那些 head 直接當投機解碼的 draft。",
    { x: MX + 0.28, y: 2.02, w: 11.3, h: 0.36, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.28, y: 2.45, w: 11.3, h: 0.45, fill: { color: COMPTINT }, line: { color: COMP, width: 1 } });
  s.addText("驗證 k 個草稿 token：權重讀取 = 1 次（不變），FLOPs = k 倍　→　AI 從 1 變成 k", { x: MX + 0.28, y: 2.45, w: 11.3, h: 0.45, align: "center", valign: "middle", fontFace: MONO, fontSize: 12.5, bold: true, color: COMP, margin: 0 });
  s.addText("DeepSeek-V3 報告第二 token 接受率 ~85–90%；Qwen3-Next 也內建 MTP。這是模型端「主動配合投機解碼」的做法——訓練時就把 draft 模型長在自己身上。",
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
  footer(s, PK);
})();

// ============================================================ 11 DeepSeek
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "10", "DeepSeek：五個旋鈕全轉的教科書", MEM);
  tableGrid(s, MX, 1.4, 11.9, [
    { t: "技術", w: 2.5 }, { t: "旋鈕", w: 1.5 }, { t: "重點", w: 7.9 },
  ], [
    ["MLA", "① 壓 KV", "K/V 投影成低秩 latent 再存。V2 論文自陳 KV cache 相對 MHA 減少 93.3%；V3 latent 576 維 × 61 層 ≈ 70 KB/token"],
    ["DeepSeekMoE", "② 少算", "細粒度專家（切更小、選更多）+ 共享專家（每 token 必經）。V3：671B 總參數、37B 活躍"],
    ["無輔助損失均衡", "② 少算", "傳統 aux loss 逼均衡會傷品質 → 改成動態調整路由 bias，均衡且不干擾主目標"],
    ["MTP", "④ 一次多產", "訓練時多預測幾步當額外訊號；推論時直接當投機解碼的 draft（第二 token 接受率 ~85–90%）"],
    ["FP8 訓練", "⑤ 降精度", "首個大規模用 FP8 完成訓練的開源前沿模型；細粒度 scaling + 高精度累加解決數值問題"],
    ["DualPipe + 通訊 kernel", "系統", "讓 EP 的 all-to-all 與計算重疊，把通訊成本藏起來（第四堂問題⑤）"],
    ["DSA（V3.2-Exp, 2025-09）", "③ 少看", "細粒度稀疏注意力，長 context 成本大降 → API 直接降價 >50%（$0.27/M input）"],
    ["V4（2026-04 預覽）", "③ + 系統", "CSA（每 m token 壓成一筆再取 top-k）+ HCA 逐層交錯、mHC 殘差、Muon。1M ctx 下 FLOPs 只需 V3.2 的 27%、KV cache 只需 10%"],
  ], MEM, 10.2);
  takeaway(s, "V3 → V3.2 → V4 的每一步，都是在原有旋鈕上再多轉一格「少看」——長 context 是現在的主戰場。", MEM);
  footer(s, PL);
})();

// ============================================================ 12 Kimi
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "11", "Moonshot / Kimi：K2 → K3 的稀疏化與量化", COMP);
  tableGrid(s, MX, 1.4, 11.9, [
    { t: "", w: 1.7 }, { t: "Kimi K2（2025-07）", w: 4.6 }, { t: "Kimi K3（2026-07）", w: 5.6 },
  ], [
    ["規模", "1T 總參數 / 32B 活躍", "2.8T 總參數 / 104B 活躍"],
    ["MoE", "384 專家 / 每 token 8 個", "896 專家 / 每 token 16 個（Stable LatentMoE，latent 3584）"],
    ["注意力", "MLA", "93 層 ＝ 69 層 KDA ＋ 24 層 Gated MLA（線性 × 全注意力混合）"],
    ["訓練", "MuonClip（Muon + QK-Clip）；15.5T tokens 全程無 loss spike", "同系列；宣稱整體 scaling efficiency ≈ 2.5× K2"],
    ["精度", "—", "MXFP4 權重 / MXFP8 activation，從 SFT 起做量化感知訓練（QAT）"],
    ["其他", "—", "原生多模態（401M vision encoder）、1M context"],
  ], COMP, 11),
    [["混合而非全押", "69 線性 + 24 full —— 留下足夠的 full attention 維持長程檢索", MEM],
    ["MuonClip", "Muon 優化器 + QK-Clip 抑制 attention logit 爆炸 → 1T 模型訓練不炸", COMP],
    ["出廠就是 4-bit", "QAT 直接產 MXFP4 → 對齊 Blackwell FP4，不靠社群事後量化", GOOD]]
      .forEach(([t, d, c], i) => {
        const x = MX + i * 4.03;
        card(s, x, 4.95, 3.85, 1.05, BG2, c);
        s.addText(t, { x: x + 0.18, y: 5.05, w: 3.5, h: 0.32, fontFace: HEAD, fontSize: 12.5, bold: true, color: c, margin: 0 });
        s.addText(d, { x: x + 0.18, y: 5.38, w: 3.5, h: 0.55, valign: "top", fontFace: BODY, fontSize: 10.5, color: MUTE, lineSpacingMultiple: 1.2, margin: 0 });
      });
  s.addText("「總參數變大、活躍參數只小幅變大」是 2026 主流——因為決定 decode 速度的是活躍參數與 KV，不是總參數。",
    { x: MX, y: 6.15, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 12, color: FOOTC, margin: 0 });
  footer(s, PL);
})();

// ============================================================ 13 Qwen / GLM
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "12", "Qwen / GLM：混合注意力成為 2026 主流", GOOD);
  tableGrid(s, MX, 1.45, 11.9, [
    { t: "模型", w: 2.6 }, { t: "規模", w: 2.4 }, { t: "注意力", w: 3.6 }, { t: "重點", w: 3.3 },
  ], [
    ["Qwen3（2025）", "235B-A22B 等", "GQA + MoE", "稠密與 MoE 全尺寸家族"],
    ["Qwen3-Next", "80B-A3B", "Gated DeltaNet : full ＝ 3:1", "極高稀疏 + MTP"],
    ["Qwen 3.5（2026-02）", "397B-A17B", "延續 GDN 3:1 混合", "宣稱 256K ctx decode 比 Qwen3-Max 快 19×"],
    ["GLM-5（2026-02）", "744B-A40B", "MLA ＋ DSA 式稀疏", "壓 KV 與少看同時用上，200K ctx"],
  ], GOOD, 11.5);
  card(s, MX, 3.95, 11.9, 1.9, BG2, GOOD);
  s.addText("四家的共同結構 —— 已經是 2026 開源前沿的預設配置", { x: MX + 0.28, y: 4.05, w: 8, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: GOOD, margin: 0 });
  [["低活躍比例 MoE", COMP], ["線性/稀疏 + full 混合", MEM], ["MLA 系的 KV 壓縮", PURP], ["MTP", GOOD]]
    .forEach(([t, c], i) => pill(s, MX + 0.3 + i * 2.9, 4.55, 2.75, 0.55, t, c, BG3, c, 11.5));
  s.addText("注意「混合比例」是新的超參數：Qwen 3:1、Kimi K3 約 3:1（69:24）——大家收斂到差不多的比例，這本身就是一個訊號。",
    { x: MX + 0.28, y: 5.22, w: 11.3, h: 0.5, valign: "middle", fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  footer(s, PL);
})();

// ============================================================ 14 全景對照
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "13", "全景：五個實驗室 × 五個旋鈕", PURP);
  tableGrid(s, MX, 1.45, 11.9, [
    { t: "", w: 2.1 }, { t: "① 壓 KV", w: 1.8 }, { t: "② 少算（MoE）", w: 2.5 }, { t: "③ 少看（attention）", w: 3.0 }, { t: "④ 一次多產", w: 1.4 }, { t: "⑤ 降精度", w: 1.1 },
  ], [
    ["DeepSeek", "MLA（−93.3%）", "671B-A37B 細粒度+共享", "DSA → CSA+HCA（1M：FLOPs 27%、KV 10%）", "MTP 85–90%", "FP8 訓練"],
    ["Kimi", "MLA → Gated MLA", "1T-A32B → 2.8T-A104B", "KDA 線性 ×69 + full ×24", "—", "MXFP4 QAT"],
    ["MiniMax", "—", "MoE", "Lightning → 退回 full → MSA 稀疏", "—", "—"],
    ["Qwen", "GQA", "397B-A17B 極高稀疏", "Gated DeltaNet : full ＝ 3:1", "MTP", "—"],
    ["GLM", "MLA", "744B-A40B", "DSA 式稀疏", "—", "—"],
  ], PURP, 10);
  s.addText("註：以各家技術報告／官方部落格公開數字為準；2026 上半年版本迭代極快（各家都有小改版），開講前請對一次官方頁面。",
    { x: MX, y: 4.6, w: 11.9, h: 0.35, fontFace: BODY, fontSize: 10.5, color: FOOTC, margin: 0 });
  card(s, MX, 5.05, 11.9, 1.35, BG2, PURP);
  s.addText("讀這張表的方法", { x: MX + 0.28, y: 5.14, w: 4, h: 0.32, fontFace: HEAD, fontSize: 13.5, bold: true, color: PURP, margin: 0 });
  s.addText("每一格都在回答同一個問題：「怎麼讓每產一個 token，少搬一點位元組？」——壓 KV 少搬 KV、MoE 少搬權重、稀疏注意力少搬歷史、MTP 把搬的成本攤給更多 token、量化讓每個數字更小。這正是第一堂 roofline 的分母。",
    { x: MX + 0.28, y: 5.5, w: 11.3, h: 0.8, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  footer(s, PL);
})();

// ============================================================ 15 開源零件
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "14", "為什麼他們連 kernel 都開源？", MEM);
  s.addText("這是最容易被忽略、但策略上最關鍵的一步——也是這堂課與第三、四堂真正的接縫。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  card(s, MX, 1.85, 11.9, 1.3, BG2, WARN);
  s.addText("困境：新架構如果沒有 kernel，就沒有人跑得動", { x: MX + 0.28, y: 1.97, w: 8, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: WARN, margin: 0 });
  s.addText("MLA 不是標準 attention，vLLM / SGLang 原本的 FlashAttention kernel 直接用不了。MoE 的 all-to-all、FP8 GEMM、專家負載均衡也都一樣——開源模型權重卻沒有配套 kernel，等於發布了一台沒有輪子的車。",
    { x: MX + 0.28, y: 2.35, w: 11.3, h: 0.7, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  const parts = [
    ["FlashMLA", "MLA 的 decode kernel", "讓 ① 壓 KV 真的跑得快", MEM],
    ["DeepEP", "專家平行的 all-to-all 通訊庫", "讓 ② 少算 在多機可行（第四堂⑤）", COMP],
    ["DeepGEMM", "FP8 GEMM", "讓 ⑤ 降精度 吃到 tensor core", GOOD],
    ["EPLB", "專家平行負載均衡器", "解決專家熱點＝資料傾斜（第四堂⑤）", PURP],
  ];
  parts.forEach(([t, d, why, c], i) => {
    const y = 3.35 + i * 0.66;
    s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 11.9, h: 0.58, fill: { color: BG2 }, line: { color: c, width: 1 } });
    s.addText(t, { x: MX + 0.22, y, w: 2.0, h: 0.58, valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: c, margin: 0 });
    s.addText(d, { x: MX + 2.35, y, w: 4.2, h: 0.58, valign: "middle", fontFace: BODY, fontSize: 11.5, color: INK, margin: 0 });
    s.addText(why, { x: MX + 6.7, y, w: 5.0, h: 0.58, align: "right", valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
  });
  takeaway(s, "開源 kernel 是讓自家架構進入 vLLM / SGLang 生態的手段——模型與框架是共生的，不是上下游。", MEM);
  footer(s, PL);
})();

// ============================================================ 16 帶走三句話
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "15", "帶走三句話", PURP);
  [["1", "架構決策就是硬體帳單", "MLA 是為了 decode 的 HBM 頻寬而發明的注意力機制；MoE 稀疏度是為了每 token 的權重讀取量；MXFP4 QAT 是為了對齊 Blackwell 的 tensor core。這批模型不是「先訓練完再想怎麼服務」，成本從第一天就寫在架構裡。", PURP],
  ["2", "五個旋鈕，一個目標", "壓 KV / 少算 / 少看 / 一次多產 / 降精度——全都在回答「怎麼讓每產一個 token 少搬一點位元組」。這是第一堂 roofline 的分母，也是第三堂框架優化的另一半。", MEM],
  ["3", "理論上更省 ≠ 實際上更快", "MiniMax M2 退回 full attention 是 2025–26 最重要的負面結果：kernel 效率、評測有效性、生態相容性三關全過，一個「更有效率的架構」才真的更有效率。而驗證它需要的算力，正是它想省下來的那些。", GOOD]]
    .forEach(([n, t, d, c], i) => {
      const y = 1.85 + i * 1.45;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 0.85, h: 0.85, rectRadius: 0.12, fill: { color: c }, line: { type: "none" } });
      s.addText(n, { x: MX, y, w: 0.85, h: 0.85, align: "center", valign: "middle", fontFace: MONO, fontSize: 30, bold: true, color: BG, margin: 0 });
      s.addText(t, { x: MX + 1.1, y, w: 11.0, h: 0.5, fontFace: HEAD, fontSize: 18, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 1.1, y: y + 0.52, w: 11.2, h: 0.9, valign: "top", fontFace: BODY, fontSize: 12.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
    });
  s.addText("全系列收束：第一堂硬體 → 第二堂多卡 → 第三堂單機引擎 → 第四堂多機服務 → 第五堂模型架構。同一個敵人，五個高度。",
    { x: MX, y: 6.25, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 12.5, color: FOOTC, margin: 0 });
  footer(s, PL);
})();

pres.writeFile({ fileName: "../class5_china_models.pptx" }).then((f) => console.log("✅ 產生：" + f + "（" + PAGE + " 頁）")).catch((e) => console.error(e));
