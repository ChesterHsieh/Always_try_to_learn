// 第三堂課 — SGLang 單機篇：沿著 SGLang 遇到的問題走（問題 ①–④）
// 產生 ../class3_sglang_single_node.pptx。沿用第一/二堂的深色「矽晶」主題。
//
// 主幹 = SGLang 一路上遇到的問題與解法，不是功能表：
//   地基   decode 是 memory-bound → AI ≈ B → 所有問題的共同方向是「把 batch 撐大撐滿」
//   問題①  LLM 程式難寫又跑不快      → 前端 DSL（把程式描述成執行圖）
//   問題②  共享前綴被反覆重算        → RadixAttention（蓋在分頁式 KV 之上的索引層）
//   問題③  結構化輸出不可控又慢      → Compressed FSM
//   問題④  CPU 排程吃掉 GPU 時間     → zero-overhead scheduler / CUDA Graph / chunked prefill
//   天花板 投機解碼·MTP、量化
// 問題 ⑤–⑧（大規模 EP、PD 分離、cache-aware router、容錯）留給第四堂「多機篇」。
//
// 搭配互動教具：interactive/serving_map.html（第 16 頁，本堂只用模式 1–4）
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
pres.title = "第三堂課 · SGLang 單機篇";

let PAGE = 0;
const base = (s) => { s.background = { color: BG }; PAGE += 1; };
function runningHeader(s) {
  s.addText("讀書會 · 第三堂課 · SGLang 單機篇", { x: W - 5.6, y: 0.3, w: 4.9, h: 0.3, align: "right", fontFace: BODY, fontSize: 10, color: MUTE, margin: 0 });
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
function obox(s, x, y, w, h, label, edge, txtColor, fs) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.06, fill: { color: BG2 }, line: { color: edge, width: 1.5 } });
  s.addText(label, { x, y, w, h, align: "center", valign: "middle", fontFace: HEAD, fontSize: fs || 12.5, bold: true, color: txtColor || edge, margin: 0 });
}
function pill(s, x, y, w, h, text, edge, fill, txt, fs) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: h / 2, fill: { color: fill || BG2 }, line: { color: edge, width: 1 } });
  s.addText(text, { x, y, w, h, align: "center", valign: "middle", fontFace: BODY, fontSize: fs || 11, bold: true, color: txt || edge, margin: 0 });
}
function arrow(s, x, y, w, color, label, up) {
  s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width: 2.5, endArrowType: "triangle" } });
  if (label) s.addText(label, { x: x - 0.5, y: up ? y - 0.4 : y + 0.07, w: w + 1.0, h: 0.3, align: "center", fontFace: MONO, fontSize: 10, color, margin: 0 });
}
function varrow(s, x, y, h, color) {
  s.addShape(pres.shapes.LINE, { x, y, w: 0, h, line: { color, width: 2.5, endArrowType: "triangle" } });
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
function timeline(s, x, y, w, segs, label, labColor) {
  s.addText(label, { x: x - 1.55, y, w: 1.45, h: 0.34, align: "right", valign: "middle", fontFace: BODY, fontSize: 11, bold: true, color: labColor || MUTE, margin: 0 });
  let cx = x;
  segs.forEach(([frac, color, txt]) => {
    const sw = w * frac;
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y, w: sw, h: 0.34, fill: { color }, line: { color: LINE, width: 0.6 } });
    if (txt) s.addText(txt, { x: cx, y, w: sw, h: 0.34, align: "center", valign: "middle", fontFace: MONO, fontSize: 8.5, bold: true, color: color === BG3 ? FOOTC : BG, margin: 0 });
    cx += sw;
  });
}

// ── 問題導覽列：本堂 ①–④，下堂 ⑤–⑧（淡出）
const PROBS = ["① 程式難平行", "② 前綴重算", "③ 輸出不可控", "④ CPU 成瓶頸", "⑤ 大 MoE", "⑥ P/D 互擾", "⑦ 多機調度", "⑧ 容錯"];
function probStepper(s, active) {
  const y = 1.4, x0 = MX, w = 1.42, gap = 0.075, h = 0.44;
  PROBS.forEach((b, i) => {
    const mine = i < 4, on = i === active;
    const x = x0 + i * (w + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.07, fill: { color: on ? COMP : BG2 }, line: { color: on ? COMP : mine ? LINE : BG3, width: 1 } });
    s.addText(b, { x, y, w, h, align: "center", valign: "middle", fontFace: BODY, fontSize: 9.5, bold: on, color: on ? BG : mine ? MUTE : FOOTC, margin: 0 });
    if (i === 3) s.addShape(pres.shapes.LINE, { x: x + w + gap / 2, y: y - 0.08, w: 0, h: h + 0.16, line: { color: FOOTC, width: 1, dashType: "dash" } });
  });
  s.addText("本堂（單機）", { x: MX, y: 1.9, w: 6.0, h: 0.24, fontFace: BODY, fontSize: 9, color: COMP, margin: 0 });
  s.addText("第四堂（多機）", { x: MX + 6.1, y: 1.9, w: 6.0, h: 0.24, fontFace: BODY, fontSize: 9, color: FOOTC, margin: 0 });
}
// 問題頁的大標區塊
function problemBanner(s, num, problem, why, accent) {
  card(s, MX, 2.35, 11.9, 1.5, BG2, accent);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.25, y: 2.72, w: 0.75, h: 0.75, rectRadius: 0.12, fill: { color: accent }, line: { type: "none" } });
  s.addText(num, { x: MX + 0.25, y: 2.72, w: 0.75, h: 0.75, align: "center", valign: "middle", fontFace: MONO, fontSize: 26, bold: true, color: BG, margin: 0 });
  s.addText(problem, { x: MX + 1.25, y: 2.5, w: 10.3, h: 0.5, valign: "middle", fontFace: HEAD, fontSize: 21, bold: true, color: INK, margin: 0 });
  s.addText(why, { x: MX + 1.25, y: 3.02, w: 10.3, h: 0.7, valign: "top", fontFace: BODY, fontSize: 13, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
}

const P0 = "讀書會 · 第三堂課";
const PA = "地基 · 為什麼這些問題都指向同一件事";
const P1 = "問題① 程式難平行 → 前端 DSL";
const P2 = "問題② 前綴重算 → RadixAttention";
const P3 = "問題③ 輸出不可控 → Compressed FSM";
const P4 = "問題④ CPU 成瓶頸 → 排程與圖執行";
const P5 = "單機的天花板";

// ============================================================ 1 標題
(() => {
  const s = pres.addSlide(); base(s);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.16, h: H, fill: { color: COMP }, line: { type: "none" } });
  s.addText("第三堂課 · SGLang 單機篇", { x: MX + 0.3, y: 1.5, w: 8, h: 0.45, fontFace: MONO, fontSize: 15, color: COMP, margin: 0 });
  s.addText("沿著 SGLang 遇到的問題走", { x: MX + 0.3, y: 2.0, w: 10.5, h: 0.85, fontFace: HEAD, fontSize: 40, bold: true, color: INK, margin: 0 });
  s.addText("一個推論引擎，是被哪些問題逼出來的？", { x: MX + 0.3, y: 2.88, w: 10.5, h: 0.7, fontFace: HEAD, fontSize: 28, bold: true, color: MUTE, margin: 0 });
  s.addText("① 程式難平行　② 前綴重算　③ 輸出不可控　④ CPU 成瓶頸　　（⑤–⑧ 多機問題見第四堂）",
    { x: MX + 0.3, y: 3.78, w: 11.3, h: 0.4, fontFace: BODY, fontSize: 14, color: COMP, margin: 0 });
  card(s, MX + 0.3, 4.5, 11.3, 1.15, BG2, MEM);
  s.addText([
    { text: "這堂不照功能表講，照問題講。", options: { bold: true, color: MEM } },
    { text: "因為每個機制都是被一個具體的痛點逼出來的——先有痛點，機制才記得住。而所有痛點的根源只有一個：", options: { color: INK } },
    { text: "decode 是 memory-bound，GPU 大半時間在空轉。", options: { bold: true, color: COMP } },
  ], { x: MX + 0.55, y: 4.5, w: 10.8, h: 1.15, valign: "middle", fontFace: HEAD, fontSize: 15, lineSpacingMultiple: 1.35, margin: 0 });
  footer(s, P0);
})();

// ============================================================ 2 問題全景
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "01", "全景：SGLang 一路上遇到的八個問題", COMP);
  s.addText("由淺入深，剛好走完「編程模型 → 單機記憶體 → 單機排程 → 多卡 → 多機」。本堂做前四個。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  const rows = [
    ["①", "LLM 程式（多次呼叫、分支、工具）難寫又跑不快", "前端 DSL：把程式描述成執行圖", "編程模型", COMP, true],
    ["②", "這些程式天然共享大量前綴，卻被反覆重算", "RadixAttention：前綴樹 + LRU + cache-aware 排程", "單機記憶體", MEM, true],
    ["③", "結構化輸出逐 token 檢查語法太慢、格式仍不保證", "Compressed FSM：預編譯 + 位元遮罩", "單機生成", PURP, true],
    ["④", "GPU 一步只要 5–10 ms，CPU 排程反而成了瓶頸", "zero-overhead scheduler / CUDA Graph", "單機排程", GOOD, true],
    ["⑤", "大 MoE（DeepSeek）單機放不下、專家負載不均", "大規模 EP + DeepEP / EPLB", "多卡", FOOTC, false],
    ["⑥", "prefill 與 decode 互相干擾（TTFT vs ITL）", "PD 分離", "多卡", FOOTC, false],
    ["⑦", "多副本之間：快取局部性 vs 負載均衡此消彼長", "cache-aware router + KV 複製", "多機", FOOTC, false],
    ["⑧", "副本掛掉，進行中的請求與它的 KV 怎麼辦", "容錯", "多機", FOOTC, false],
  ];
  rows.forEach(([n, p, sol, lv, c, mine], i) => {
    const y = 1.85 + i * 0.54;
    s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 11.9, h: 0.48, fill: { color: mine ? BG2 : BG }, line: { color: mine ? c : LINE, width: mine ? 1 : 0.6 } });
    s.addText(n, { x: MX + 0.12, y, w: 0.5, h: 0.48, align: "center", valign: "middle", fontFace: MONO, fontSize: 14, bold: true, color: c, margin: 0 });
    s.addText(p, { x: MX + 0.72, y, w: 4.8, h: 0.48, valign: "middle", fontFace: BODY, fontSize: 11.5, color: mine ? INK : FOOTC, margin: 0 });
    s.addText("→", { x: MX + 5.55, y, w: 0.3, h: 0.48, align: "center", valign: "middle", fontFace: BODY, fontSize: 11, color: FOOTC, margin: 0 });
    s.addText(sol, { x: MX + 5.9, y, w: 4.4, h: 0.48, valign: "middle", fontFace: HEAD, fontSize: 11.5, bold: mine, color: mine ? c : FOOTC, margin: 0 });
    s.addText(lv, { x: MX + 10.4, y, w: 1.4, h: 0.48, align: "right", valign: "middle", fontFace: MONO, fontSize: 10, color: FOOTC, margin: 0 });
  });
  s.addShape(pres.shapes.LINE, { x: MX, y: 1.85 + 4 * 0.54 - 0.03, w: 11.9, h: 0, line: { color: COMP, width: 1.5, dashType: "dash" } });
  takeaway(s, "線以上是本堂（單機），線以下是第四堂（多機）。同一條主幹，分兩次走完。", COMP);
  footer(s, P0);
})();

// ============================================================ 3 地基① 0.3%
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "02", "地基①：所有問題的共同根源", WARN);
  s.addText("在講任何機制之前，先確認「敵人是誰」。答案在第一堂就給過了。", { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  card(s, MX, 1.85, 5.8, 3.4, BG2, WARN);
  s.addText("每產一個 token（batch = 1）", { x: MX + 0.28, y: 2.02, w: 5.2, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0 });
  [["搬（Bytes）", "整份權重讀一遍 = 2N bytes", MEM],
  ["算（FLOPs）", "每個參數一次乘加 = 2N FLOPs", COMP],
  ["算術強度 AI", "2N ÷ 2N = 1 FLOP/Byte", WARN]].forEach(([k, v, c], i) => {
    const y = 2.55 + i * 0.62;
    s.addText(k, { x: MX + 0.28, y, w: 1.9, h: 0.5, valign: "middle", fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
    s.addText(v, { x: MX + 2.2, y, w: 3.4, h: 0.5, valign: "middle", fontFace: MONO, fontSize: 12.5, bold: true, color: c, margin: 0 });
  });
  s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.28, y: 4.45, w: 5.25, h: 0.62, fill: { color: WARNTINT }, line: { color: WARN, width: 1 } });
  s.addText("H100 ridge point ≈ 296 FLOPs/Byte", { x: MX + 0.28, y: 4.45, w: 5.25, h: 0.62, align: "center", valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: WARN, margin: 0 });

  card(s, 7.0, 1.85, 5.6, 3.4, BG2, WARN);
  s.addText("算力利用率上限", { x: 7.28, y: 2.02, w: 5.0, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0 });
  s.addText("1 ÷ 296", { x: 7.28, y: 2.5, w: 5.0, h: 0.5, fontFace: MONO, fontSize: 18, color: MUTE, margin: 0 });
  s.addText("≈ 0.34 %", { x: 7.28, y: 3.0, w: 5.0, h: 1.0, fontFace: MONO, fontSize: 54, bold: true, color: WARN, margin: 0 });
  s.addText("你買的 H100，99.7% 的算力在閒置。\n卡越強，batch=1 越浪費（A100 是 0.6%）。", { x: 7.28, y: 4.15, w: 5.0, h: 0.9, fontFace: BODY, fontSize: 12.5, color: MUTE, lineSpacingMultiple: 1.35, margin: 0 });
  takeaway(s, "接下來每一個問題，本質上都是「這 99.7% 為什麼填不滿」的不同面向。", WARN);
  footer(s, PA);
})();

// ============================================================ 4 地基② AI≈B
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "03", "地基②：AI ≈ B，所以答案永遠是「把 batch 撐大撐滿」", COMP);
  card(s, MX, 1.45, 11.9, 1.6, BG2, COMP);
  s.addText("搬：權重還是只讀「一遍」（2N bytes）　　算：2N × B FLOPs　　→　AI ≈ B",
    { x: MX + 0.3, y: 1.58, w: 11.3, h: 0.5, valign: "middle", fontFace: MONO, fontSize: 17, bold: true, color: COMP, margin: 0 });
  s.addText("既然 ridge point ≈ 296，B 要拉到「幾百」GPU 才餵得飽。而 B 撐不大的每一個理由，就是後面每一個問題。",
    { x: MX + 0.3, y: 2.12, w: 11.3, h: 0.45, valign: "middle", fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.3, y: 2.55, w: 11.3, h: 0.42, fill: { color: COMPTINT }, line: { color: COMP, width: 1 } });
  s.addText("Llama-3-8B on H100：batch=1 ≈ 208 tok/s　→　batch=64 ≈ 13,000 tok/s（步時幾乎不變）",
    { x: MX + 0.3, y: 2.55, w: 11.3, h: 0.42, align: "center", valign: "middle", fontFace: MONO, fontSize: 12.5, bold: true, color: COMP, margin: 0 });

  s.addText("四個問題各自擋住 B 的哪一段", { x: MX, y: 3.3, w: 6, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0 });
  const map = [
    ["①", "程式難平行", "根本沒有那麼多請求同時進來——runtime 看不見可平行的分支", COMP],
    ["②", "前綴重算", "HBM 被浪費的 KV 佔住 → 放不下更多條；算力浪費在不必算的東西上", MEM],
    ["③", "輸出不可控", "每步的語法檢查是 CPU 上的序列工作 → 拖慢整批", PURP],
    ["④", "CPU 成瓶頸", "GPU 算完在等 CPU 決定下一步 → 有效 batch 再大也填不滿時間軸", GOOD],
  ];
  map.forEach(([n, t, d, c], i) => {
    const y = 3.75 + i * 0.58;
    s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 11.9, h: 0.52, fill: { color: BG2 }, line: { color: c, width: 1 } });
    s.addText(n, { x: MX + 0.12, y, w: 0.5, h: 0.52, align: "center", valign: "middle", fontFace: MONO, fontSize: 14, bold: true, color: c, margin: 0 });
    s.addText(t, { x: MX + 0.72, y, w: 2.2, h: 0.52, valign: "middle", fontFace: HEAD, fontSize: 12.5, bold: true, color: c, margin: 0 });
    s.addText(d, { x: MX + 3.0, y, w: 8.8, h: 0.52, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
  });
  takeaway(s, "把這張圖記住，後面就不會覺得是在背功能表——每個機制都在拆掉擋住 B 的一塊石頭。", COMP);
  footer(s, PA);
})();

// ============================================================ 5 問題①
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "04", "問題①：LLM 程式難寫，而且跑不快", COMP);
  probStepper(s, 0);
  problemBanner(s, "①", "Python 迴圈把「依賴關係」寫死在控制流裡",
    "多次呼叫、分支、工具使用、self-consistency 取樣——這些步驟裡有很多其實可以同時跑，但你用 for 迴圈寫出來之後，runtime 只看得到「一個接一個的請求」，看不見哪幾步彼此無關。加上 Python 的 GIL，前端本身也不擅長真正的並行。", COMP);
  const bad = [
    ["寫的人痛", "分支、重試、工具呼叫的膠水程式又臭又長"],
    ["跑的人痛", "runtime 收到的是一串互不相干的請求，無從批處理"],
    ["更痛的是", "這些請求其實高度共享前綴（同一個 system prompt、同一段歷史），但沒人告訴 runtime"],
  ];
  bad.forEach(([t, d], i) => {
    const x = MX + i * 4.03;
    card(s, x, 4.1, 3.85, 1.55, BG2, WARN);
    s.addText(t, { x: x + 0.2, y: 4.22, w: 3.5, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: WARN, margin: 0 });
    s.addText(d, { x: x + 0.2, y: 4.6, w: 3.5, h: 0.95, valign: "top", fontFace: BODY, fontSize: 11.2, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  });
  takeaway(s, "SGLang 的名字就是這裡來的：Structured Generation Language——它先是一個語言，才是一個引擎。", COMP);
  footer(s, P1);
})();

// ============================================================ 6 解法① DSL
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "05", "解法①：把程式描述成一張執行圖", COMP);
  s.addText("前端 DSL（sgl.gen / sgl.fork / sgl.select）不是語法糖——它讓 runtime 看見「什麼依賴什麼」。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });

  card(s, MX, 1.8, 5.5, 3.6, BG2, WARN);
  s.addText("Python for 迴圈：序列", { x: MX + 0.25, y: 1.92, w: 5.0, h: 0.32, fontFace: HEAD, fontSize: 13.5, bold: true, color: WARN, margin: 0 });
  ["prompt + 分支 A", "prompt + 分支 B", "prompt + 分支 C", "彙整"].forEach((t, i) => {
    obox(s, MX + 1.35, 2.35 + i * 0.72, 2.8, 0.5, t, WARN, WARN, 11.5);
    if (i < 3) varrow(s, MX + 2.75, 2.87 + i * 0.72, 0.18, FOOTC);
  });
  s.addText("runtime 看到的是 4 個先後到達的請求", { x: MX + 0.25, y: 5.0, w: 5.0, h: 0.3, align: "center", fontFace: BODY, fontSize: 11, color: WARN, margin: 0 });

  arrow(s, 6.4, 3.4, 0.7, MUTE);

  card(s, 7.35, 1.8, 5.25, 3.6, BG2, GOOD);
  s.addText("sgl.fork()：一張圖", { x: 7.6, y: 1.92, w: 4.8, h: 0.32, fontFace: HEAD, fontSize: 13.5, bold: true, color: GOOD, margin: 0 });
  obox(s, 8.85, 2.35, 2.3, 0.5, "共享 prompt", PURP, PURP, 11.5);
  varrow(s, 9.4, 2.87, 0.28, PURP);
  ["A", "B", "C"].forEach((t, i) => obox(s, 7.7 + i * 1.62, 3.25, 1.45, 0.5, "分支 " + t, GOOD, GOOD, 11.5));
  s.addText("▼ 三條可同時跑", { x: 7.6, y: 3.85, w: 4.8, h: 0.3, align: "center", fontFace: BODY, fontSize: 10.5, color: GOOD, margin: 0 });
  obox(s, 8.85, 4.2, 2.3, 0.5, "彙整", GOOD, GOOD, 11.5);
  s.addText("runtime 看到：一個共享前綴 + 三條可平行分支", { x: 7.6, y: 4.85, w: 4.8, h: 0.4, align: "center", fontFace: BODY, fontSize: 11, color: GOOD, margin: 0 });

  takeaway(s, "關鍵副作用：fork 產生的分支天然共享同一段前綴——這正好把問題② 端到 RadixAttention 面前。", COMP);
  footer(s, P1);
})();

// ============================================================ 7 DSL → 問題② 的橋
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "06", "但是——好寫的並行程式，會製造出大量重複計算", MEM);
  s.addText("DSL 解決了「看得見平行」，卻立刻暴露下一個問題：那些分支的前綴，難道要各算一遍？",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  card(s, MX, 1.85, 11.9, 1.9, BG2, WARN);
  s.addText("一個 fork 出 8 條分支的程式，system prompt 2,000 token", { x: MX + 0.28, y: 1.97, w: 8, h: 0.35, fontFace: HEAD, fontSize: 14.5, bold: true, color: WARN, margin: 0 });
  timeline(s, MX + 1.9, 2.45, 9.4, [[0.75, COMP, "重算 system prompt（2,000 token prefill）"], [0.25, MEM, "真正新增的內容"]], "分支 1", MUTE);
  timeline(s, MX + 1.9, 2.83, 9.4, [[0.75, COMP, "又算一遍"], [0.25, MEM, "新增"]], "分支 2", MUTE);
  timeline(s, MX + 1.9, 3.21, 9.4, [[0.75, COMP, "又算一遍 ×8…"], [0.25, MEM, "新增"]], "分支 3–8", MUTE);

  s.addText("而且真實流量本來就長這樣", { x: MX, y: 4.0, w: 6, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0 });
  const cases = [["RAG", "同幾份文件反覆進 prompt", MEM], ["多輪對話", "每輪都帶著整段歷史", GOOD], ["Few-shot", "同一批範例貼在每個請求前面", PURP], ["Agent", "工具描述 + 前面所有步驟", COMP]];
  cases.forEach(([t, d, c], i) => {
    const x = MX + i * 3.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.45, w: 2.85, h: 1.0, rectRadius: 0.06, fill: { color: BG2 }, line: { color: c, width: 1 } });
    s.addText(t, { x: x + 0.15, y: 4.55, w: 2.55, h: 0.32, fontFace: HEAD, fontSize: 13, bold: true, color: c, margin: 0 });
    s.addText(d, { x: x + 0.15, y: 4.88, w: 2.55, h: 0.5, valign: "top", fontFace: BODY, fontSize: 10.8, color: MUTE, lineSpacingMultiple: 1.2, margin: 0 });
  });
  takeaway(s, "所以問題②不是「順便做個快取」，它是 DSL 這條路能不能走通的前提。", MEM);
  footer(s, P1);
})();

// ============================================================ 8 問題②
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "07", "問題②：共享前綴被反覆重算", MEM);
  probStepper(s, 1);
  problemBanner(s, "②", "要複用前綴，得先解決「KV cache 怎麼放」",
    "複用的東西是 KV cache。但傳統做法是照 max_seq_len 連續預留一整塊——不知道會生成多長，只好照最壞情況要。結果多數請求只用到預留量的 20–30%，其餘是死重，HBM 放不下更多條（B 上不去），更別說跨請求共享。", MEM);
  card(s, MX, 4.05, 5.8, 1.6, BG2, MEM);
  s.addText("KV cache 一個 token 多少？", { x: MX + 0.25, y: 4.15, w: 5.3, h: 0.32, fontFace: HEAD, fontSize: 13.5, bold: true, color: INK, margin: 0 });
  s.addText("2(K,V) × n_kv_heads × head_dim × n_layers × dtype", { x: MX + 0.25, y: 4.5, w: 5.3, h: 0.3, fontFace: MONO, fontSize: 10.5, color: MEM, margin: 0 });
  s.addText("Llama-3-8B（GQA 8 heads、128 dim、32 層、fp16）＝ 128 KB/token → 一條 8K 序列就是 1 GB", { x: MX + 0.25, y: 4.82, w: 5.3, h: 0.7, valign: "top", fontFace: BODY, fontSize: 11, color: MUTE, lineSpacingMultiple: 1.25, margin: 0 });

  s.addText("預留 vs 實際使用", { x: 6.85, y: 4.15, w: 5.7, h: 0.32, fontFace: HEAD, fontSize: 13.5, bold: true, color: INK, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 6.85, y: 4.55, w: 5.75, h: 0.55, fill: { color: BG3 }, line: { color: WARN, width: 1.2 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 6.85, y: 4.55, w: 5.75 * 0.25, h: 0.55, fill: { color: MEM }, line: { type: "none" } });
  s.addText("用到 25%", { x: 6.85, y: 4.55, w: 5.75 * 0.25, h: 0.55, align: "center", valign: "middle", fontFace: BODY, fontSize: 10.5, bold: true, color: BG, margin: 0 });
  s.addText("浪費的 60–80%", { x: 6.85 + 5.75 * 0.25, y: 4.55, w: 5.75 * 0.75, h: 0.55, align: "center", valign: "middle", fontFace: BODY, fontSize: 11.5, bold: true, color: WARN, margin: 0 });
  s.addText("放得下的條數直接被砍到 1/4 → B 上不去", { x: 6.85, y: 5.15, w: 5.75, h: 0.35, fontFace: BODY, fontSize: 11, color: WARN, margin: 0 });
  takeaway(s, "記憶體碎片化不是「記憶體問題」，是 batch size 問題；batch size 就是 GPU 利用率。", MEM);
  footer(s, P2);
})();

// ============================================================ 9 分頁（KV 怎麼放）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "08", "第一層解法：分頁——KV cache 怎麼「放」", MEM);
  s.addText("把 1960 年代的虛擬記憶體分頁機制搬到 KV cache 上（vLLM 的 PagedAttention 讓它出名，SGLang 底下同樣是分頁式記憶體池）。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  s.addText("邏輯序列（使用者看到的）", { x: MX, y: 1.82, w: 4.0, h: 0.3, fontFace: HEAD, fontSize: 12.5, bold: true, color: MUTE, margin: 0 });
  ["tok 0–15", "tok 16–31", "tok 32–47"].forEach((t, i) => obox(s, MX, 2.2 + i * 0.62, 2.7, 0.5, t, MEM, MEM, 11.5));
  card(s, 3.75, 2.05, 2.5, 2.4, BG2, COMP);
  s.addText("block table", { x: 3.75, y: 2.15, w: 2.5, h: 0.32, align: "center", fontFace: MONO, fontSize: 12, bold: true, color: COMP, margin: 0 });
  ["0 → #47", "1 → #12", "2 → #89"].forEach((t, i) =>
    s.addText(t, { x: 3.9, y: 2.55 + i * 0.5, w: 2.2, h: 0.42, align: "center", valign: "middle", fontFace: MONO, fontSize: 12, color: INK, margin: 0 }));
  arrow(s, 2.75, 2.45, 0.9, MEM);
  arrow(s, 6.35, 2.45, 0.75, COMP);
  s.addText("實體 HBM block pool（16 token / block，可不連續）", { x: 7.2, y: 1.82, w: 5.4, h: 0.3, fontFace: HEAD, fontSize: 12.5, bold: true, color: MUTE, margin: 0 });
  const used = { 3: MEM, 7: MEM, 12: MEM, 1: GOOD, 9: GOOD, 14: PURP };
  for (let i = 0; i < 20; i++) {
    const c = used[i], x = 7.2 + (i % 5) * 1.05, y = 2.2 + Math.floor(i / 5) * 0.56;
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.92, h: 0.44, fill: { color: c || BG3 }, line: { color: c || LINE, width: 1 } });
  }
  s.addText([{ text: "■ 請求 A　", options: { color: MEM } }, { text: "■ 請求 B　", options: { color: GOOD } },
  { text: "■ 請求 C　", options: { color: PURP } }, { text: "■ free_block_queue", options: { color: FOOTC } }],
    { x: 7.2, y: 4.6, w: 5.4, h: 0.3, fontFace: BODY, fontSize: 10.5, margin: 0 });
  [["浪費 60–80% → < 4%", GOOD], ["用完才要、不用連續", MEM], ["請求結束立刻歸還", MEM], ["block 可跨請求共享 ←關鍵", PURP]]
    .forEach(([t, c], i) => pill(s, MX + i * 3.0, 5.1, 2.85, 0.52, t, c, BG2, c, 11.5));
  takeaway(s, "最後那格才是重點：block 能被多個請求指向——這就打開了「複用」的門。", MEM);
  footer(s, P2);
})();

// ============================================================ 10 RadixAttention
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "09", "第二層解法：RadixAttention——block 怎麼被「找到」", PURP);
  s.addText("光能共享還不夠，得有人負責「這個新請求的前綴，之前算過嗎？算到哪？」——這是索引問題。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  card(s, MX, 1.8, 6.6, 3.5, BG2, PURP);
  s.addText("KV cache 組成一棵 radix tree（鍵＝token 序列）", { x: MX + 0.25, y: 1.92, w: 6.1, h: 0.32, fontFace: HEAD, fontSize: 13, bold: true, color: PURP, margin: 0 });
  obox(s, MX + 2.0, 2.4, 2.6, 0.5, "共用 system prompt", PURP, PURP, 11);
  varrow(s, MX + 2.3, 2.92, 0.42, PURP); varrow(s, MX + 4.1, 2.92, 0.42, PURP);
  obox(s, MX + 0.6, 3.36, 2.6, 0.48, "對話 A · 第 1 輪", MEM, MEM, 10.5);
  obox(s, MX + 3.3, 3.36, 2.6, 0.48, "對話 B · 第 1 輪", GOOD, GOOD, 10.5);
  varrow(s, MX + 1.9, 3.86, 0.4, MEM);
  obox(s, MX + 0.6, 4.3, 2.6, 0.48, "對話 A · 第 2 輪", MEM, MEM, 10.5);
  s.addText("命中的節點直接複用 KV block，只 prefill 新增那一段；LRU 淘汰 + cache-aware 排程把命中率最大化。",
    { x: MX + 0.25, y: 4.86, w: 6.1, h: 0.36, fontFace: BODY, fontSize: 10.8, color: MUTE, margin: 0 });

  card(s, 7.65, 1.8, 4.95, 3.5, BG2, WARN);
  s.addText("⚠️ 最常見的誤解", { x: 7.9, y: 1.92, w: 4.5, h: 0.32, fontFace: HEAD, fontSize: 13, bold: true, color: WARN, margin: 0 });
  s.addText("前綴匹配是嚴格按 token 順序從頭比對。", { x: 7.9, y: 2.32, w: 4.5, h: 0.32, fontFace: BODY, fontSize: 11.5, color: INK, margin: 0 });
  s.addText("「怎麼退貨」與「運費怎麼算」都含「怎麼」——", { x: 7.9, y: 2.68, w: 4.5, h: 0.3, fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: 7.9, y: 3.0, w: 4.45, h: 0.45, fill: { color: WARNTINT }, line: { color: WARN, width: 1 } });
  s.addText("但它們不會被合併成同一個節點", { x: 7.9, y: 3.0, w: 4.45, h: 0.45, align: "center", valign: "middle", fontFace: HEAD, fontSize: 11.5, bold: true, color: WARN, margin: 0 });
  s.addText("只有共同「開頭」才算數。局部相同沒有用——因為 KV 是按位置逐 token 累積出來的，第 3 個 token 的 K/V 依賴前面 2 個。",
    { x: 7.9, y: 3.55, w: 4.5, h: 0.85, valign: "top", fontFace: BODY, fontSize: 11, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  s.addText("快取滿了：LRU 淘汰最久未複用的節點；也可分層下放 CPU／磁碟。",
    { x: 7.9, y: 4.5, w: 4.5, h: 0.7, valign: "top", fontFace: BODY, fontSize: 11, color: FOOTC, lineSpacingMultiple: 1.3, margin: 0 });
  takeaway(s, "SGLang 論文宣稱：可共用前綴的工作負載，最高 6.4× 吞吐。", PURP);
  footer(s, P2);
})();

// ============================================================ 11 假對立澄清
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "10", "澄清：「RadixAttention vs PagedAttention」是假對立", GOOD);
  s.addText("網路文章常把兩者寫成競品。它們其實在不同層——而且 SGLang 兩層都有。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  const layers = [
    ["索引 / 複用層", "已經放好的 KV block 怎麼被找到並複用？", "SGLang：radix tree（前綴樹）　｜　vLLM：16-token 區塊的滾動雜湊表", PURP],
    ["記憶體配置層", "KV cache 在 HBM 裡怎麼放？", "兩邊都是分頁式：固定大小 block、不連續、用完才要", MEM],
  ];
  layers.forEach(([t, q, impl, c], i) => {
    const y = 1.85 + i * 1.45;
    card(s, MX, y, 11.9, 1.25, BG2, c);
    s.addText(t, { x: MX + 0.3, y: y + 0.12, w: 3.0, h: 0.45, valign: "middle", fontFace: HEAD, fontSize: 16, bold: true, color: c, margin: 0 });
    s.addText(q, { x: MX + 3.4, y: y + 0.12, w: 8.2, h: 0.45, valign: "middle", fontFace: BODY, fontSize: 12.5, color: INK, margin: 0 });
    s.addText(impl, { x: MX + 0.3, y: y + 0.62, w: 11.3, h: 0.5, valign: "middle", fontFace: MONO, fontSize: 11.5, color: MUTE, margin: 0 });
  });
  s.addText("▲ 蓋在上面", { x: MX + 0.3, y: 3.1, w: 3, h: 0.28, fontFace: BODY, fontSize: 10.5, color: FOOTC, margin: 0 });
  card(s, MX, 4.75, 11.9, 1.1, BG2, GOOD);
  s.addText("所以真正的差異是「索引結構與排程策略」（radix tree + cache-aware 排序 vs 雜湊表），不是「要不要分頁」。兩邊功能持續趨同——別把某一次 benchmark 當永久結論。",
    { x: MX + 0.28, y: 4.75, w: 11.3, h: 1.1, valign: "middle", fontFace: HEAD, fontSize: 14, bold: true, color: GOOD, lineSpacingMultiple: 1.35, margin: 0 });
  footer(s, P2);
})();

// ============================================================ 12 continuous batching
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "11", "撐大 B 的另一隻腳：continuous batching", COMP);
  s.addText("分頁 + radix 解決了「放得下多少條」，但還有一個浪費：靜態批次要整批等最慢的那條生完。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  card(s, MX, 1.8, 11.9, 1.85, BG2, WARN);
  s.addText("靜態批次（request-level）", { x: MX + 0.25, y: 1.9, w: 5, h: 0.32, fontFace: HEAD, fontSize: 13.5, bold: true, color: WARN, margin: 0 });
  s.addText("整批等最慢的那個 → GPU 大半時間在空轉", { x: MX + 5.4, y: 1.9, w: 6.1, h: 0.32, align: "right", fontFace: BODY, fontSize: 11.5, color: WARN, margin: 0 });
  [[[0.15, COMP, "R1"], [0.85, BG3, "空轉（等 R4 生完）"]],
  [[0.3, COMP, "R2"], [0.7, BG3, "空轉"]],
  [[0.45, COMP, "R3"], [0.55, BG3, "空轉"]],
  [[1.0, COMP, "R4（2000 token，最慢）"]]].forEach((segs, i) => timeline(s, MX + 1.7, 2.34 + i * 0.34, 9.6, segs, `R${i + 1}`, MUTE));

  card(s, MX, 3.82, 11.9, 1.95, BG2, GOOD);
  s.addText("Continuous batching（iteration-level）", { x: MX + 0.25, y: 3.92, w: 6, h: 0.32, fontFace: HEAD, fontSize: 13.5, bold: true, color: GOOD, margin: 0 });
  s.addText("做完就退場還 block，佇列有人立刻補進來", { x: MX + 6.4, y: 3.92, w: 5.1, h: 0.32, align: "right", fontFace: BODY, fontSize: 11.5, color: GOOD, margin: 0 });
  [[[0.15, COMP, "R1"], [0.35, GOOD, "R5 補位"], [0.5, MEM, "R8"]],
  [[0.3, COMP, "R2"], [0.4, GOOD, "R6"], [0.3, MEM, "R9"]],
  [[0.45, COMP, "R3"], [0.55, GOOD, "R7"]],
  [[1.0, COMP, "R4"]]].forEach((segs, i) => timeline(s, MX + 1.7, 4.34 + i * 0.34, 9.6, segs, `slot ${i + 1}`, MUTE));
  takeaway(s, "分頁撐大「B 的上限」、continuous batching 撐滿「B 的實際值」、radix 省掉「根本不必算的部分」。三隻腳。", COMP);
  footer(s, P2);
})();

// ============================================================ 13 收益邊界
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "12", "誠實話：RadixAttention 不是萬靈丹", WARN);
  s.addText("它的收益完全取決於一件事——你的請求之間，前綴到底重不重合。", { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  card(s, MX, 1.85, 5.8, 3.0, BG2, GOOD);
  s.addText("高收益", { x: MX + 0.28, y: 1.97, w: 5.2, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: GOOD, margin: 0 });
  ["RAG（同幾份文件反覆進 prompt）", "多輪對話（每輪帶整段歷史）", "Few-shot（同一批範例）", "Agent（工具描述 + 前面所有步驟）", "sgl.fork() 的分支（天然同前綴）"]
    .forEach((t, i) => s.addText("· " + t, { x: MX + 0.28, y: 2.42 + i * 0.44, w: 5.2, h: 0.4, valign: "middle", fontFace: BODY, fontSize: 11.8, color: MUTE, margin: 0 }));

  card(s, 7.0, 1.85, 5.6, 3.0, BG2, WARN);
  s.addText("低收益", { x: 7.28, y: 1.97, w: 5.0, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: WARN, margin: 0 });
  ["客服：各種不相干的問題", "批次翻譯：每篇文章都不同", "一次性摘要任務", "→ 此時 SGLang 與 vLLM 吞吐差距在 5% 內"]
    .forEach((t, i) => s.addText((i === 3 ? "" : "· ") + t, { x: 7.28, y: 2.42 + i * 0.5, w: 5.0, h: 0.45, valign: "middle", fontFace: BODY, fontSize: 11.8, color: i === 3 ? WARN : MUTE, bold: i === 3, margin: 0 }));

  card(s, MX, 5.05, 11.9, 0.85, BG2, MEM);
  s.addText("選框架看流量長相，不是看誰的 benchmark 數字大：前綴共用 >60%（agent／多輪／RAG）→ SGLang 的 TTFT 通常低 20–40%；每個 prompt 都獨立 → 兩者差 <5%，那就選生態成熟度。",
    { x: MX + 0.28, y: 5.05, w: 11.3, h: 0.85, valign: "middle", fontFace: BODY, fontSize: 12.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  footer(s, P2);
})();

// ============================================================ 14 互動環節
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "13", "互動環節：把「有效 batch」變成看得到的東西", GOOD);
  s.addText("開啟 interactive/serving_map.html —— 本堂只用模式 1–4（模式 5 PD 分離留給第四堂）。",
    { x: MX, y: 1.4, w: 11.9, h: 0.35, fontFace: BODY, fontSize: 13.5, color: MUTE, margin: 0 });
  const layers = [
    ["1 Naive", "靜態批次 + 連續預留 KV：時間軸有多空、HBM 有多浪費", WARN, "有效 batch ≈ 20 → 利用率 6.8%"],
    ["2 Continuous batching", "做完就換人上 → 時間軸被填滿，但 KV 還是預留制", COMP, "有效 batch ≈ 60 → 20%"],
    ["3 Paged KV", "block 化 + 用完才要 → 同 HBM 放得下 ~4 倍的請求", MEM, "有效 batch ≈ 240 → 81%"],
    ["4 Radix 前綴共用", "共用 prompt 的請求合併成一棵樹 → prefill 直接省掉", PURP, "同上，且 TTFT ↓20–40%"],
  ];
  layers.forEach(([t, d, c, n], i) => {
    const y = 1.95 + i * 0.82;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 11.9, h: 0.7, rectRadius: 0.06, fill: { color: BG2 }, line: { color: c, width: 1 } });
    s.addText(t, { x: MX + 0.22, y, w: 2.9, h: 0.7, valign: "middle", fontFace: HEAD, fontSize: 13, bold: true, color: c, margin: 0 });
    s.addText(d, { x: MX + 3.2, y, w: 5.6, h: 0.7, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
    s.addText(n, { x: MX + 8.9, y, w: 2.8, h: 0.7, align: "right", valign: "middle", fontFace: MONO, fontSize: 10.5, bold: true, color: c, margin: 0 });
  });
  s.addText("操作：數字鍵 1–4 切換 · 右側面板同步顯示有效 batch、KV 浪費比例、算術強度與 roofline 上的位置",
    { x: MX, y: 5.35, w: 11.9, h: 0.35, align: "center", fontFace: MONO, fontSize: 11, color: FOOTC, margin: 0 });
  takeaway(s, "從 6.8% 走到 81%——這三步就是問題②的全部價值。", GOOD);
  footer(s, P2);
})();

// ============================================================ 15 問題③
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "14", "問題③：輸出格式不可控（而且檢查很慢）", PURP);
  probStepper(s, 2);
  problemBanner(s, "③", "只靠 prompt 說「請輸出 JSON」，模型不保證遵守",
    "常見翻車：前後多加解釋文字、欄位型別錯、語法小錯（多逗號／單引號）、幻覺出 schema 沒有的欄位。而 agent／function calling 一旦解析失敗，整條鏈就斷了。天真的補救是每步生成後用正則檢查再重試——CPU 成本高得離譜，還是不保證對。", PURP);
  const fails = [["多餘文字", '"當然！以下是您要的 JSON："', WARN], ["型別錯誤", '{"age": "twenty"}', WARN], ["語法錯誤", "多逗號、單引號、缺括號", WARN], ["幻覺欄位", "schema 裡根本沒有的 key", WARN]];
  fails.forEach(([t, d, c], i) => {
    const x = MX + i * 3.0;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.1, w: 2.85, h: 1.0, rectRadius: 0.06, fill: { color: BG2 }, line: { color: c, width: 1 } });
    s.addText(t, { x: x + 0.15, y: 4.2, w: 2.55, h: 0.32, fontFace: HEAD, fontSize: 12.5, bold: true, color: c, margin: 0 });
    s.addText(d, { x: x + 0.15, y: 4.53, w: 2.55, h: 0.5, valign: "top", fontFace: MONO, fontSize: 10, color: MUTE, lineSpacingMultiple: 1.2, margin: 0 });
  });
  takeaway(s, "澄清：這不需要另一個小模型來審核——它純粹是符號計算／規則引擎的問題。", PURP);
  footer(s, P3);
})();

// ============================================================ 16 解法③ FSM
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "15", "解法③：把合法性變成物理限制", PURP);
  s.addText("生成每個 token 前，先算出「此刻合法的 token 集合」，把不合法的機率壓成 −∞ → 物理上不可能生成非法內容。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  const steps = [
    ["① 預編譯成 FSM", "把 JSON schema / 正則編成有限狀態機。SGLang 用 compressed FSM，vLLM 預設 XGrammar", COMP],
    ["② 非同步編譯", "編譯要時間 → 請求先進 WAITING_FOR_FSM，編好才轉 WAITING，不阻塞其他請求", MEM],
    ["③ 位元遮罩", "每步查表得出合法集合，把其餘 token 的 logit 設成 −∞，再於合法子集重新歸一化採樣", PURP],
    ["④ 確定路徑跳躍", '若 FSM 上某段路徑唯一確定（如 {"name": " 必然出現），一次吐掉多個 token，不必逐 token 推理', GOOD],
  ];
  steps.forEach(([t, d, c], i) => {
    const y = 1.85 + i * 1.02;
    card(s, MX, y, 11.9, 0.9, BG2, c);
    s.addText(t, { x: MX + 0.25, y, w: 3.0, h: 0.9, valign: "middle", fontFace: HEAD, fontSize: 14, bold: true, color: c, margin: 0 });
    s.addText(d, { x: MX + 3.35, y, w: 8.3, h: 0.9, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
  });
  takeaway(s, "④ 是反直覺的一點：加了語法約束之後，生成甚至可能比自由生成更快——有些 token 根本不用推理。", PURP);
  footer(s, P3);
})();

// ============================================================ 17 正交性
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "16", "問題② 與 ③ 的關係：正交，但有一條紅線", GOOD);
  card(s, MX, 1.5, 5.8, 2.2, BG2, MEM);
  s.addText("RadixAttention 複用的是「算力」", { x: MX + 0.28, y: 1.62, w: 5.2, h: 0.35, fontFace: HEAD, fontSize: 14.5, bold: true, color: MEM, margin: 0 });
  s.addText("避免重複的矩陣運算。作用在「這段 token 的中間結果」上。", { x: MX + 0.28, y: 2.02, w: 5.2, h: 0.6, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  pill(s, MX + 0.28, 2.95, 5.2, 0.5, "與請求無關 → 可共享", MEM, MEMTINT, MEM, 12);

  card(s, 7.0, 1.5, 5.6, 2.2, BG2, PURP);
  s.addText("約束生成裁剪的是「候選空間」", { x: 7.28, y: 1.62, w: 5.0, h: 0.35, fontFace: HEAD, fontSize: 14.5, bold: true, color: PURP, margin: 0 });
  s.addText("決定哪些 token 允許被選。作用在「這個請求走到語法的哪裡」上。", { x: 7.28, y: 2.02, w: 5.0, h: 0.6, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  pill(s, 7.28, 2.95, 5.0, 0.5, "屬於請求本身 → 不可共享", PURP, PURPTINT, PURP, 12);

  card(s, MX, 3.95, 11.9, 1.05, BG2, WARN);
  s.addText("⚠️ 紅線：即使兩個請求共享同一段前綴快取，它們各自的 FSM 狀態仍必須獨立推進，不能互相干擾。",
    { x: MX + 0.28, y: 3.95, w: 11.3, h: 1.05, valign: "middle", fontFace: HEAD, fontSize: 15, bold: true, color: WARN, margin: 0 });

  card(s, MX, 5.15, 11.9, 0.8, BG2, GOOD);
  s.addText("這條紅線給了一個通用判準：可以快取的是「計算結果」，不可以快取的是「請求狀態」。之後看任何快取設計，先問這個問題。",
    { x: MX + 0.28, y: 5.15, w: 11.3, h: 0.8, valign: "middle", fontFace: BODY, fontSize: 13, color: GOOD, margin: 0 });
  footer(s, P3);
})();

// ============================================================ 18 問題④
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "17", "問題④：GPU 快到讓 CPU 變成瓶頸", GOOD);
  probStepper(s, 3);
  problemBanner(s, "④", "一步只花 5–10 ms，Python 那邊來得及嗎？",
    "前面三個問題解完，batch 撐起來了、重複計算省掉了。這時每個 forward step 只剩幾毫秒，而 tokenize、排程決策、取樣、序列化、數百次 kernel launch 全在 CPU 上——GPU 開始出現「算完在等 CPU 決定下一步」的空窗。", GOOD);
  const items = [
    ["CUDA Graph", "把整串 kernel launch 錄成 DAG，之後直接 replay → 省掉每步數百次 launch", MEM],
    ["Zero-overhead scheduler", "SGLang：把 CPU 排程完全藏進上一步的 GPU 執行時間裡", GOOD],
    ["多進程 + async 排程", "vLLM V1：EngineCore 獨立進程只跑排程＋執行，tokenize／串流輸出與它重疊", COMP],
  ];
  items.forEach(([t, d, c], i) => {
    const y = 4.05 + i * 0.62;
    s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 11.9, h: 0.55, fill: { color: BG2 }, line: { color: c, width: 1 } });
    s.addText(t, { x: MX + 0.22, y, w: 3.4, h: 0.55, valign: "middle", fontFace: HEAD, fontSize: 13, bold: true, color: c, margin: 0 });
    s.addText(d, { x: MX + 3.75, y, w: 8.0, h: 0.55, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
  });
  takeaway(s, "vLLM V1 相對 V0 吞吐提升 ~1.7×，完全來自 CPU 開銷削減——一個 GPU kernel 都沒改。", GOOD);
  footer(s, P4);
})();

// ============================================================ 19 TTFT vs ITL
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "18", "問題④的雙胞胎：TTFT 與 ITL 互相打架", WARN);
  tableGrid(s, MX, 1.4, 11.9, [
    { t: "指標", w: 2.0 }, { t: "全名", w: 2.6 }, { t: "被誰決定", w: 3.3 }, { t: "想要它小，你會想…", w: 4.0 },
  ], [
    ["TTFT", "Time To First Token", "prefill（compute-bound）", "讓 prefill 立刻插隊、獨佔 GPU"],
    ["ITL", "Inter-Token Latency", "decode（memory-bound）", "別讓 prefill 打斷 decode"],
  ], COMP, 11.5);
  card(s, MX, 3.0, 11.9, 1.35, BG2, WARN);
  s.addText("衝突點：head-of-line blocking", { x: MX + 0.28, y: 3.1, w: 6, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: WARN, margin: 0 });
  timeline(s, MX + 2.0, 3.5, 9.3, [[0.55, COMP, "一個 32K prompt 的 prefill（0.5–2 秒）"], [0.45, MEM, "decode"]], "GPU", INK);
  s.addText("↑ 這段時間裡所有串流輸出的使用者都卡住——畫面上就是「打字打到一半停住」", { x: MX + 2.0, y: 3.92, w: 9.3, h: 0.3, fontFace: BODY, fontSize: 11, color: WARN, margin: 0 });

  card(s, MX, 4.55, 11.9, 1.35, BG2, GOOD);
  s.addText("解法：chunked prefill —— 每步固定 token 預算，剩下的預算拿去混 decode", { x: MX + 0.28, y: 4.65, w: 8, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: GOOD, margin: 0 });
  [1, 2, 3].forEach((n, i) => timeline(s, MX + 2.0, 5.05 + i * 0.27, 9.3, [[0.45, COMP, `prefill chunk ${n}/8`], [0.55, MEM, "同批 decode（12 條在跑）"]], `step ${n}`, MUTE));
  takeaway(s, "附帶好處：compute-bound 的 prefill 與 memory-bound 的 decode 混同一批，兩種瓶頸互補，利用率反而更高。", COMP);
  footer(s, P4);
})();

// ============================================================ 20 天花板：投機解碼
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "19", "單機的天花板①：投機解碼 / MTP", COMP);
  s.addText("前面四個問題解完，B 撐大了。但「單一請求的延遲」仍被『讀一遍權重』鎖死——要打破它只有這條路。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  [["Draft", "便宜地提出 k 個候選：小模型 / n-gram / Medusa 頭 / EAGLE / 模型自帶的 MTP head", MEM],
  ["Verify", "大模型把「context + k 個草稿 token」在一次 forward 裡跑完", COMP],
  ["Accept", "由左而右比對機率：large ≥ draft 就收；否則按 large/draft 機率收。第一個被拒就停，並免費得到第 k+1 個 token", GOOD]]
    .forEach(([t, d, c], i) => {
      const x = MX + i * 4.03;
      card(s, x, 1.85, 3.83, 1.75, BG2, c);
      s.addText(t, { x: x + 0.2, y: 1.95, w: 3.4, h: 0.38, fontFace: HEAD, fontSize: 15, bold: true, color: c, margin: 0 });
      s.addText(d, { x: x + 0.2, y: 2.38, w: 3.45, h: 1.1, valign: "top", fontFace: BODY, fontSize: 10.8, color: MUTE, lineSpacingMultiple: 1.25, margin: 0 });
      if (i < 2) s.addText("▶", { x: x + 3.85, y: 2.5, w: 0.2, h: 0.4, fontFace: BODY, fontSize: 14, color: FOOTC, margin: 0 });
    });
  card(s, MX, 3.8, 11.9, 1.9, BG2, COMP);
  s.addText("為什麼會賺（用第 4 頁的同一把尺）", { x: MX + 0.28, y: 3.9, w: 6, h: 0.35, fontFace: HEAD, fontSize: 14.5, bold: true, color: COMP, margin: 0 });
  s.addText("驗證 k 個草稿 token 的 forward：權重讀取 = 1 次（不變），FLOPs = k 倍　→　AI 從 1 變成 k",
    { x: MX + 0.28, y: 4.32, w: 11.3, h: 0.4, valign: "middle", fontFace: MONO, fontSize: 14, bold: true, color: INK, margin: 0 });
  s.addText("＝用閒置算力換頻寬。這是唯一能在「不增加 batch」的前提下改善單請求延遲的招式 → 低流量、互動式、本機部署特別有效。\n輸出分佈嚴格不變（rejection sampling 保證），不是近似加速。代價：接受率低時純虧。",
    { x: MX + 0.28, y: 4.75, w: 11.3, h: 0.85, valign: "top", fontFace: BODY, fontSize: 11.8, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  takeaway(s, "DeepSeek-V3 的 MTP head 報告第二 token 接受率 ~85–90% → 訓練時的輔助目標，推論時直接當草稿模型。", COMP);
  footer(s, P5);
})();

// ============================================================ 21 天花板：量化
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "20", "單機的天花板②：量化——直接砍分母", MEM);
  s.addText("AI = FLOPs ÷ Bytes。前面全在動分子，量化直接動分母——decode 被權重讀取綁死，權重減半 ≈ 延遲減半。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  [["權重", "fp16 → FP8（÷2）→ FP4 / MXFP4（÷4）", "decode 步時直接等比下降", COMP],
  ["KV cache", "fp16 → FP8 → 更低", "KV 變小 ⇒ 同 HBM 放得下更多序列 ⇒ B 又能更大", MEM],
  ["Activation", "FP8 / MXFP8", "決定 GEMM 能不能走 FP8/FP4 tensor core 路徑", PURP]]
    .forEach(([t, d, r, c], i) => {
      const y = 1.85 + i * 1.05;
      card(s, MX, y, 11.9, 0.92, BG2, c);
      s.addText(t, { x: MX + 0.25, y, w: 1.9, h: 0.92, valign: "middle", fontFace: HEAD, fontSize: 15, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 2.25, y, w: 4.2, h: 0.92, valign: "middle", fontFace: MONO, fontSize: 11.5, color: INK, margin: 0 });
      s.addText(r, { x: MX + 6.6, y, w: 5.1, h: 0.92, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
    });
  card(s, MX, 5.05, 11.9, 0.85, BG2, GOOD);
  s.addText("硬體對齊：Hopper 有 FP8 tensor core、Blackwell 有 FP4。精度格式是硬體規格表上的一行，直接決定模型能跑多快——第五堂會看到模型端已經開始「出廠就是 4-bit」。",
    { x: MX + 0.28, y: 5.05, w: 11.3, h: 0.85, valign: "middle", fontFace: BODY, fontSize: 12.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  footer(s, P5);
})();

// ============================================================ 22 單機篇彙整
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "21", "單機篇彙整：問題 → 解法 → 宣稱效果", MEM);
  tableGrid(s, MX, 1.45, 11.9, [
    { t: "問題", w: 3.0 }, { t: "解法", w: 3.6 }, { t: "擋住 B 的哪一段被拆掉", w: 3.2 }, { t: "宣稱效果", w: 2.1 },
  ], [
    ["① 程式難平行", "前端 DSL（執行圖）", "根本沒有並行請求進來", "—"],
    ["② KV 預留浪費", "分頁式 KV（block table）", "放不下更多條", "浪費 <4%、吞吐 2–4×"],
    ["② 前綴重算", "RadixAttention（前綴樹 + LRU）", "算了不必算的東西", "共用場景最高 6.4×"],
    ["② 靜態批次空等", "Continuous batching", "有效 B 遠小於名目 B", "GPU 幾乎不留空隙"],
    ["③ 輸出不可控", "Compressed FSM + 位元遮罩", "CPU 上的序列檢查拖慢整批", "格式 100% 保證"],
    ["④ CPU 成瓶頸", "zero-overhead scheduler / CUDA Graph", "GPU 等 CPU 的空窗", "vLLM V1 較 V0 ~1.7×"],
    ["④' TTFT vs ITL", "Chunked prefill", "長 prompt 卡住 decode", "兩指標同時可控"],
    ["天花板", "投機解碼·MTP、量化", "單請求延遲 / 搬的位元組數", "AI 從 1 → k；權重 ÷2~÷4"],
  ], MEM, 10.5),
    takeaway(s, "全部都在做同一件事：把 roofline 上的那個點，從左邊的記憶體牆推向右邊。", MEM);
  footer(s, P5);
})();

// ============================================================ 23 預告第四堂
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "22", "但這一切，都只發生在「一台機器」裡", PURP);
  s.addText("問題 ①–④ 解完，單一副本已經被榨到接近極限。接下來的問題全都來自「不只一台」。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  const nexts = [
    ["⑤", "大 MoE 單機放不下", "DeepSeek 671B 一顆 HBM 裝不下 → 專家散到幾十上百張卡（EP），但 all-to-all 通訊與專家負載不均隨之而來", MEM],
    ["⑥", "prefill 與 decode 互擾", "chunked prefill 只是緩解。乾脆拆成兩群機器，各自調到最佳操作點——代價是 KV 要跨機器搬", COMP],
    ["⑦", "多副本怎麼分流", "每台各有自己的 radix tree。集中請求→命中率高但單點過熱；打散→負載均勻但命中率崩潰。此消彼長", PURP],
    ["⑧", "掛了怎麼辦", "副本當機時，進行中的請求與它佔用的 KV cache 該怎麼處理", WARN],
  ];
  nexts.forEach(([n, t, d, c], i) => {
    const y = 1.85 + i * 1.05;
    card(s, MX, y, 11.9, 0.92, BG2, c);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.2, y: y + 0.16, w: 0.6, h: 0.6, rectRadius: 0.1, fill: { color: c }, line: { type: "none" } });
    s.addText(n, { x: MX + 0.2, y: y + 0.16, w: 0.6, h: 0.6, align: "center", valign: "middle", fontFace: MONO, fontSize: 18, bold: true, color: BG, margin: 0 });
    s.addText(t, { x: MX + 1.0, y, w: 2.9, h: 0.92, valign: "middle", fontFace: HEAD, fontSize: 14, bold: true, color: c, margin: 0 });
    s.addText(d, { x: MX + 4.0, y, w: 7.7, h: 0.92, valign: "middle", fontFace: BODY, fontSize: 11.2, color: MUTE, margin: 0 });
  });
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 6.1, w: 11.9, h: 0.6, fill: { color: PURPTINT }, line: { color: PURP, width: 1.5 } });
  s.addText("第四堂 · SGLang 多機篇：從一台到一群", { x: MX, y: 6.1, w: 11.9, h: 0.6, align: "center", valign: "middle", fontFace: HEAD, fontSize: 17, bold: true, color: PURP, margin: 0 });
  footer(s, P5);
})();

// ============================================================ 24 帶走三句話
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "23", "帶走三句話", COMP);
  [["1", "每個機制都是被一個具體痛點逼出來的", "DSL 因為程式難平行、RadixAttention 因為 DSL 製造了大量共享前綴、FSM 因為 agent 需要能解析的輸出、zero-overhead scheduler 因為前三個解完 GPU 快到 CPU 跟不上。順著問題走，就不用背功能表。", COMP],
  ["2", "所有解法都在拆掉擋住 batch 的石頭", "batch=1 的 decode 只用 0.3% 算力，AI ≈ B。分頁撐大 B 的上限、continuous batching 撐滿實際值、radix 省掉不必算的、CPU 優化不讓時間軸留空。一個 kernel 都沒改。", MEM],
  ["3", "先看流量長相，再選框架", "RadixAttention 的收益完全取決於前綴重合度。前綴共用 >60%（agent／多輪／RAG）SGLang 的 TTFT 低 20–40%；請求彼此無關時兩者差 <5%。別把某一次 benchmark 當永久結論。", GOOD]]
    .forEach(([n, t, d, c], i) => {
      const y = 1.85 + i * 1.45;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 0.85, h: 0.85, rectRadius: 0.12, fill: { color: c }, line: { type: "none" } });
      s.addText(n, { x: MX, y, w: 0.85, h: 0.85, align: "center", valign: "middle", fontFace: MONO, fontSize: 30, bold: true, color: BG, margin: 0 });
      s.addText(t, { x: MX + 1.1, y, w: 11.0, h: 0.5, fontFace: HEAD, fontSize: 18, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 1.1, y: y + 0.52, w: 11.2, h: 0.85, valign: "top", fontFace: BODY, fontSize: 12.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
    });
  s.addText("下一步：拿一個小模型在自己的卡上量「batch sweep 的吞吐曲線」與「開/關 prefix caching 的 TTFT」——把問題②驗一次。",
    { x: MX, y: 6.25, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 12.5, color: FOOTC, margin: 0 });
  footer(s, P5);
})();

pres.writeFile({ fileName: "../class3_sglang_single_node.pptx" }).then((f) => console.log("✅ 產生：" + f + "（" + PAGE + " 頁）")).catch((e) => console.error(e));
