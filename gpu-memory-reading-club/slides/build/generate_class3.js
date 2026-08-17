// 第三堂課 — 推論引擎單機篇：八個問題，兩種解法（SGLang × vLLM）
// 產生 ../class3_engine_single_node.pptx。沿用系列的深色「矽晶」主題。
//
// 主幹 = 問題導向。那八個問題不是 SGLang 專有的，是任何推論引擎都會撞到的；
//        SGLang 的發展史剛好把它們依序列了出來。每個問題都對比兩家的「寫法」。
//   地基   decode 是 memory-bound → AI ≈ B → 所有問題都在拆掉擋住 batch 的石頭
//   問題①  LLM 程式難寫又跑不快   → SGLang：前端 DSL｜vLLM：沒有對應物（職責邊界不同）
//   問題②  共享前綴被反覆重算      → 共同：分頁 KV｜索引：radix tree vs 鏈式雜湊表｜排程：主動 vs 被動
//   問題③  結構化輸出不可控又慢    → 共同：XGrammar + 位元遮罩｜差異：jump-forward decoding
//   問題④  CPU 排程吃掉 GPU 時間   → zero-overhead scheduler vs 多進程 + async（兩條路同一目標）
//   天花板 投機解碼·MTP、量化（兩家都有）
// 問題 ⑤–⑧（多機）留給第四堂。
const pptxgen = require("pptxgenjs");

const BG = "0E1726", BG2 = "16233A", BG3 = "1C2E4A";
const INK = "EAF1FB", MUTE = "8FA6C4", LINE = "2A3D5C", FOOTC = "5C7299";
const MEM = "38BDF8", COMP = "F59E0B", WARN = "FB7185", GOOD = "34D399", PURP = "A78BFA";
const MEMTINT = "10455F", COMPTINT = "4A3410", WARNTINT = "4A2433", GOODTINT = "123D31", PURPTINT = "2A2150";
const HEAD = "PingFang TC", BODY = "PingFang TC", MONO = "Menlo";
// 兩家的固定色：SGLang = purple、vLLM = cyan
const SG = PURP, SGT = PURPTINT, VL = MEM, VLT = MEMTINT;

const W = 13.33, H = 7.5, MX = 0.7, TITLE_Y = 0.62, FOOT_Y = 7.05, TOTAL = 28;
const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 });

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "GPU 記憶體與資料搬遷讀書會";
pres.title = "第三堂課 · 推論引擎單機篇：SGLang × vLLM";

let PAGE = 0;
const base = (s) => { s.background = { color: BG }; PAGE += 1; };
function runningHeader(s) {
  s.addText("讀書會 · 第三堂課 · 單機篇 SGLang × vLLM", { x: W - 6.1, y: 0.3, w: 5.4, h: 0.3, align: "right", fontFace: BODY, fontSize: 10, color: MUTE, margin: 0 });
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
function arrow(s, x, y, w, color, label) {
  s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width: 2.5, endArrowType: "triangle" } });
  if (label) s.addText(label, { x: x - 0.5, y: y + 0.07, w: w + 1.0, h: 0.3, align: "center", fontFace: MONO, fontSize: 10, color, margin: 0 });
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
    s.addText(c.t, { x: cx + 0.08, y, w: c.w - 0.16, h: hh, valign: "middle", fontFace: HEAD, fontSize: (fs || 11) + 0.5, bold: true, color: c.c || INK, margin: 0 });
    cx += c.w;
  });
  rows.forEach((r, ri) => {
    cx = x;
    r.forEach((cell, ci) => {
      s.addShape(pres.shapes.RECTANGLE, { x: cx, y: y + hh + ri * rh, w: cols[ci].w, h: rh, fill: { color: ri % 2 ? BG2 : BG }, line: { color: LINE, width: 0.8 } });
      s.addText(cell, { x: cx + 0.08, y: y + hh + ri * rh, w: cols[ci].w - 0.16, h: rh, valign: "middle", fontFace: ci === 0 ? HEAD : BODY, fontSize: fs || 11, bold: ci === 0, color: ci === 0 ? accent : cols[ci].c || MUTE, margin: 0 });
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

// ── 問題導覽列：本堂 ①–④，第四堂 ⑤–⑧（淡出）
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
function problemBanner(s, num, problem, why, accent) {
  card(s, MX, 2.35, 11.9, 1.5, BG2, accent);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.25, y: 2.72, w: 0.75, h: 0.75, rectRadius: 0.12, fill: { color: accent }, line: { type: "none" } });
  s.addText(num, { x: MX + 0.25, y: 2.72, w: 0.75, h: 0.75, align: "center", valign: "middle", fontFace: MONO, fontSize: 26, bold: true, color: BG, margin: 0 });
  s.addText(problem, { x: MX + 1.25, y: 2.5, w: 10.3, h: 0.5, valign: "middle", fontFace: HEAD, fontSize: 20, bold: true, color: INK, margin: 0 });
  s.addText(why, { x: MX + 1.25, y: 3.02, w: 10.3, h: 0.7, valign: "top", fontFace: BODY, fontSize: 12.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
}
// ── 兩家對比版面：左 SGLang、右 vLLM
function versus(s, y, h, sgSub, sgLines, vlSub, vlLines) {
  card(s, MX, y, 5.8, h, BG2, SG);
  pill(s, MX + 0.25, y + 0.16, 1.65, 0.44, "SGLang", SG, SGT, SG, 13);
  s.addText(sgSub, { x: MX + 2.05, y: y + 0.16, w: 3.5, h: 0.44, valign: "middle", fontFace: HEAD, fontSize: 12.5, bold: true, color: INK, margin: 0 });
  sgLines.forEach((t, i) => s.addText("· " + t, { x: MX + 0.28, y: y + 0.7 + i * 0.44, w: 5.25, h: 0.42, valign: "middle", fontFace: BODY, fontSize: 11.3, color: MUTE, margin: 0 }));
  card(s, 7.0, y, 5.6, h, BG2, VL);
  pill(s, 7.25, y + 0.16, 1.5, 0.44, "vLLM", VL, VLT, VL, 13);
  s.addText(vlSub, { x: 8.9, y: y + 0.16, w: 3.5, h: 0.44, valign: "middle", fontFace: HEAD, fontSize: 12.5, bold: true, color: INK, margin: 0 });
  vlLines.forEach((t, i) => s.addText("· " + t, { x: 7.28, y: y + 0.7 + i * 0.44, w: 5.05, h: 0.42, valign: "middle", fontFace: BODY, fontSize: 11.3, color: MUTE, margin: 0 }));
}
function verdict(s, y, text, color) {
  card(s, MX, y, 11.9, 0.85, BG2, color);
  s.addText(text, { x: MX + 0.28, y, w: 11.3, h: 0.85, valign: "middle", fontFace: HEAD, fontSize: 13.5, bold: true, color: color, margin: 0 });
}

const P0 = "讀書會 · 第三堂課";
const PA = "地基 · 為什麼這些問題都指向同一件事";
const P1 = "問題① 程式難平行";
const P2 = "問題② 前綴重算";
const P3 = "問題③ 輸出不可控";
const P4 = "問題④ CPU 成瓶頸";
const P5 = "單機的天花板";

// ============================================================ 1 標題
(() => {
  const s = pres.addSlide(); base(s);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.08, h: H, fill: { color: SG }, line: { type: "none" } });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.08, y: 0, w: 0.08, h: H, fill: { color: VL }, line: { type: "none" } });
  s.addText("第三堂課 · 推論引擎單機篇", { x: MX + 0.3, y: 1.45, w: 8, h: 0.45, fontFace: MONO, fontSize: 15, color: COMP, margin: 0 });
  s.addText("八個問題，兩種解法", { x: MX + 0.3, y: 1.95, w: 10.5, h: 0.85, fontFace: HEAD, fontSize: 42, bold: true, color: INK, margin: 0 });
  s.addText([{ text: "SGLang", options: { color: SG, bold: true } }, { text: "  ×  ", options: { color: MUTE } }, { text: "vLLM", options: { color: VL, bold: true } }],
    { x: MX + 0.3, y: 2.85, w: 10.5, h: 0.7, fontFace: HEAD, fontSize: 30, margin: 0 });
  s.addText("① 程式難平行　② 前綴重算　③ 輸出不可控　④ CPU 成瓶頸　　（⑤–⑧ 多機問題見第四堂）",
    { x: MX + 0.3, y: 3.7, w: 11.3, h: 0.4, fontFace: BODY, fontSize: 14, color: COMP, margin: 0 });
  card(s, MX + 0.3, 4.4, 11.3, 1.3, BG2, GOOD);
  s.addText([
    { text: "這堂不照功能表講，照問題講。", options: { bold: true, color: GOOD } },
    { text: "而且這八個問題不是某一家專有的——", options: { color: INK } },
    { text: "任何推論引擎都會撞到，兩家都給了答案。", options: { bold: true, color: INK } },
    { text: "\n所以最好的學法是：一個問題、兩種寫法，對比之後你會看到兩家哲學的差異。", options: { color: MUTE } },
  ], { x: MX + 0.55, y: 4.4, w: 10.8, h: 1.3, valign: "middle", fontFace: HEAD, fontSize: 15, lineSpacingMultiple: 1.35, margin: 0 });
  footer(s, P0);
})();

// ============================================================ 2 八問題全景
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "01", "全景：推論引擎會撞到的八個問題", COMP);
  s.addText("由淺入深，剛好走完「編程模型 → 單機記憶體 → 單機排程 → 多卡 → 多機」。SGLang 的發展史剛好把它們依序列了出來。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  const rows = [
    ["①", "LLM 程式（多次呼叫、分支、工具）難寫又跑不快", "前端 DSL", "—", true],
    ["②", "這些程式天然共享大量前綴，卻被反覆重算", "radix tree + cache-aware 排程", "鏈式雜湊表 APC", true],
    ["③", "結構化輸出逐 token 檢查太慢、格式仍不保證", "XGrammar + jump-forward", "XGrammar（非同步編譯）", true],
    ["④", "GPU 一步只要 5–10 ms，CPU 排程成了瓶頸", "zero-overhead scheduler", "多進程 + async 排程", true],
    ["⑤", "大 MoE（DeepSeek）單機放不下、專家負載不均", "大規模 EP + DeepEP/EPLB", "支援 EP", false],
    ["⑥", "prefill 與 decode 互相干擾（TTFT vs ITL）", "內建 PD 分離", "KV connector 抽象", false],
    ["⑦", "多副本之間：局部性 vs 負載均衡此消彼長", "自帶 cache-aware router", "生態系（Dynamo/llm-d）", false],
    ["⑧", "副本掛掉，進行中的請求與它的 KV 怎麼辦", "—", "—", false],
  ];
  s.addText("SGLang", { x: MX + 5.9, y: 1.78, w: 3.0, h: 0.26, fontFace: HEAD, fontSize: 10.5, bold: true, color: SG, margin: 0 });
  s.addText("vLLM", { x: MX + 9.0, y: 1.78, w: 2.8, h: 0.26, fontFace: HEAD, fontSize: 10.5, bold: true, color: VL, margin: 0 });
  rows.forEach(([n, p, sg, vl, mine], i) => {
    const y = 2.08 + i * 0.52;
    s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 11.9, h: 0.46, fill: { color: mine ? BG2 : BG }, line: { color: mine ? LINE : BG3, width: mine ? 1 : 0.6 } });
    s.addText(n, { x: MX + 0.1, y, w: 0.5, h: 0.46, align: "center", valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: mine ? COMP : FOOTC, margin: 0 });
    s.addText(p, { x: MX + 0.68, y, w: 5.1, h: 0.46, valign: "middle", fontFace: BODY, fontSize: 11, color: mine ? INK : FOOTC, margin: 0 });
    s.addText(sg, { x: MX + 5.9, y, w: 3.0, h: 0.46, valign: "middle", fontFace: HEAD, fontSize: 10.5, bold: mine, color: mine ? SG : FOOTC, margin: 0 });
    s.addText(vl, { x: MX + 9.0, y, w: 2.8, h: 0.46, valign: "middle", fontFace: HEAD, fontSize: 10.5, bold: mine, color: mine ? VL : FOOTC, margin: 0 });
  });
  s.addShape(pres.shapes.LINE, { x: MX, y: 2.08 + 4 * 0.52 - 0.03, w: 11.9, h: 0, line: { color: COMP, width: 1.5, dashType: "dash" } });
  takeaway(s, "線以上是本堂。注意 ① 那一列——vLLM 沒有對應物，那不是缺漏，是職責邊界的選擇。", COMP);
  footer(s, P0);
})();

// ============================================================ 3 兩家哲學（先給結論）
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "02", "先給結論：兩家的哲學差異", GOOD);
  s.addText("這頁的判斷後面每一題都會驗證一次。先放在這裡，聽眾才有掛東西的鉤子。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  card(s, MX, 1.8, 5.8, 3.5, BG2, SG);
  pill(s, MX + 0.25, 1.98, 1.65, 0.46, "SGLang", SG, SGT, SG, 13.5);
  s.addText("「結構化 LLM 程式」的執行引擎", { x: MX + 0.25, y: 2.58, w: 5.3, h: 0.4, fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0 });
  s.addText("第一性問題是：一個會分支、會重試、會反覆用同樣前綴與語法的 LLM 程式，怎麼跑得最快？",
    { x: MX + 0.25, y: 3.0, w: 5.3, h: 0.75, valign: "top", fontFace: BODY, fontSize: 11.8, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  ["DSL 描述執行圖", "radix tree 做前綴索引", "cache-aware 排程主動提高命中率", "jump-forward 跳過確定的 token"]
    .forEach((t, i) => s.addText("· " + t, { x: MX + 0.28, y: 3.85 + i * 0.34, w: 5.25, h: 0.32, valign: "middle", fontFace: BODY, fontSize: 11.2, color: SG, margin: 0 }));

  card(s, 7.0, 1.8, 5.6, 3.5, BG2, VL);
  pill(s, 7.25, 1.98, 1.5, 0.46, "vLLM", VL, VLT, VL, 13.5);
  s.addText("「模型服務」的基礎設施", { x: 7.25, y: 2.58, w: 5.1, h: 0.4, fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0 });
  s.addText("第一性問題是：任何模型、任何硬體、任何部署形態，怎麼都能穩定地服務起來？",
    { x: 7.25, y: 3.0, w: 5.1, h: 0.75, valign: "top", fontFace: BODY, fontSize: 11.8, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  ["模型／硬體後端覆蓋最廣", "KV connector、語法後端皆可插拔", "路由與編排交給生態系", "V1 重寫把 CPU 開銷壓到最低"]
    .forEach((t, i) => s.addText("· " + t, { x: 7.28, y: 3.85 + i * 0.34, w: 5.05, h: 0.32, valign: "middle", fontFace: BODY, fontSize: 11.2, color: VL, margin: 0 }));

  verdict(s, 5.45, "一句話：SGLang 在「前綴與語法重複」這條路上挖得深；vLLM 在「什麼都能跑」這個面上鋪得廣。不是誰比較好，是最佳化的目標函數不同。", GOOD);
  footer(s, P0);
})();

// ============================================================ 4 地基① 0.34%
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "03", "地基①：所有問題的共同根源", WARN);
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
  takeaway(s, "接下來每一個問題，本質上都是「這 99.7% 為什麼填不滿」的不同面向——兩家都在打這件事。", WARN);
  footer(s, PA);
})();

// ============================================================ 5 地基② AI≈B
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "04", "地基②：AI ≈ B，所以答案永遠是「把 batch 撐大撐滿」", COMP);
  card(s, MX, 1.45, 11.9, 1.6, BG2, COMP);
  s.addText("搬：權重還是只讀「一遍」（2N bytes）　　算：2N × B FLOPs　　→　AI ≈ B",
    { x: MX + 0.3, y: 1.58, w: 11.3, h: 0.5, valign: "middle", fontFace: MONO, fontSize: 17, bold: true, color: COMP, margin: 0 });
  s.addText("既然 ridge point ≈ 296，B 要拉到「幾百」GPU 才餵得飽。而 B 撐不大的每一個理由，就是後面每一個問題。",
    { x: MX + 0.3, y: 2.12, w: 11.3, h: 0.45, valign: "middle", fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.3, y: 2.55, w: 11.3, h: 0.42, fill: { color: COMPTINT }, line: { color: COMP, width: 1 } });
  s.addText("Llama-3-8B on H100：batch=1 ≈ 208 tok/s　→　batch=64 ≈ 13,000 tok/s（步時幾乎不變）",
    { x: MX + 0.3, y: 2.55, w: 11.3, h: 0.42, align: "center", valign: "middle", fontFace: MONO, fontSize: 12.5, bold: true, color: COMP, margin: 0 });
  s.addText("四個問題各自擋住 B 的哪一段", { x: MX, y: 3.3, w: 6, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0 });
  [["①", "程式難平行", "根本沒有那麼多請求同時進來——runtime 看不見可平行的分支", COMP],
  ["②", "前綴重算", "HBM 被浪費的 KV 佔住 → 放不下更多條；算力浪費在不必算的東西上", MEM],
  ["③", "輸出不可控", "每步的語法檢查是 CPU 上的序列工作 → 拖慢整批", PURP],
  ["④", "CPU 成瓶頸", "GPU 算完在等 CPU 決定下一步 → 有效 batch 再大也填不滿時間軸", GOOD]]
    .forEach(([n, t, d, c], i) => {
      const y = 3.75 + i * 0.58;
      s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 11.9, h: 0.52, fill: { color: BG2 }, line: { color: c, width: 1 } });
      s.addText(n, { x: MX + 0.12, y, w: 0.5, h: 0.52, align: "center", valign: "middle", fontFace: MONO, fontSize: 14, bold: true, color: c, margin: 0 });
      s.addText(t, { x: MX + 0.72, y, w: 2.2, h: 0.52, valign: "middle", fontFace: HEAD, fontSize: 12.5, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 3.0, y, w: 8.8, h: 0.52, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
    });
  takeaway(s, "把這張圖記住，後面就不會覺得是在背功能表——每個機制都在拆掉擋住 B 的一塊石頭。", COMP);
  footer(s, PA);
})();

// ============================================================ 6 問題①
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "05", "問題①：LLM 程式難寫，而且跑不快", COMP);
  probStepper(s, 0);
  problemBanner(s, "①", "Python 迴圈把「依賴關係」寫死在控制流裡",
    "多次呼叫、分支、工具使用、self-consistency 取樣——這些步驟裡有很多其實可以同時跑，但你用 for 迴圈寫出來之後，runtime 只看得到「一個接一個的請求」，看不見哪幾步彼此無關。加上 Python 的 GIL，前端本身也不擅長真正的並行。", COMP);
  [["寫的人痛", "分支、重試、工具呼叫的膠水程式又臭又長"],
  ["跑的人痛", "runtime 收到的是一串互不相干的請求，無從批處理"],
  ["更痛的是", "這些請求其實高度共享前綴（同一個 system prompt、同一段歷史），但沒人告訴 runtime"]]
    .forEach(([t, d], i) => {
      const x = MX + i * 4.03;
      card(s, x, 4.1, 3.85, 1.55, BG2, WARN);
      s.addText(t, { x: x + 0.2, y: 4.22, w: 3.5, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: WARN, margin: 0 });
      s.addText(d, { x: x + 0.2, y: 4.6, w: 3.5, h: 0.95, valign: "top", fontFace: BODY, fontSize: 11.2, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
    });
  takeaway(s, "SGLang 的名字就是這裡來的：Structured Generation Language——它先是一個語言，才是一個引擎。", COMP);
  footer(s, P1);
})();

// ============================================================ 7 解法① DSL
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "06", "SGLang 的解法：把程式描述成一張執行圖", SG);
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
  card(s, 7.35, 1.8, 5.25, 3.6, BG2, SG);
  s.addText("sgl.fork()：一張圖", { x: 7.6, y: 1.92, w: 4.8, h: 0.32, fontFace: HEAD, fontSize: 13.5, bold: true, color: SG, margin: 0 });
  obox(s, 8.85, 2.35, 2.3, 0.5, "共享 prompt", SG, SG, 11.5);
  varrow(s, 9.4, 2.87, 0.28, SG);
  ["A", "B", "C"].forEach((t, i) => obox(s, 7.7 + i * 1.62, 3.25, 1.45, 0.5, "分支 " + t, GOOD, GOOD, 11.5));
  s.addText("▼ 三條可同時跑", { x: 7.6, y: 3.85, w: 4.8, h: 0.3, align: "center", fontFace: BODY, fontSize: 10.5, color: GOOD, margin: 0 });
  obox(s, 8.85, 4.2, 2.3, 0.5, "彙整", GOOD, GOOD, 11.5);
  s.addText("runtime 看到：一個共享前綴 + 三條可平行分支", { x: 7.6, y: 4.85, w: 4.8, h: 0.4, align: "center", fontFace: BODY, fontSize: 11, color: SG, margin: 0 });
  takeaway(s, "關鍵副作用：fork 產生的分支天然共享同一段前綴——這正好把問題② 端到 RadixAttention 面前。", SG);
  footer(s, P1);
})();

// ============================================================ 8 對比①：vLLM 沒有對應物
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "07", "對比①：vLLM 沒有對應物——而那是刻意的", VL);
  versus(s, 1.4, 2.3,
    "有一層語言", ["sgl.gen / fork / select 描述執行圖", "runtime 看得見分支結構，主動批處理", "fork 的共享前綴直接餵給 radix tree", "代價：要學一套 DSL、綁定這個 runtime"],
    "只有 API", ["Python LLM 類 + OpenAI 相容 server", "平行化交給呼叫端（asyncio、批次提交）", "或交給上層框架（LangChain、DSPy…）", "好處：任何客戶端都能用，零學習成本"]);
  card(s, MX, 3.9, 11.9, 1.85, BG2, GOOD);
  s.addText("為什麼這不是「缺漏」？", { x: MX + 0.28, y: 4.0, w: 6, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: GOOD, margin: 0 });
  s.addText("因為兩家對「引擎的職責邊界」判斷不同。vLLM 認為「怎麼組織 LLM 程式」是應用層的事，引擎只該把單一請求服務好；SGLang 認為程式結構是效能資訊，不交給引擎就浪費了。\n實務上：多數人把 SGLang 當純 server 用、從沒碰過 DSL——所以這一層在日常使用上常常是沒被啟用的器官。但它解釋了 SGLang 為什麼那麼在意前綴共享。",
    { x: MX + 0.28, y: 4.4, w: 11.3, h: 1.3, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.35, margin: 0 });
  takeaway(s, "第一個對比就看到哲學差異：一家往上長出語言，一家往外接生態。後面三題會反覆看到這個分野。", VL);
  footer(s, P1);
})();

// ============================================================ 9 橋
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "08", "但是——好寫的並行程式，會製造出大量重複計算", MEM);
  s.addText("DSL 解決了「看得見平行」，卻立刻暴露下一個問題：那些分支的前綴，難道要各算一遍？",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  card(s, MX, 1.85, 11.9, 1.9, BG2, WARN);
  s.addText("一個 fork 出 8 條分支的程式，system prompt 2,000 token", { x: MX + 0.28, y: 1.97, w: 8, h: 0.35, fontFace: HEAD, fontSize: 14.5, bold: true, color: WARN, margin: 0 });
  timeline(s, MX + 1.9, 2.45, 9.4, [[0.75, COMP, "重算 system prompt（2,000 token prefill）"], [0.25, MEM, "真正新增的內容"]], "分支 1", MUTE);
  timeline(s, MX + 1.9, 2.83, 9.4, [[0.75, COMP, "又算一遍"], [0.25, MEM, "新增"]], "分支 2", MUTE);
  timeline(s, MX + 1.9, 3.21, 9.4, [[0.75, COMP, "又算一遍 ×8…"], [0.25, MEM, "新增"]], "分支 3–8", MUTE);
  s.addText("而且真實流量本來就長這樣——不用 DSL 也一樣", { x: MX, y: 4.0, w: 8, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0 });
  [["RAG", "同幾份文件反覆進 prompt", MEM], ["多輪對話", "每輪都帶著整段歷史", GOOD], ["Few-shot", "同一批範例貼在每個請求前面", PURP], ["Agent", "工具描述 + 前面所有步驟", COMP]]
    .forEach(([t, d, c], i) => {
      const x = MX + i * 3.0;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.45, w: 2.85, h: 1.0, rectRadius: 0.06, fill: { color: BG2 }, line: { color: c, width: 1 } });
      s.addText(t, { x: x + 0.15, y: 4.55, w: 2.55, h: 0.32, fontFace: HEAD, fontSize: 13, bold: true, color: c, margin: 0 });
      s.addText(d, { x: x + 0.15, y: 4.88, w: 2.55, h: 0.5, valign: "top", fontFace: BODY, fontSize: 10.8, color: MUTE, lineSpacingMultiple: 1.2, margin: 0 });
    });
  takeaway(s, "所以問題②是兩家都必須解的——只是 SGLang 因為 DSL 被逼得更早、更徹底。", MEM);
  footer(s, P1);
})();

// ============================================================ 10 問題②
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "09", "問題②：共享前綴被反覆重算", MEM);
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

// ============================================================ 11 共同地基：分頁
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "10", "共同地基：分頁——兩家都這樣放 KV", GOOD);
  s.addText("1960 年代的虛擬記憶體分頁機制搬到 KV cache 上。vLLM 的 PagedAttention 讓它出名，但 SGLang 底下同樣是分頁式記憶體池。",
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
  { text: "■ 請求 C　", options: { color: PURP } }, { text: "■ free pool", options: { color: FOOTC } }],
    { x: 7.2, y: 4.6, w: 5.4, h: 0.3, fontFace: BODY, fontSize: 10.5, margin: 0 });
  [["浪費 60–80% → < 4%", GOOD], ["用完才要、不用連續", MEM], ["請求結束立刻歸還", MEM], ["block 可跨請求共享 ←關鍵", PURP]]
    .forEach(([t, c], i) => pill(s, MX + i * 3.0, 5.1, 2.85, 0.52, t, c, BG2, c, 11.5));
  takeaway(s, "這一層兩家沒有分歧。分歧在下一頁：block 放好之後，「怎麼被找到」。", GOOD);
  footer(s, P2);
})();

// ============================================================ 12 對比②a 索引層
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "11", "對比②a 索引層：前綴樹 vs 鏈式雜湊表", PURP);
  s.addText("同一個問題：「這個新請求的前綴，之前算過嗎？算到哪？」——兩家用了完全不同的資料結構。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  // 左：radix tree
  card(s, MX, 1.8, 5.8, 3.4, BG2, SG);
  pill(s, MX + 0.25, 1.95, 1.65, 0.42, "SGLang", SG, SGT, SG, 12.5);
  s.addText("radix tree（前綴樹）", { x: MX + 2.05, y: 1.95, w: 3.5, h: 0.42, valign: "middle", fontFace: HEAD, fontSize: 13, bold: true, color: INK, margin: 0 });
  obox(s, MX + 1.9, 2.5, 2.4, 0.44, "共用 system prompt", SG, SG, 10.5);
  varrow(s, MX + 2.2, 2.96, 0.34, SG); varrow(s, MX + 3.9, 2.96, 0.34, SG);
  obox(s, MX + 0.5, 3.34, 2.4, 0.42, "對話 A", MEM, MEM, 10.5);
  obox(s, MX + 3.0, 3.34, 2.4, 0.42, "對話 B", GOOD, GOOD, 10.5);
  varrow(s, MX + 1.7, 3.78, 0.3, MEM);
  obox(s, MX + 0.5, 4.1, 2.4, 0.42, "A · 第 2 輪", MEM, MEM, 10.5);
  s.addText("樹狀結構 → 一次 O(D) 最長前綴匹配；分叉點可落在任意 token，不必對齊 block 邊界", { x: MX + 0.28, y: 4.62, w: 5.25, h: 0.5, valign: "top", fontFace: BODY, fontSize: 10.8, color: MUTE, lineSpacingMultiple: 1.25, margin: 0 });
  // 右：hash chain
  card(s, 7.0, 1.8, 5.6, 3.4, BG2, VL);
  pill(s, 7.25, 1.95, 1.5, 0.42, "vLLM", VL, VLT, VL, 12.5);
  s.addText("鏈式雜湊（APC）", { x: 8.9, y: 1.95, w: 3.5, h: 0.42, valign: "middle", fontFace: HEAD, fontSize: 13, bold: true, color: INK, margin: 0 });
  ["blk0", "blk1", "blk2"].forEach((t, i) => {
    obox(s, 7.3 + i * 1.75, 2.5, 1.55, 0.44, t + " (16 tok)", VL, VL, 10);
    if (i < 2) s.addText("→", { x: 8.85 + i * 1.75, y: 2.5, w: 0.2, h: 0.44, align: "center", valign: "middle", fontFace: BODY, fontSize: 12, color: FOOTC, margin: 0 });
  });
  s.addText("hash(前一塊 hash + 本塊 tokens) → 平坦雜湊表，每個 block 各自可尋址", { x: 7.28, y: 3.05, w: 5.05, h: 0.45, valign: "top", fontFace: MONO, fontSize: 10, color: VL, lineSpacingMultiple: 1.25, margin: 0 });
  ["查表 O(1)、V1 常數時間淘汰", "預設開啟；命中率 0% 時幾乎零開銷", "代價：命中必須對齊 16-token 邊界"]
    .forEach((t, i) => s.addText("· " + t, { x: 7.28, y: 3.6 + i * 0.38, w: 5.05, h: 0.36, valign: "middle", fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 }));
  verdict(s, 5.35, "差異的本質：樹能表達「任意長度的共同開頭」，雜湊表只能表達「對齊的區塊」。所以極端共享場景 radix 佔優，一般場景兩者接近。", PURP);
  footer(s, P2);
})();

// ============================================================ 13 對比②b 排程
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "12", "對比②b 排程：主動提高命中率 vs 被動命中", SG);
  s.addText("有了索引還不夠——請求「以什麼順序進來」也會改變命中率。這是兩家差最多的一點。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  card(s, MX, 1.8, 11.9, 1.5, BG2, WARN);
  s.addText("同樣三個請求，順序不同，命中率就不同", { x: MX + 0.28, y: 1.9, w: 6, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: WARN, margin: 0 });
  timeline(s, MX + 2.0, 2.3, 9.3, [[0.34, COMP, "A（前綴 X）"], [0.33, MEM, "B（前綴 Y）"], [0.33, COMP, "C（前綴 X）"]], "亂序", MUTE);
  timeline(s, MX + 2.0, 2.7, 9.3, [[0.34, COMP, "A（前綴 X）"], [0.33, GOOD, "C（前綴 X）命中！"], [0.33, MEM, "B（前綴 Y）"]], "排過", MUTE);
  s.addText("↑ 亂序時 X 的快取可能已被 LRU 淘汰；排在一起就一定命中", { x: MX + 2.0, y: 3.05, w: 9.3, h: 0.28, fontFace: BODY, fontSize: 10.8, color: GOOD, margin: 0 });
  versus(s, 3.45, 2.0,
    "cache-aware 排程", ["主動把同前綴的請求排在一起送", "v0.4 起還有 cache-aware load balancer", "→ 這是「引擎主動經營快取」的做法"],
    "被動命中", ["FCFS / priority 為主，不按內容重排", "命中率取決於流量自然的到達順序", "→ 引擎不猜工作負載，交給上層"]);
  verdict(s, 5.6, "實測（前綴共用重、c=50）：SGLang 的 TTFT p50 低約 37%、p95 低約 41%——差距主要來自這一頁，不只是資料結構。", SG);
  footer(s, P2);
})();

// ============================================================ 14 continuous batching
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "13", "共同地基：continuous batching（撐滿 B 的那隻腳）", COMP);
  s.addText("排程單位從「一個請求」改成「一次 forward」。兩家都有——vLLM V1 更把 prefill/decode 統一成一個排程字典。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
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
  takeaway(s, "三隻腳：分頁撐大「B 的上限」、continuous batching 撐滿「實際值」、前綴複用省掉「不必算的」。", COMP);
  footer(s, P2);
})();

// ============================================================ 15 假對立澄清
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "14", "澄清：「RadixAttention vs PagedAttention」是假對立", GOOD);
  s.addText("網路文章常把兩者寫成競品。它們其實在不同層——而且兩家兩層都有。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  [["索引 / 複用層", "已經放好的 KV block 怎麼被找到並複用？", "SGLang：radix tree（+ cache-aware 排程）　｜　vLLM：鏈式雜湊表 APC", PURP],
  ["記憶體配置層", "KV cache 在 HBM 裡怎麼放？", "兩邊都是分頁式：固定大小 block、不連續、用完才要", MEM]]
    .forEach(([t, q, impl, c], i) => {
      const y = 1.85 + i * 1.45;
      card(s, MX, y, 11.9, 1.25, BG2, c);
      s.addText(t, { x: MX + 0.3, y: y + 0.12, w: 3.0, h: 0.45, valign: "middle", fontFace: HEAD, fontSize: 16, bold: true, color: c, margin: 0 });
      s.addText(q, { x: MX + 3.4, y: y + 0.12, w: 8.2, h: 0.45, valign: "middle", fontFace: BODY, fontSize: 12.5, color: INK, margin: 0 });
      s.addText(impl, { x: MX + 0.3, y: y + 0.62, w: 11.3, h: 0.5, valign: "middle", fontFace: MONO, fontSize: 11, color: MUTE, margin: 0 });
    });
  s.addText("▲ 蓋在上面", { x: MX + 0.3, y: 3.1, w: 3, h: 0.28, fontFace: BODY, fontSize: 10.5, color: FOOTC, margin: 0 });
  verdict(s, 4.75, "所以真正的差異是「索引結構 + 排程策略」，不是「要不要分頁」。兩邊功能持續趨同——別把某一次 benchmark 當永久結論。", GOOD);
  footer(s, P2);
})();

// ============================================================ 16 怎麼選
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "15", "所以問題②該選哪一家？看你的流量長相", WARN);
  card(s, MX, 1.4, 5.8, 2.5, BG2, SG);
  s.addText("前綴共用 > 60% → SGLang", { x: MX + 0.28, y: 1.52, w: 5.2, h: 0.38, fontFace: HEAD, fontSize: 16, bold: true, color: SG, margin: 0 });
  ["RAG（同幾份文件反覆進 prompt）", "多輪對話（每輪帶整段歷史）", "Few-shot（同一批範例）", "Agent（工具描述 + 前面所有步驟）", "sgl.fork() 的分支"]
    .forEach((t, i) => s.addText("· " + t, { x: MX + 0.28, y: 1.96 + i * 0.38, w: 5.2, h: 0.36, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 }));

  card(s, 7.0, 1.4, 5.6, 2.5, BG2, VL);
  s.addText("前綴各自獨立 → 看生態成熟度", { x: 7.28, y: 1.52, w: 5.0, h: 0.38, fontFace: HEAD, fontSize: 16, bold: true, color: VL, margin: 0 });
  ["客服：各種不相干的問題", "批次翻譯：每篇文章都不同", "一次性摘要任務", "→ 此時兩家吞吐差距在 5% 內", "→ 就選模型/硬體支援更廣的那家"]
    .forEach((t, i) => s.addText((i >= 3 ? "" : "· ") + t, { x: 7.28, y: 1.96 + i * 0.38, w: 5.0, h: 0.36, valign: "middle", fontFace: BODY, fontSize: 11.5, bold: i >= 3, color: i >= 3 ? VL : MUTE, margin: 0 }));

  card(s, MX, 4.1, 11.9, 1.65, BG2, WARN);
  s.addText("三個容易被忽略的選型因素", { x: MX + 0.28, y: 4.2, w: 6, h: 0.32, fontFace: HEAD, fontSize: 14.5, bold: true, color: WARN, margin: 0 });
  [["硬體", "非 NVIDIA（AMD / TPU / Gaudi / CPU）→ vLLM 覆蓋明顯更廣"],
  ["模型", "剛出的新架構誰先支援？通常 vLLM 廣、SGLang 對 DeepSeek 系特別快"],
  ["多租戶安全", "前綴快取跨使用者共享會讓 TTFT 洩漏「這段 prompt 是否被用過」→ 要分租戶命名空間"]]
    .forEach(([k, v], i) => {
      const y = 4.6 + i * 0.38;
      s.addText(k, { x: MX + 0.3, y, w: 1.5, h: 0.36, valign: "middle", fontFace: HEAD, fontSize: 11.5, bold: true, color: WARN, margin: 0 });
      s.addText(v, { x: MX + 1.9, y, w: 9.7, h: 0.36, valign: "middle", fontFace: BODY, fontSize: 11.2, color: MUTE, margin: 0 });
    });
  footer(s, P2);
})();

// ============================================================ 17 互動環節
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "16", "互動環節：把「有效 batch」變成看得到的東西", GOOD);
  s.addText("開啟 interactive/serving_map.html —— 本堂只用模式 1–4（模式 5 PD 分離留給第四堂）。",
    { x: MX, y: 1.4, w: 11.9, h: 0.35, fontFace: BODY, fontSize: 13.5, color: MUTE, margin: 0 });
  [["1 Naive", "靜態批次 + 連續預留 KV：時間軸有多空、HBM 有多浪費", WARN, "有效 batch ≈ 20 → 6.8%"],
  ["2 Continuous batching", "做完就換人上 → 時間軸被填滿，但 KV 還是預留制", COMP, "≈ 60 → 20%"],
  ["3 Paged KV", "block 化 + 用完才要 → 同 HBM 放得下 ~4 倍的請求", MEM, "≈ 240 → 81%"],
  ["4 前綴共用", "共用 prompt 的請求合併 → prefill 直接省掉", PURP, "同上，TTFT ↓20–40%"]]
    .forEach(([t, d, c, n], i) => {
      const y = 1.95 + i * 0.82;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 11.9, h: 0.7, rectRadius: 0.06, fill: { color: BG2 }, line: { color: c, width: 1 } });
      s.addText(t, { x: MX + 0.22, y, w: 2.9, h: 0.7, valign: "middle", fontFace: HEAD, fontSize: 13, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 3.2, y, w: 5.6, h: 0.7, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
      s.addText(n, { x: MX + 8.9, y, w: 2.8, h: 0.7, align: "right", valign: "middle", fontFace: MONO, fontSize: 10.5, bold: true, color: c, margin: 0 });
    });
  s.addText("操作：數字鍵 1–4 切換 · 右側面板同步顯示有效 batch、KV 浪費比例、算術強度與 roofline 上的位置",
    { x: MX, y: 5.35, w: 11.9, h: 0.35, align: "center", fontFace: MONO, fontSize: 11, color: FOOTC, margin: 0 });
  takeaway(s, "從 6.8% 走到 81%——這三步是兩家共有的地基，不是誰的專利。", GOOD);
  footer(s, P2);
})();

// ============================================================ 18 問題③
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "17", "問題③：輸出格式不可控（而且檢查很慢）", PURP);
  probStepper(s, 2);
  problemBanner(s, "③", "只靠 prompt 說「請輸出 JSON」，模型不保證遵守",
    "常見翻車：前後多加解釋文字、欄位型別錯、語法小錯（多逗號／單引號）、幻覺出 schema 沒有的欄位。而 agent／function calling 一旦解析失敗，整條鏈就斷了。天真的補救是每步生成後用正則檢查再重試——CPU 成本高得離譜，還是不保證對。", PURP);
  [["多餘文字", '"當然！以下是您要的 JSON："'], ["型別錯誤", '{"age": "twenty"}'], ["語法錯誤", "多逗號、單引號、缺括號"], ["幻覺欄位", "schema 裡根本沒有的 key"]]
    .forEach(([t, d], i) => {
      const x = MX + i * 3.0;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 4.1, w: 2.85, h: 1.0, rectRadius: 0.06, fill: { color: BG2 }, line: { color: WARN, width: 1 } });
      s.addText(t, { x: x + 0.15, y: 4.2, w: 2.55, h: 0.32, fontFace: HEAD, fontSize: 12.5, bold: true, color: WARN, margin: 0 });
      s.addText(d, { x: x + 0.15, y: 4.53, w: 2.55, h: 0.5, valign: "top", fontFace: MONO, fontSize: 10, color: MUTE, lineSpacingMultiple: 1.2, margin: 0 });
    });
  takeaway(s, "澄清：這不需要另一個小模型來審核——它純粹是符號計算／規則引擎的問題。", PURP);
  footer(s, P3);
})();

// ============================================================ 19 共同地基：FSM
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "18", "共同地基：FSM + 位元遮罩（兩家預設都用 XGrammar）", GOOD);
  s.addText("生成每個 token 前，先算出「此刻合法的 token 集合」，把不合法的 logit 壓成 −∞ → 物理上不可能生成非法內容。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  [["① 預編譯成 FSM", "把 JSON schema / 正則 / EBNF 編成有限狀態機。XGrammar 是 SGLang、vLLM、TensorRT-LLM 三家的預設後端（另可換 Outlines、llguidance）", COMP],
  ["② 非同步編譯", "編譯要時間 → vLLM 讓請求先進 WAITING_FOR_FSM，編好才轉 WAITING，不阻塞其他請求", MEM],
  ["③ 位元遮罩", "每步查表得出合法集合，其餘 token 的 logit 設成 −∞，再於合法子集重新歸一化採樣", PURP]]
    .forEach(([t, d, c], i) => {
      const y = 1.85 + i * 1.15;
      card(s, MX, y, 11.9, 1.02, BG2, c);
      s.addText(t, { x: MX + 0.25, y: y + 0.06, w: 3.0, h: 0.42, valign: "middle", fontFace: HEAD, fontSize: 14.5, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 0.25, y: y + 0.48, w: 11.4, h: 0.46, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
    });
  verdict(s, 5.4, "⚠️ 常見誤解：以為 SGLang 是自研語法引擎、vLLM 是外掛。實際上兩家的預設後端都是 XGrammar——差異在下一頁。", GOOD);
  footer(s, P3);
})();

// ============================================================ 20 對比③ jump-forward
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "19", "對比③：語法允許時，可以一次吐好幾個 token", SG);
  s.addText("這是 SGLang「compressed FSM」真正的價值：不是換一個語法引擎，是在同一個 FSM 上多走幾步。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  card(s, MX, 1.8, 11.9, 1.75, BG2, GOOD);
  s.addText("觀察：JSON 裡有一大段路徑是「唯一確定」的", { x: MX + 0.28, y: 1.9, w: 7, h: 0.35, fontFace: HEAD, fontSize: 14.5, bold: true, color: GOOD, margin: 0 });
  s.addText('{"name": "', { x: MX + 0.3, y: 2.32, w: 2.5, h: 0.5, valign: "middle", fontFace: MONO, fontSize: 16, bold: true, color: COMP, margin: 0 });
  s.addText("← schema 一確定，這 8 個 token 就沒有第二種可能", { x: MX + 2.9, y: 2.32, w: 5.5, h: 0.5, valign: "middle", fontFace: BODY, fontSize: 12, color: MUTE, margin: 0 });
  s.addText("逐 token 推理 8 次　→　直接吐出，0 次推理", { x: MX + 8.3, y: 2.32, w: 3.3, h: 0.5, align: "right", valign: "middle", fontFace: HEAD, fontSize: 12, bold: true, color: GOOD, margin: 0 });
  timeline(s, MX + 1.9, 2.95, 9.4, [[0.5, COMP, "逐 token：8 步 forward"], [0.5, BG3, ""]], "沒有跳躍", MUTE);
  card(s, MX, 3.7, 11.9, 1.55, BG2, SG);
  s.addText("jump-forward decoding（SGLang）", { x: MX + 0.28, y: 3.8, w: 6, h: 0.35, fontFace: HEAD, fontSize: 14.5, bold: true, color: SG, margin: 0 });
  timeline(s, MX + 1.9, 4.22, 9.4, [[0.1, SG, "跳"], [0.4, COMP, "只推理真正有選擇的部分"], [0.5, GOOD, "省下來的時間拿去服務其他請求"]], "有跳躍", MUTE);
  s.addText("結果：SGLang 的結構化輸出吞吐約為 vLLM 的 3× —— 加了語法約束之後，生成反而比自由生成更快。",
    { x: MX + 0.28, y: 4.7, w: 11.3, h: 0.45, valign: "middle", fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  takeaway(s, "反直覺但重要：約束不是成本，是資訊——它告訴引擎「這幾個 token 不用問模型」。", SG);
  footer(s, P3);
})();

// ============================================================ 21 正交性
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "20", "問題② 與 ③ 的關係：正交，但有一條紅線", GOOD);
  card(s, MX, 1.5, 5.8, 2.2, BG2, MEM);
  s.addText("前綴複用複用的是「算力」", { x: MX + 0.28, y: 1.62, w: 5.2, h: 0.35, fontFace: HEAD, fontSize: 14.5, bold: true, color: MEM, margin: 0 });
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
  s.addText("這條紅線給了一個通用判準：可以快取的是「計算結果」，不可以快取的是「請求狀態」。第四堂的跨機 KV 複製也要問同一個問題。",
    { x: MX + 0.28, y: 5.15, w: 11.3, h: 0.8, valign: "middle", fontFace: BODY, fontSize: 13, color: GOOD, margin: 0 });
  footer(s, P3);
})();

// ============================================================ 22 問題④
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "21", "問題④：GPU 快到讓 CPU 變成瓶頸", GOOD);
  probStepper(s, 3);
  problemBanner(s, "④", "一步只花 5–10 ms，Python 那邊來得及嗎？",
    "前面三個問題解完，batch 撐起來了、重複計算省掉了。這時每個 forward step 只剩幾毫秒，而 tokenize、排程決策、取樣、序列化、數百次 kernel launch 全在 CPU 上——GPU 開始出現「算完在等 CPU 決定下一步」的空窗。", GOOD);
  card(s, MX, 4.05, 11.9, 1.35, BG2, MEM);
  s.addText("共同地基：CUDA Graph", { x: MX + 0.28, y: 4.15, w: 5, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: MEM, margin: 0 });
  s.addText("初始化時對各種 batch size 做 dummy forward，把整串 kernel launch 錄成 DAG，之後直接 replay → 省掉每步數百次 launch 開銷。兩家都有（vLLM V1 用 piecewise CUDA graph 兼顧動態形狀）。",
    { x: MX + 0.28, y: 4.52, w: 11.3, h: 0.8, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.35, margin: 0 });
  takeaway(s, "但「把 CPU 工作藏起來」這件事，兩家走了兩條不同的路——下一頁。", GOOD);
  footer(s, P4);
})();

// ============================================================ 23 對比④
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "22", "對比④：兩條不同的路，同一個目標", VL);
  s.addText("目標都是「GPU 不要等 CPU」。差別在把 CPU 工作搬到哪裡。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  versus(s, 1.8, 2.5,
    "zero-overhead scheduler", ["把 CPU 排程藏進「上一步的 GPU 執行時間」裡", "排程與 GPU kernel 在時間上完全重疊", "v0.4 起導入，是它能撐大規模部署的關鍵", "單一進程內把時序排好"],
    "多進程 + async 排程", ["EngineCore 獨立進程只跑排程 + 執行", "tokenize / 多模態前處理 / 串流輸出各自重疊", "async scheduling：下一步決策與本步執行重疊", "用進程邊界隔開，工程上較易維護"]);
  card(s, MX, 4.5, 11.9, 1.35, BG2, GOOD);
  s.addText("為什麼值得講這一頁", { x: MX + 0.28, y: 4.6, w: 5, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: GOOD, margin: 0 });
  s.addText("聽眾很容易以為「推論優化＝寫 CUDA」。實際上一大塊收益來自「別讓 GPU 等 CPU」——vLLM V1 相對 V0 吞吐提升 ~1.7×，完全來自 CPU 開銷削減，一個 GPU kernel 都沒改。這跟第一堂 prefetch/overlap 的 demo 是同一個道理。",
    { x: MX + 0.28, y: 4.95, w: 11.3, h: 0.85, valign: "top", fontFace: BODY, fontSize: 12, color: MUTE, lineSpacingMultiple: 1.35, margin: 0 });
  footer(s, P4);
})();

// ============================================================ 24 TTFT vs ITL
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "23", "問題④的雙胞胎：TTFT 與 ITL 互相打架", WARN);
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
  s.addText("共同解法：chunked prefill —— 每步固定 token 預算，剩下的預算拿去混 decode（兩家都有，vLLM V1 預設開啟）", { x: MX + 0.28, y: 4.65, w: 11.3, h: 0.32, fontFace: HEAD, fontSize: 13.5, bold: true, color: GOOD, margin: 0 });
  [1, 2, 3].forEach((n, i) => timeline(s, MX + 2.0, 5.05 + i * 0.27, 9.3, [[0.45, COMP, `prefill chunk ${n}/8`], [0.55, MEM, "同批 decode（12 條在跑）"]], `step ${n}`, MUTE));
  takeaway(s, "附帶好處：compute-bound 的 prefill 與 memory-bound 的 decode 混同一批，兩種瓶頸互補。⚠️ 但這只是緩解——根治要靠第四堂的 PD 分離。", COMP);
  footer(s, P4);
})();

// ============================================================ 25 天花板① 投機解碼
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "24", "單機的天花板①：投機解碼 / MTP（兩家都有）", COMP);
  s.addText("前面四個問題解完，B 撐大了。但「單一請求的延遲」仍被『讀一遍權重』鎖死——要打破它只有這條路。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  [["Draft", "便宜地提出 k 個候選：n-gram / 小模型 / Medusa / EAGLE / 模型自帶的 MTP head", MEM],
  ["Verify", "大模型把「context + k 個草稿 token」在一次 forward 裡跑完", COMP],
  ["Accept", "由左而右比對機率：large ≥ draft 就收；否則按 large/draft 機率收。第一個被拒就停，並免費得到第 k+1 個", GOOD]]
    .forEach(([t, d, c], i) => {
      const x = MX + i * 4.03;
      card(s, x, 1.8, 3.83, 1.6, BG2, c);
      s.addText(t, { x: x + 0.2, y: 1.9, w: 3.4, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: c, margin: 0 });
      s.addText(d, { x: x + 0.2, y: 2.28, w: 3.45, h: 1.05, valign: "top", fontFace: BODY, fontSize: 10.8, color: MUTE, lineSpacingMultiple: 1.25, margin: 0 });
      if (i < 2) s.addText("▶", { x: x + 3.85, y: 2.4, w: 0.2, h: 0.4, fontFace: BODY, fontSize: 14, color: FOOTC, margin: 0 });
    });
  card(s, MX, 3.6, 11.9, 1.15, BG2, COMP);
  s.addText("為什麼會賺（用第 4 頁的同一把尺）", { x: MX + 0.28, y: 3.68, w: 6, h: 0.3, fontFace: HEAD, fontSize: 14, bold: true, color: COMP, margin: 0 });
  s.addText("驗證 k 個草稿 token 的 forward：權重讀取 = 1 次（不變），FLOPs = k 倍　→　AI 從 1 變成 k　＝用閒置算力換頻寬",
    { x: MX + 0.28, y: 4.02, w: 11.3, h: 0.65, valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: INK, margin: 0 });
  versus(s, 4.9, 1.05,
    "EAGLE / EAGLE3 / MTP", ["對 DeepSeek 系的 MTP head 支援特別完整"],
    "n-gram / EAGLE / Medusa", ["內建多種 draft 方式，可依 workload 選"]);
  takeaway(s, "輸出分佈嚴格不變（rejection sampling 保證）。DeepSeek-V3 的 MTP head 報告第二 token 接受率 ~85–90%。", COMP);
  footer(s, P5);
})();

// ============================================================ 26 天花板② 量化
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "25", "單機的天花板②：量化——直接砍分母", MEM);
  s.addText("AI = FLOPs ÷ Bytes。前面全在動分子，量化直接動分母——decode 被權重讀取綁死，權重減半 ≈ 延遲減半。兩家都支援。",
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

// ============================================================ 27 彙整總表
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "26", "彙整：四個問題 × 兩種寫法", GOOD);
  tableGrid(s, MX, 1.4, 11.9, [
    { t: "問題", w: 2.0 }, { t: "共同地基", w: 3.0 }, { t: "SGLang 的寫法", w: 3.5, c: SG }, { t: "vLLM 的寫法", w: 3.4, c: VL },
  ], [
    ["① 程式難平行", "—", "前端 DSL：程式＝執行圖", "沒有對應物（交給應用層）"],
    ["② KV 怎麼放", "分頁式 KV、block table", "同（分頁記憶體池）", "同（PagedAttention 命名）"],
    ["② 怎麼被找到", "前綴複用", "radix tree，O(D) 最長前綴", "鏈式雜湊表，O(1) 對齊 block"],
    ["② 什麼順序進來", "continuous batching", "cache-aware 排程（主動）", "FCFS / priority（被動）"],
    ["③ 格式保證", "XGrammar FSM + 位元遮罩", "＋ jump-forward（約 3× 吞吐）", "＋ 非同步 FSM 編譯"],
    ["④ CPU 開銷", "CUDA Graph", "zero-overhead scheduler", "多進程 + async 排程（V1 ~1.7×）"],
    ["④' TTFT vs ITL", "chunked prefill", "有", "V1 預設開啟"],
    ["天花板", "投機解碼、量化", "EAGLE3 / MTP 支援強", "n-gram / EAGLE / Medusa 多選"],
  ], MEM, 10.5);
  verdict(s, 5.15, "看這張表的方法：共同地基那一欄是「這個領域已經收斂的共識」，右邊兩欄才是差異——而差異幾乎都落在「要不要替使用者猜工作負載」。", GOOD);
  footer(s, P5);
})();

// ============================================================ 28 帶走三句話
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "27", "帶走三句話", COMP);
  [["1", "每個機制都是被一個具體痛點逼出來的", "DSL 因為程式難平行、前綴複用因為程式製造了大量共享前綴、FSM 因為 agent 需要能解析的輸出、CPU 優化因為前三個解完 GPU 快到 CPU 跟不上。順著問題走，就不用背功能表。", COMP],
  ["2", "兩家的差異幾乎都在「要不要替你猜工作負載」", "SGLang 猜你會重複用前綴與語法（radix tree、cache-aware 排程、jump-forward），猜對了就贏很多；vLLM 不猜，把廣度與抽象做好（最多模型/硬體、可插拔後端）。共同地基（分頁 KV、continuous batching、XGrammar、CUDA Graph）兩家一致。", GOOD],
  ["3", "所以選型看流量長相，不看 benchmark 排名", "前綴共用 >60%（agent／多輪／RAG）→ SGLang 的 TTFT 低 20–40%、結構化輸出快約 3×；請求彼此獨立 → 兩者差 <5%，就選硬體與模型支援更廣的那家。", MEM]]
    .forEach(([n, t, d, c], i) => {
      const y = 1.8 + i * 1.5;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 0.85, h: 0.85, rectRadius: 0.12, fill: { color: c }, line: { type: "none" } });
      s.addText(n, { x: MX, y, w: 0.85, h: 0.85, align: "center", valign: "middle", fontFace: MONO, fontSize: 30, bold: true, color: BG, margin: 0 });
      s.addText(t, { x: MX + 1.1, y, w: 11.0, h: 0.5, fontFace: HEAD, fontSize: 17.5, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 1.1, y: y + 0.52, w: 11.2, h: 0.95, valign: "top", fontFace: BODY, fontSize: 12.2, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
    });
  s.addText("下一堂：同一批問題到了「一群機器」之上（⑤–⑧）——而那裡兩家的分工方式差得更明顯。",
    { x: MX, y: 6.3, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 12.5, color: FOOTC, margin: 0 });
  footer(s, P5);
})();

pres.writeFile({ fileName: "../class3_engine_single_node.pptx" }).then((f) => console.log("✅ 產生：" + f + "（" + PAGE + " 頁）")).catch((e) => console.error(e));
