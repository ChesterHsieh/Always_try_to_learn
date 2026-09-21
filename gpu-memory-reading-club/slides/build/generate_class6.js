// 第六堂課（最終章）— 一個字，穿過一整排機櫃
// 產生 ../class6_multi_rack_inference.html。沿用系列的深色「矽晶」主題。
//
// 主軸 = 拆散變便宜的前提：最常搬的資料，走最快的路。
// 拆法 = 按「多常搬 × 一次多大」把資料分四種（請求 / KV / MoE 交換 / 權重），
//        中段跟著一個請求依時序走，合回來指出真瓶頸＝一個 scale-up 域裝得下幾張卡。
// 主角 = SGLang 96×H100（prefill 4 台 EP32、decode 9 台 EP72）；DeepSeek 官方只當開場鉤子。
// 逐頁事實原子與數字來源見 ../../notes/class6_multi_rack_inference.md
const pptxgen = require("./pptx-html");

const BG = "0E1726", BG2 = "16233A", BG3 = "1C2E4A";
const INK = "EAF1FB", MUTE = "8FA6C4", LINE = "2A3D5C", FOOTC = "5C7299";
const MEM = "38BDF8", COMP = "F59E0B", WARN = "FB7185", GOOD = "34D399", PURP = "A78BFA";
const MEMTINT = "10455F", COMPTINT = "4A3410", WARNTINT = "4A2433", GOODTINT = "123D31", PURPTINT = "2A2150";
const HEAD = "PingFang TC", BODY = "PingFang TC", MONO = "Menlo";

const W = 13.33, H = 7.5, MX = 0.7, TITLE_Y = 0.62, FOOT_Y = 7.05, TOTAL = 20;
const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.3 });

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "GPU 記憶體與資料搬遷讀書會";
pres.title = "第六堂課 · 一個字，穿過一整排機櫃";

let PAGE = 0;
const base = (s) => { s.background = { color: BG }; PAGE += 1; };
function runningHeader(s) {
  s.addText("讀書會 · 第六堂課 · 機櫃群的一輪 inference", { x: W - 5.9, y: 0.3, w: 5.2, h: 0.3, align: "right", fontFace: BODY, fontSize: 10, color: MUTE, margin: 0 });
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
// 旅程進度條（第 5–12 頁）
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

const P0 = "讀書會 · 第六堂課（最終章）";
const PA = "鉤子與拆法";
const PB = "跟著一個請求走";
const PC = "合回來：真瓶頸";
const PD = "推論：NVIDIA vs AMD";
const PE = "收尾";

// ============================================================ 1 鉤子（封面）
(() => {
  const s = pres.addSlide(); base(s);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.16, h: H, fill: { color: MEM }, line: { type: "none" } });
  s.addText("第六堂課 · 最終章", { x: MX + 0.3, y: 0.95, w: 8, h: 0.42, fontFace: MONO, fontSize: 15, color: MEM, margin: 0 });
  s.addText("你看到的每一個字，", { x: MX + 0.3, y: 1.45, w: 11.5, h: 0.8, fontFace: HEAD, fontSize: 38, bold: true, color: INK, margin: 0 });
  s.addText("背後有 72 張卡同時踏了一步", { x: MX + 0.3, y: 2.2, w: 11.5, h: 0.8, fontFace: HEAD, fontSize: 38, bold: true, color: INK, margin: 0 });
  s.addText("一個字，穿過一整排機櫃 —— SGLang / vLLM × DeepSeek × NVIDIA / AMD", { x: MX + 0.3, y: 3.1, w: 11.5, h: 0.45, fontFace: HEAD, fontSize: 18, color: MUTE, margin: 0 });
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
    { text: "　—— 把前五堂的零件裝回機櫃，跟著一個請求算帳。", options: { color: INK } },
  ], { x: MX + 0.55, y: 5.55, w: 10.8, h: 0.85, valign: "middle", fontFace: HEAD, fontSize: 16, margin: 0 });
  footer(s, P0);
})();

// ============================================================ 2 拆散反而更便宜
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "01", "把 DeepSeek 拆到越多卡上，每張卡反而產出越多 token", COMP);
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

// ============================================================ 3 服務長什麼樣
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "02", "一個「模型服務」不是一台機器，是一個 router、兩個池子、三張網", MEM);
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

// ============================================================ 4 四種資料 × 四條路
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "03", "機櫃群＝多了三層的記憶體階層：每種資料走它付得起的路", PURP);
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

// ============================================================ 5 第一站 router
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "04", "進門那一跳只搬幾 KB，卻決定了一半的 prefill 要不要算", MEM);
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
  s.addText("接第四堂⑦：cache-aware router 的局部性 vs 均衡", { x: MX, y: 6.6, w: 11.9, h: 0.3, fontFace: BODY, fontSize: 10.5, color: FOOTC, margin: 0 });
  footer(s, PB);
})();

// ============================================================ 6 第二站 prefill
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "05", "Prefill 是整批的大塊通訊，跨 InfiniBand 也吃得消", COMP);
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

// ============================================================ 7 第三站 KV 交接流程
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "06", "KV 交接不是「傳過去」而已：SGLang 先預留再推，vLLM 先算完再拉", MEM);
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
  takeaway(s, "兩家做的都是第四堂的「KV 所有權轉移」，差別只在誰先動——推的那一方先算，拉的那一方先等。", MEM);
  footer(s, PB);
})();

// ============================================================ 8 KV 要搬多久
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "07", "一份 5K token 的 KV 跨機只要 7 ms——MLA 在第五堂就替這一跳付了帳", MEM);
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

// ============================================================ 9 decode 一步
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "08", "Decode 一步：attention 各算各的，MoE 大家一起交換", PURP);
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

// ============================================================ 10 通訊的時間帳
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "09", "不做重疊時，通訊佔掉 decode 一步約四成", MEM);
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

// ============================================================ 11 為什麼拆散便宜
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "10", "拆散之所以便宜：每卡只放 4 個專家，省下的 HBM 拿去開大 batch", GOOD);
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
  ["結果：decode vs TP16 為 5.2×", "這就是第 1 頁那條長條的來源", GOOD]]
    .forEach(([t, d, c], i) => {
      const y = 4.55 + i * 0.44;
      s.addText("· " + t, { x: 6.8, y, w: 3.3, h: 0.4, valign: "middle", fontFace: HEAD, fontSize: 11.5, bold: true, color: c, margin: 0 });
      s.addText(d, { x: 10.0, y, w: 2.6, h: 0.4, valign: "middle", fontFace: BODY, fontSize: 9.8, color: MUTE, margin: 0 });
    });
  takeaway(s, "拆散買到的不是算力，是 HBM：權重佔得少 → KV 放得多 → batch 開得大 → memory-bound 的 decode 才划算。", GOOD);
  footer(s, PB);
})();

// ============================================================ 12 代價：節拍器
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "11", "代價是 72 張卡綁成一個節拍器：最慢那張決定速度", WARN);
  lede(s, "DP attention + EP 下，同一個 decode 單元的所有 rank 必須同步進入每一層的 all-to-all。");
  for (let i = 0; i < 72; i++) {
    const x = MX + (i % 18) * 0.49, y = 1.85 + Math.floor(i / 18) * 0.42;
    const slow = i === 29;
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.4, h: 0.33, fill: { color: slow ? WARNTINT : MEMTINT }, line: { color: slow ? WARN : MEM, width: slow ? 1.5 : 0.6 } });
  }
  s.addText("← 這一張卡慢\n（專家熱點／掉卡）\n另外 71 張全部在等", { x: MX + 9.0, y: 2.0, w: 3.2, h: 1.0, valign: "middle", fontFace: BODY, fontSize: 11, bold: true, color: WARN, margin: 0 });
  [["最慢那張決定整體 ITL", "沒請求的卡也得跑空批次陪跑——集合通訊的本質（第四堂 Part A 的 straggler）", COMP],
  ["掉一張卡，整個 72 卡單元停擺", "爆炸半徑隨 EP 變大（第四堂⑧）", WARN],
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

// ============================================================ 13 互動環節
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "12", "互動環節：同一個請求放到四種硬體上各走一次", PURP);
  card(s, MX, 1.4, 11.9, 0.95, BG2, PURP);
  s.addText("interactive/rack_journey_map.html", { x: MX + 0.3, y: 1.4, w: 6.5, h: 0.95, valign: "middle", fontFace: MONO, fontSize: 17, bold: true, color: PURP, margin: 0 });
  [["1–4", "切層"], ["H B G A", "切硬體"], ["空白鍵", "播放／暫停"]].forEach(([k, d], i) => {
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
  takeaway(s, "四種硬體跑同一個請求，router、prefill、KV 交接幾乎不動——只有 decode 的交換那一站變了。", PURP);
  footer(s, PC);
})();

// ============================================================ 14 合回來：真瓶頸
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "13", "四種資料只有一種付不起跨機的路", COMP);
  lede(s, "回到第 3 頁的地圖，逐格結帳：");
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

// ============================================================ 15 證據
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "14", "同一顆 GPU、同樣的 NVLink 頻寬，只把域從 8 擴到 72", GOOD);
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

// ============================================================ 16 AMD 軟體
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "15", "軟體圖搬得過去，零件得重寫一遍", MEM);
  lede(s, "第 4–10 頁那張圖在 AMD 上長一模一樣，只是每個零件換成 ROCm 的對應物。");
  tableGrid(s, MX, 1.8, 11.9, [
    { t: "旅程的一站", w: 3.1 }, { t: "NVIDIA 生態", w: 3.6 }, { t: "AMD / ROCm 對應物", w: 5.2 },
  ], [
    ["集合通訊", "NCCL", "RCCL（ROCm 10 併入上游 NCCL 2.30.4）"],
    ["MoE dispatch / combine", "DeepEP（NVSHMEM + IBGDA）", "MoRI-EP；另有 ROCm 版 DeepEP（只列 MI300X/MI308X）"],
    ["KV 交接", "NIXL、Mooncake", "MoRI-IO（實測比 Mooncake 快約 10%）"],
    ["MLA / MoE kernel", "FlashMLA、DeepGEMM、FlashInfer", "AITER（vLLM ROCM_AITER_MLA 比 Triton MLA 快 1.2–1.6×）"],
    ["服務層編排", "Dynamo", "沒有完整上游對應物（靠 SGLang router / llm-d）"],
    ["算力網卡", "ConnectX-7 / 8", "Pensando Pollara 400（UEC）、Vulcano 800"],
  ], MEM, 11, 0.4);
  card(s, MX, 4.65, 5.8, 1.25, BG2, GOOD);
  s.addText("搬得過去（最佳情境）", { x: MX + 0.22, y: 4.72, w: 5.4, h: 0.32, fontFace: HEAD, fontSize: 13, bold: true, color: GOOD, margin: 0 });
  s.addText("R1-0528 MXFP4、24 張 MI355X、SGLang + MoRI：$0.169／1M token，對照 B200 + TRT-LLM 的 $0.178（廠商合寫，只比 B200）",
    { x: MX + 0.22, y: 5.05, w: 5.4, h: 0.8, valign: "top", fontFace: BODY, fontSize: 10.8, color: MUTE, lineSpacingMultiple: 1.25, margin: 0 });
  card(s, 6.8, 4.65, 5.8, 1.25, BG2, WARN);
  s.addText("但不等於一樣成熟（第三方）", { x: 7.02, y: 4.72, w: 5.4, h: 0.32, fontFace: HEAD, fontSize: 13, bold: true, color: WARN, margin: 0 });
  s.addText("InferenceX v2：FP4 被 B200 大幅甩開、高互動性下 PD 分離反而比單機慢、開源分散式推論「落後超過六個月」；DeepSeek-V4 上線兩個月，多機在 ROCm 上仍跑不起來",
    { x: 7.02, y: 5.05, w: 5.4, h: 0.8, valign: "top", fontFace: BODY, fontSize: 10.8, color: MUTE, lineSpacingMultiple: 1.25, margin: 0 });
  takeaway(s, "落後的不是架構設計，是「新模型上線那天，零件有沒有人寫好」——老模型追得上，新模型差好幾倍。", MEM);
  footer(s, PD);
})();

// ============================================================ 17 AMD 硬體
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "16", "AMD 的強項是「一台裝得下」，弱項是「一個域只有 8 張」", COMP);
  tableGrid(s, MX, 1.4, 11.9, [
    { t: "", w: 3.2 }, { t: "AMD MI355X 8 卡機", w: 3.2 }, { t: "NVIDIA HGX B200 8 卡機", w: 2.9 }, { t: "NVIDIA GB200 NVL72", w: 2.6 },
  ], [
    ["單卡 HBM", "288 GB HBM3E", "180 GB", "~186 GB"],
    ["一台／一櫃總 HBM", "~2.3 TB", "1.44 TB", "13.4 TB"],
    ["scale-up 拓樸", "8 卡全網狀（xGMI，無交換器）", "8 卡、NVSwitch 交換", "72 卡、NVSwitch 交換"],
    ["對任一張卡的頻寬（每方向）", "~77 GB/s（7 條合計 ≈ 538）", "900 GB/s", "900 GB/s"],
  ], COMP, 11, 0.46);
  card(s, MX, 3.75, 5.8, 1.95, BG2, GOOD);
  s.addText("強項：一台就裝得下", { x: MX + 0.25, y: 3.85, w: 5.3, h: 0.35, fontFace: HEAD, fontSize: 14.5, bold: true, color: GOOD, margin: 0 });
  s.addText("DeepSeek-V3 權重 688.6 GB → 一台 MI355X（2.3 TB）裝得下還剩 ~1.6 TB 給 KV；一台 H100（640 GB）連權重都放不下。中低 QPS、想避開跨機複雜度時，這是真優勢。",
    { x: MX + 0.25, y: 4.25, w: 5.35, h: 1.35, valign: "top", fontFace: BODY, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  card(s, 6.8, 3.75, 5.8, 1.95, BG2, WARN);
  s.addText("弱項：EP 一超過 8 就下慢車道", { x: 7.05, y: 3.85, w: 5.3, h: 0.35, fontFace: HEAD, fontSize: 14.5, bold: true, color: WARN, margin: 0 });
  s.addText("MoRI-EP 在 MI355X 上實測：dispatch 走 xGMI 345 GB/s、走 RDMA 只剩 54 GB/s（6.4×）；combine 420 → 71 GB/s。教具第 4 層裡，MI355X 在 EP72 的通訊和 H100 一樣是 ≈ 50 ms。",
    { x: 7.05, y: 4.25, w: 5.35, h: 1.35, valign: "top", fontFace: BODY, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  takeaway(s, "付不起慢車道的那種資料，在 AMD 現役機器上只有 8 張卡的快車道可走。", COMP);
  footer(s, PD);
})();

// ============================================================ 18 選型
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "17", "選型只問一題：你的最小單位是一台，還是一個機櫃？", PURP);
  tableGrid(s, MX, 1.4, 11.9, [
    { t: "如果你的單位是…", w: 3.4 }, { t: "決定性的數字", w: 3.0 }, { t: "2026 年 9 月的答案", w: 5.5 },
  ], [
    ["一台 node", "單台總 HBM", "MI355X（2.3 TB）> B200（1.44 TB）> H200（1.13 TB）→ AMD 在這一格有結構優勢"],
    ["數個機櫃", "scale-up 域大小 × 軟體成熟度", "現役 GB200 / GB300 NVL72 領先；AMD 8 卡網狀在這一格吃虧"],
  ], PURP, 11.5, 0.56);
  s.addText("機櫃級 scale-up 域的時間線", { x: MX, y: 3.0, w: 6, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0 });
  tableGrid(s, MX, 3.4, 11.9, [
    { t: "系統", w: 2.9 }, { t: "域", w: 0.8 }, { t: "每卡 scale-up（雙向）", w: 3.0 }, { t: "每卡 HBM", w: 1.8 }, { t: "狀態（2026-09）", w: 3.4 },
  ], [
    ["GB200 NVL72", "72", "1.8 TB/s（NVLink 5）", "~186 GB", "主流部署，~120 kW／櫃"],
    ["GB300 NVL72", "72", "1.8 TB/s", "~288 GB", "部署中，每卡 800G 網卡"],
    ["Vera Rubin NVL72", "72", "3.6 TB/s（NVLink 6）", "288 GB HBM4", "2026-05 宣布量產，秋季開始出貨"],
    ["AMD Helios", "72", "3.6 TB/s（UALink over Ethernet）", "432 GB HBM4", "AMD：Q3 末出貨；SemiAnalysis：2027 Q2 量產"],
  ], PURP, 11, 0.4);
  s.addText("什麼會推翻這個判斷：① Helios 出貨時，MoRI／RCCL 在 72 卡域上還沒有公開的大規模 EP 實測——硬體對等 ≠ 軟體對等。② 工作負載轉向超長上下文、KV 容量變主角時，單卡 HBM 大的一方優勢放大。",
    { x: MX, y: 5.5, w: 11.9, h: 0.5, valign: "top", fontFace: BODY, fontSize: 10.5, color: FOOTC, lineSpacingMultiple: 1.2, margin: 0 });
  takeaway(s, "不要問「AMD 還是 NVIDIA」，要問「我的最小單位是什麼」——2027 年 Helios 對 Rubin 的勝負會落在軟體。", PURP);
  footer(s, PD);
})();

// ============================================================ 19 旅程 × 全系列
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "18", "這趟旅程的每一站，都是前五堂的某一頁", MEM);
  tableGrid(s, MX, 1.4, 11.9, [
    { t: "旅程的一站", w: 3.6 }, { t: "決定它快慢的東西", w: 5.3 }, { t: "在哪一堂", w: 3.0 },
  ], [
    ["router 選誰", "cache-aware、局部性 vs 均衡", "第四堂 ⑦"],
    ["prefill 跨機", "compute-bound、大 GEMM", "第一堂 roofline"],
    ["KV 交接 7 ms", "MLA 把 KV 壓到 70 KB／token", "第五堂 旋鈕①"],
    ["KV 放得進 HBM", "分頁 KV、continuous batching", "第三堂 ②"],
    ["decode 為什麼要大 batch", "memory-bound、AI ≈ B", "第一堂、第三堂地基"],
    ["116 次 all-to-all 走哪條線", "scale-up 域 vs RDMA 的頻寬階梯", "第二堂 Part B"],
    ["每步不回 CPU", "CUDA Graph、固定 buffer", "第三堂 ④"],
    ["專家熱點、掉卡", "EPLB、爆炸半徑", "第四堂 ⑤⑧"],
    ["每字少搬幾個 byte", "MoE 稀疏、FP8 / NVFP4", "第五堂 旋鈕②⑤"],
  ], MEM, 11, 0.44);
  takeaway(s, "第一堂那張記憶體階層表，在第六堂長成了一座資料中心。", MEM);
  footer(s, PE);
})();

// ============================================================ 20 帶走三句話
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "19", "帶走三句話", MEM);
  [["1", "72 張卡同步踏一步划算，是因為最常搬的東西走了最快的路",
    "每字 116 次的 MoE 交換被 DeepEP 壓到每層不到 1 ms；一次性的 KV 交接被 MLA 壓到 7 ms，所以它可以跨機櫃。四種資料，四條路——這就是機櫃群版的記憶體階層。", MEM],
  ["2", "拆散的收益在 HBM，代價在節拍",
    "每卡只放 4 個專家、每卡跑 256 條序列，decode 因此比 TP16 快 5.2×；但整個單元被最慢的那張卡和最脆弱的那張卡綁在一起。", GOOD],
  ["3", "選型先問最小單位",
    "單位是一台，看總 HBM（MI355X 一台 2.3 TB 裝得下 DeepSeek）；單位是機櫃，看域的大小（同一顆 Blackwell，域從 8 到 72，每卡 4.4×）——而 2027 年那一仗會落在軟體。", PURP]]
    .forEach(([n, t, d, c], i) => {
      const y = 1.6 + i * 1.55;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 0.85, h: 0.85, rectRadius: 0.12, fill: { color: c }, line: { type: "none" } });
      s.addText(n, { x: MX, y, w: 0.85, h: 0.85, align: "center", valign: "middle", fontFace: MONO, fontSize: 30, bold: true, color: BG, margin: 0 });
      s.addText(t, { x: MX + 1.1, y, w: 11.0, h: 0.5, valign: "middle", fontFace: HEAD, fontSize: 17, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 1.1, y: y + 0.52, w: 11.2, h: 0.95, valign: "top", fontFace: BODY, fontSize: 12.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
    });
  s.addText("全系列收束：第一堂硬體 → 第二堂多卡 → 第三堂單機引擎 → 第四堂多機服務 → 第五堂模型架構 → 第六堂把它們裝回一整排機櫃，跟著一個字走完全程。",
    { x: MX, y: 6.35, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 12, color: FOOTC, margin: 0 });
  footer(s, PE);
})();

pres.writeFile({ fileName: "../class6_multi_rack_inference.html" }).then((f) => console.log("✅ 產生：" + f + "（" + PAGE + " 頁）")).catch((e) => console.error(e));
