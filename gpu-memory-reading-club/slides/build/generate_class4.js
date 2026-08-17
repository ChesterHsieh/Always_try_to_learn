// 第四堂課 — SGLang 多機篇：從一台到一群（問題 ⑤–⑧）
// 產生 ../class4_sglang_multi_node.pptx。沿用系列的深色「矽晶」主題。
//
// 主幹延續第三堂：沿著 SGLang 遇到的問題走。①–④ 是單機（第三堂），⑤–⑧ 是多機（本堂）。
// 開場框架：用「經典分散式系統的八類共同問題」當影子，逐格對照 GPU 推論叢集
//           ——哪些被規避（不是被解決）、哪些變形、哪些反而被放大成核心工程難題。
//   ⑤ 大 MoE 單機放不下、專家負載不均 → 大規模 EP + DeepEP / EPLB
//   ⑥ prefill 與 decode 互擾          → PD 分離
//   ⑦ 多副本：局部性 vs 負載均衡      → cache-aware router + KV 複製（壓軸推導：該搬還是該重算）
//   ⑧ 副本掛掉                        → 容錯（KV 是可重算的快取）
const pptxgen = require("pptxgenjs");

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
pres.title = "第四堂課 · SGLang 多機篇";

let PAGE = 0;
const base = (s) => { s.background = { color: BG }; PAGE += 1; };
function runningHeader(s) {
  s.addText("讀書會 · 第四堂課 · SGLang 多機篇", { x: W - 5.6, y: 0.3, w: 4.9, h: 0.3, align: "right", fontFace: BODY, fontSize: 10, color: MUTE, margin: 0 });
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
function takeaway(s, text, color) {
  s.addShape(pres.shapes.RECTANGLE, { x: MX, y: 6.02, w: 0.09, h: 0.62, fill: { color: color || MEM }, line: { type: "none" } });
  s.addText(text, { x: MX + 0.22, y: 6.0, w: 11.9, h: 0.66, fontFace: HEAD, fontSize: 14.5, bold: true, color: color || MEM, valign: "middle", margin: 0 });
}
function tableGrid(s, x, y, w, cols, rows, accent, fs, rowColors) {
  const rh = 0.42, hh = 0.42;
  let cx = x;
  cols.forEach((c) => {
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y, w: c.w, h: hh, fill: { color: BG3 }, line: { color: LINE, width: 0.8 } });
    s.addText(c.t, { x: cx + 0.08, y, w: c.w - 0.16, h: hh, valign: "middle", fontFace: HEAD, fontSize: (fs || 11) + 0.5, bold: true, color: INK, margin: 0 });
    cx += c.w;
  });
  rows.forEach((r, ri) => {
    cx = x;
    const ac = (rowColors && rowColors[ri]) || accent;
    r.forEach((cell, ci) => {
      s.addShape(pres.shapes.RECTANGLE, { x: cx, y: y + hh + ri * rh, w: cols[ci].w, h: rh, fill: { color: ri % 2 ? BG2 : BG }, line: { color: LINE, width: 0.8 } });
      const last = ci === r.length - 1 && rowColors;
      s.addText(cell, { x: cx + 0.08, y: y + hh + ri * rh, w: cols[ci].w - 0.16, h: rh, valign: "middle", fontFace: ci === 0 || last ? HEAD : BODY, fontSize: fs || 11, bold: ci === 0 || last, color: ci === 0 ? ac : last ? ac : MUTE, margin: 0 });
      cx += cols[ci].w;
    });
  });
}

// ── 問題導覽列：①–④ 第三堂（淡出），⑤–⑧ 本堂
const PROBS = ["① 程式難平行", "② 前綴重算", "③ 輸出不可控", "④ CPU 成瓶頸", "⑤ 大 MoE", "⑥ P/D 互擾", "⑦ 多機調度", "⑧ 容錯"];
function probStepper(s, active) {
  const y = 1.4, x0 = MX, w = 1.42, gap = 0.075, h = 0.44;
  PROBS.forEach((b, i) => {
    const mine = i >= 4, on = i === active;
    const x = x0 + i * (w + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.07, fill: { color: on ? PURP : BG2 }, line: { color: on ? PURP : mine ? LINE : BG3, width: 1 } });
    s.addText(b, { x, y, w, h, align: "center", valign: "middle", fontFace: BODY, fontSize: 9.5, bold: on, color: on ? BG : mine ? MUTE : FOOTC, margin: 0 });
    if (i === 3) s.addShape(pres.shapes.LINE, { x: x + w + gap / 2, y: y - 0.08, w: 0, h: h + 0.16, line: { color: FOOTC, width: 1, dashType: "dash" } });
  });
  s.addText("第三堂（單機）", { x: MX, y: 1.9, w: 6.0, h: 0.24, fontFace: BODY, fontSize: 9, color: FOOTC, margin: 0 });
  s.addText("本堂（多機）", { x: MX + 6.1, y: 1.9, w: 6.0, h: 0.24, fontFace: BODY, fontSize: 9, color: PURP, margin: 0 });
}
function problemBanner(s, num, problem, why, accent) {
  card(s, MX, 2.35, 11.9, 1.5, BG2, accent);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.25, y: 2.72, w: 0.75, h: 0.75, rectRadius: 0.12, fill: { color: accent }, line: { type: "none" } });
  s.addText(num, { x: MX + 0.25, y: 2.72, w: 0.75, h: 0.75, align: "center", valign: "middle", fontFace: MONO, fontSize: 26, bold: true, color: BG, margin: 0 });
  s.addText(problem, { x: MX + 1.25, y: 2.5, w: 10.3, h: 0.5, valign: "middle", fontFace: HEAD, fontSize: 20, bold: true, color: INK, margin: 0 });
  s.addText(why, { x: MX + 1.25, y: 3.02, w: 10.3, h: 0.7, valign: "top", fontFace: BODY, fontSize: 12.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
}

const P0 = "讀書會 · 第四堂課";
const PA = "Part A · 用分散式系統的問題清單當影子";
const P5 = "問題⑤ 大 MoE → 大規模專家平行";
const P6 = "問題⑥ P/D 互擾 → PD 分離";
const P7 = "問題⑦ 多機調度 → cache-aware router";
const P8 = "問題⑧ 容錯";
const PZ = "收尾";

// ============================================================ 1 標題
(() => {
  const s = pres.addSlide(); base(s);
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.16, h: H, fill: { color: PURP }, line: { type: "none" } });
  s.addText("第四堂課 · SGLang 多機篇", { x: MX + 0.3, y: 1.5, w: 8, h: 0.45, fontFace: MONO, fontSize: 15, color: PURP, margin: 0 });
  s.addText("從一台到一群", { x: MX + 0.3, y: 2.0, w: 10.5, h: 0.85, fontFace: HEAD, fontSize: 42, bold: true, color: INK, margin: 0 });
  s.addText("這一群卡，為什麼有的忙死有的閒死？", { x: MX + 0.3, y: 2.9, w: 10.5, h: 0.7, fontFace: HEAD, fontSize: 27, bold: true, color: MUTE, margin: 0 });
  s.addText("⑤ 大 MoE　⑥ P/D 互擾　⑦ 多機調度　⑧ 容錯　　（①–④ 單機問題見第三堂）",
    { x: MX + 0.3, y: 3.75, w: 11.3, h: 0.4, fontFace: BODY, fontSize: 14, color: PURP, margin: 0 });
  card(s, MX + 0.3, 4.45, 11.3, 1.25, BG2, MEM);
  s.addText([
    { text: "第三堂在回答「這張卡為什麼閒著」，第四堂在回答「這一群卡為什麼調度不好」。", options: { bold: true, color: MEM } },
    { text: "\n開場我們先不講 SGLang——先問一個更大的問題：", options: { color: INK } },
    { text: "任何分散式系統，理論上都會遇到哪些共同問題？", options: { bold: true, color: COMP } },
  ], { x: MX + 0.55, y: 4.45, w: 10.8, h: 1.25, valign: "middle", fontFace: HEAD, fontSize: 15, lineSpacingMultiple: 1.35, margin: 0 });
  footer(s, P0);
})();

// ============================================================ 2 八問題全景
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "01", "回顧主幹：SGLang 一路上遇到的八個問題", PURP);
  s.addText("同一條線，第三堂走完前四個（單機），本堂走後四個（多機）。", { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  const rows = [
    ["①", "LLM 程式難寫又跑不快", "前端 DSL：把程式描述成執行圖", "編程模型", FOOTC, false],
    ["②", "共享前綴被反覆重算", "分頁 KV + RadixAttention + continuous batching", "單機記憶體", FOOTC, false],
    ["③", "結構化輸出太慢、格式不保證", "Compressed FSM：預編譯 + 位元遮罩", "單機生成", FOOTC, false],
    ["④", "GPU 太快，CPU 排程成了瓶頸", "zero-overhead scheduler / CUDA Graph", "單機排程", FOOTC, false],
    ["⑤", "大 MoE 單機放不下、專家負載不均", "大規模 EP + DeepEP / EPLB", "多卡", MEM, true],
    ["⑥", "prefill 與 decode 互相干擾", "PD 分離", "多卡", COMP, true],
    ["⑦", "多副本：局部性 vs 負載均衡此消彼長", "cache-aware router + KV 複製", "多機", PURP, true],
    ["⑧", "副本掛掉，請求與 KV 怎麼辦", "容錯", "多機", WARN, true],
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
  s.addShape(pres.shapes.LINE, { x: MX, y: 1.85 + 4 * 0.54 - 0.03, w: 11.9, h: 0, line: { color: PURP, width: 1.5, dashType: "dash" } });
  takeaway(s, "線以下這四個，全都來自同一件事：不只一台。", PURP);
  footer(s, P0);
})();

// ============================================================ 3 先問一個更大的問題
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "02", "先問：任何分散式系統都會遇到哪些共同問題？", COMP);
  s.addText("先別想 GPU。這是分散式系統教科書的標準分類——請聽眾自己列，通常會列出下面這八類。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  const cls = [
    ["1", "時間與順序", "沒有全局時鐘；用 Lamport / Vector clock 近似因果順序"],
    ["2", "一致性", "多副本讀到最新值？CAP、強一致 vs 最終一致"],
    ["3", "容錯", "部分失效、拜占庭錯誤、故障偵測（只能做到「疑似故障」）"],
    ["4", "共識", "有故障與延遲時如何對某個值達成一致；Paxos / Raft、FLP"],
    ["5", "通訊", "訊息丟失、延遲、重複、亂序；網路分區；分不清「慢」與「掛」"],
    ["6", "並發與競態", "多節點操作共享資源；分散式鎖、2PC / 3PC"],
    ["7", "可擴展性", "協調開銷非線性成長；負載均衡、分片（sharding）"],
    ["8", "可觀測性", "狀態分散各處，沒有單一全局視角 → 除錯／監控／追蹤變難"],
  ];
  cls.forEach(([n, t, d], i) => {
    const x = MX + (i % 2) * 6.05, y = 1.85 + Math.floor(i / 2) * 1.02;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.85, h: 0.9, rectRadius: 0.07, fill: { color: BG2 }, line: { color: LINE, width: 1 } });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.15, y: y + 0.22, w: 0.46, h: 0.46, rectRadius: 0.08, fill: { color: COMP }, line: { type: "none" } });
    s.addText(n, { x: x + 0.15, y: y + 0.22, w: 0.46, h: 0.46, align: "center", valign: "middle", fontFace: MONO, fontSize: 15, bold: true, color: BG, margin: 0 });
    s.addText(t, { x: x + 0.75, y: y + 0.08, w: 5.0, h: 0.36, valign: "middle", fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0 });
    s.addText(d, { x: x + 0.75, y: y + 0.44, w: 5.0, h: 0.42, valign: "middle", fontFace: BODY, fontSize: 10.8, color: MUTE, margin: 0 });
  });
  takeaway(s, "下一頁：一格一格對照到 GPU 推論叢集。重點不是把不適用的劃掉，而是看清楚共同性。", COMP);
  footer(s, PA);
})();

// ============================================================ 4 對照表
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "03", "對照：在 GPU 推論叢集裡，這八類長什麼樣？", PURP);
  tableGrid(s, MX, 1.35, 11.9, [
    { t: "經典問題", w: 2.0 }, { t: "在 GPU 推論叢集裡", w: 7.6 }, { t: "判定", w: 2.3 },
  ], [
    ["1 時間與順序", "不需要邏輯時鐘（通訊模式是 SPMD 事先規劃的）；但變形成 barrier 同步與 pipeline bubble", "變形"],
    ["2 一致性", "沒有多副本各自更新再收斂；但 router 對「每台機器快取裡有什麼」只有近似且過時的視圖", "大部分規避 · 有影子"],
    ["3 容錯", "拜占庭不用（受信任封閉環境）；但部分失效被放大——一張卡掉，整個 TP group 廢掉", "簡化 · 但仍是核心"],
    ["4 共識", "計算路徑上不需要投票／選主；集合通訊是確定性的、框架事先規劃好的", "規避"],
    ["5 通訊", "可靠性由 RDMA / NCCL 保證；問題換了一根軸——變成頻寬、延遲、拓撲", "換軸：可靠性→效率"],
    ["6 並發與競態", "沒有衝突寫入 → 不需要鎖／事務；但 PD 分離的 KV 所有權轉移是最小形式的影子", "大部分規避 · 有影子"],
    ["7 可擴展性", "all-to-all 隨 EP 規模成長、局部性 vs 均衡、專家負載不均＝資料傾斜", "放大 → 核心"],
    ["8 可觀測性", "掉卡／慢節點偵測、各機命中率與 KV 佔用；router 要全局視角卻只有近似的", "存在"],
  ], PURP, 10.5, [COMP, WARN, WARN, FOOTC, MEM, FOOTC, WARN, MEM]);
  takeaway(s, "沒有一格是「不用管」——它們只是換了形態。橘/紅＝仍然要處理，灰＝真的規避掉了。", PURP);
  footer(s, PA);
})();

// ============================================================ 5 共同的底層結構
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "04", "共同的底層結構：三個物理事實", MEM);
  s.addText("所有分散式系統的問題，往下挖都來自這三件事——GPU 叢集一件都逃不掉。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  [["1", "沒有共享記憶體", "資訊必須靠傳遞，傳遞需要時間 → 任何「全局視角」都是過時的", MEM],
  ["2", "部分失效", "系統的一部分可以壞掉，其他部分不知道，也分不清「壞了」還是「慢了」", WARN],
  ["3", "協調需要通訊", "協調越多，通訊越多，擴展性越差", COMP]]
    .forEach(([n, t, d, c], i) => {
      const x = MX + i * 4.03;
      card(s, x, 1.8, 3.85, 1.65, BG2, c);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.2, y: 1.95, w: 0.5, h: 0.5, rectRadius: 0.09, fill: { color: c }, line: { type: "none" } });
      s.addText(n, { x: x + 0.2, y: 1.95, w: 0.5, h: 0.5, align: "center", valign: "middle", fontFace: MONO, fontSize: 17, bold: true, color: BG, margin: 0 });
      s.addText(t, { x: x + 0.8, y: 1.95, w: 2.9, h: 0.5, valign: "middle", fontFace: HEAD, fontSize: 15, bold: true, color: c, margin: 0 });
      s.addText(d, { x: x + 0.2, y: 2.55, w: 3.45, h: 0.8, valign: "top", fontFace: BODY, fontSize: 11.2, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
    });

  s.addText("那 GPU 叢集為什麼能「規避」拜占庭、共識、CAP？因為它主動放棄了兩樣東西", { x: MX, y: 3.65, w: 11.9, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: INK, margin: 0 });
  [["放棄「互不信任」", "封閉、受信任、同構的環境", "→ 拜占庭容錯不用做", GOOD],
  ["放棄「執行期協商」", "通訊模式由 SPMD 程式事先規劃", "→ 共識協定不用做", GOOD],
  ["（訓練）放棄「持續可用」", "fail-stop + checkpoint 重啟即可", "→ CAP 取捨不存在", MUTE]]
    .forEach(([t, d, r, c], i) => {
      const y = 4.1 + i * 0.6;
      s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 11.9, h: 0.53, fill: { color: BG2 }, line: { color: c === MUTE ? LINE : c, width: 1 } });
      s.addText(t, { x: MX + 0.22, y, w: 3.0, h: 0.53, valign: "middle", fontFace: HEAD, fontSize: 13, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 3.3, y, w: 4.6, h: 0.53, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
      s.addText(r, { x: MX + 8.0, y, w: 3.7, h: 0.53, align: "right", valign: "middle", fontFace: HEAD, fontSize: 12, bold: true, color: c, margin: 0 });
    });
  takeaway(s, "代價是：全部的複雜度被壓到「效率」這一根軸上。經典分散式系統問「怎麼保證正確」，GPU 叢集問「怎麼不浪費」。", MEM);
  footer(s, PA);
})();

// ============================================================ 6 關鍵轉折
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "05", "關鍵轉折：推論服務比訓練更像分散式系統", WARN);
  s.addText("上一頁第三個「放棄」，在推論這邊收不回來——這正是第二堂與本堂的分水嶺。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  tableGrid(s, MX, 1.8, 11.9, [
    { t: "", w: 2.4 }, { t: "訓練（第二堂）", w: 4.5 }, { t: "推論服務（本堂）", w: 5.0 },
  ], [
    ["性質", "一個 batch job", "長期在線服務，有 SLO"],
    ["掛一台怎麼辦", "整個 job 停下、從 checkpoint 重啟即可", "不能停整個服務 → 可用性重新變成真問題"],
    ["有狀態散在各機嗎", "參數每步同步，沒有分歧", "有：KV cache 散在各副本，router 只有近似視圖"],
    ["負載", "事先決定、均勻", "執行期才知道、長度差異巨大、有熱點"],
  ], WARN, 11.5);
  card(s, MX, 4.2, 11.9, 1.55, BG2, WARN);
  s.addText("所以「可用性」與「快取視圖不一致」這兩件在訓練場景可以忽略的事，在推論服務裡部分回來了。",
    { x: MX + 0.28, y: 4.32, w: 11.3, h: 0.4, valign: "middle", fontFace: HEAD, fontSize: 15, bold: true, color: WARN, margin: 0 });
  s.addText("這正好解釋了為什麼第二堂（訓練式多卡）不需要 router 與容錯策略，而本堂需要。也解釋了為什麼推論服務的工程，長得比訓練更像後端系統——只是名詞不同。",
    { x: MX + 0.28, y: 4.8, w: 11.3, h: 0.85, valign: "top", fontFace: BODY, fontSize: 12.5, color: MUTE, lineSpacingMultiple: 1.35, margin: 0 });
  takeaway(s, "有了這張地圖，後面四個問題就不是零散的技術點，而是可以掛回去的老問題。", WARN);
  footer(s, PA);
})();

// ============================================================ 7 問題⑤
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "06", "問題⑤：大 MoE 單機放不下", MEM);
  probStepper(s, 4);
  problemBanner(s, "⑤", "MoE 在單卡上不省容量——所有專家都得待在 HBM",
    "「每 token 只活躍一小部分專家」省的是每 token 的 FLOPs 與權重讀取，不是容量。DeepSeek-V3 的 671B 權重，一顆 HBM 根本裝不下。要連容量也省，只有一條路：把專家散到多張卡。（掛回經典問題 7：可擴展性 + 分片）", MEM);
  card(s, MX, 4.05, 5.8, 1.85, BG2, WARN);
  s.addText("單卡放整份 MoE", { x: MX + 0.25, y: 4.15, w: 5.3, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: WARN, margin: 0 });
  for (let i = 0; i < 16; i++) {
    s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.35 + (i % 8) * 0.62, y: 4.58 + Math.floor(i / 8) * 0.48, w: 0.54, h: 0.4, fill: { color: i < 2 ? COMP : BG3 }, line: { color: LINE, width: 0.8 } });
  }
  s.addText("活躍 2 / 16，但 16 個都佔 HBM", { x: MX + 0.25, y: 5.55, w: 5.3, h: 0.3, fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });

  card(s, 7.0, 4.05, 5.6, 1.85, BG2, GOOD);
  s.addText("EP：專家散到多卡", { x: 7.25, y: 4.15, w: 5.1, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: GOOD, margin: 0 });
  [0, 1, 2, 3].forEach((g) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.25 + g * 1.32, y: 4.55, w: 1.2, h: 0.85, rectRadius: 0.06, fill: { color: BG3 }, line: { color: GOOD, width: 1 } });
    s.addText(`GPU ${g}`, { x: 7.25 + g * 1.32, y: 4.58, w: 1.2, h: 0.24, align: "center", fontFace: MONO, fontSize: 9, color: GOOD, margin: 0 });
    for (let i = 0; i < 4; i++) {
      s.addShape(pres.shapes.RECTANGLE, { x: 7.34 + g * 1.32 + (i % 2) * 0.52, y: 4.86 + Math.floor(i / 2) * 0.26, w: 0.45, h: 0.2, fill: { color: (g === 1 && i === 0) || (g === 3 && i === 2) ? COMP : BG2 }, line: { color: LINE, width: 0.6 } });
    }
  });
  s.addText("每卡只放 4 個 → 每卡讀的位元組數變少 → decode 步時下降", { x: 7.25, y: 5.55, w: 5.1, h: 0.3, fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });
  footer(s, P5);
})();

// ============================================================ 8 EP 的代價 + EPLB
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "07", "EP 的代價：通訊變主角，而且負載會傾斜", WARN);
  card(s, MX, 1.4, 11.9, 1.35, BG2, COMP);
  s.addText("代價①：每層兩次 all-to-all", { x: MX + 0.28, y: 1.5, w: 5, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: COMP, margin: 0 });
  obox(s, MX + 0.4, 1.95, 2.4, 0.6, "token 送去對應專家", COMP, COMP, 11);
  arrow(s, 3.6, 2.25, 0.8, MUTE);
  obox(s, 4.55, 1.95, 2.4, 0.6, "各卡算自己的專家", MEM, MEM, 11);
  arrow(s, 7.15, 2.25, 0.8, MUTE);
  obox(s, 8.1, 1.95, 2.4, 0.6, "算完送回原卡", COMP, COMP, 11);
  s.addText("→ 回到第二堂的頻寬階梯：EP 通常必須綁在 NVLink 域內", { x: 10.7, y: 1.95, w: 1.9, h: 0.6, valign: "middle", fontFace: BODY, fontSize: 10, color: FOOTC, margin: 0 });

  card(s, MX, 2.9, 11.9, 1.75, BG2, WARN);
  s.addText("代價②：專家負載不均 ＝ 經典的資料傾斜（data skew）", { x: MX + 0.28, y: 3.0, w: 8, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: WARN, margin: 0 });
  const loads = [0.95, 0.3, 0.2, 0.55, 0.25, 0.85, 0.15, 0.35];
  loads.forEach((v, i) => {
    const x = MX + 0.5 + i * 1.42, h = 0.62 * v;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 4.02 - h, w: 1.05, h, fill: { color: v > 0.8 ? WARN : MEM }, line: { type: "none" } });
    s.addText(`E${i}`, { x, y: 4.04, w: 1.05, h: 0.2, align: "center", fontFace: MONO, fontSize: 8.5, color: FOOTC, margin: 0 });
  });
  s.addText("熱門專家所在的卡塞爆、其他卡閒著 → 整體被最慢那張卡拖住", { x: MX + 0.28, y: 4.28, w: 11.3, h: 0.28, fontFace: BODY, fontSize: 11, color: WARN, margin: 0 });

  [["推論期解法", "EPLB（Expert Parallelism Load Balancer）：熱門專家做副本、重排到不同卡", GOOD],
  ["訓練期解法", "DeepSeek 的無輔助損失偏置調整：動態調路由 bias 維持均衡，不干擾主目標", MEM],
  ["配套 kernel", "DeepEP（all-to-all 通訊庫）、DeepGEMM（FP8 GEMM）、FlashMLA —— DeepSeek 開源、SGLang 整合", PURP]]
    .forEach(([t, d, c], i) => {
      const y = 4.8 + i * 0.42;
      s.addShape(pres.shapes.RECTANGLE, { x: MX, y, w: 11.9, h: 0.38, fill: { color: BG2 }, line: { color: c, width: 1 } });
      s.addText(t, { x: MX + 0.2, y, w: 2.2, h: 0.38, valign: "middle", fontFace: HEAD, fontSize: 11.5, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 2.5, y, w: 9.2, h: 0.38, valign: "middle", fontFace: BODY, fontSize: 11, color: MUTE, margin: 0 });
    });
  takeaway(s, "資料傾斜是分散式資料庫的老問題（分片熱點）——只是這裡的「分片鍵」是 router 選的專家。", WARN);
  footer(s, P5);
})();

// ============================================================ 9 落地數字
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "08", "落地數字：大規模 EP 到底值不值得？", GOOD);
  tableGrid(s, MX, 1.45, 11.9, [
    { t: "系統", w: 3.2 }, { t: "硬體", w: 2.2 }, { t: "數字", w: 3.6 }, { t: "備註", w: 2.9 },
  ], [
    ["DeepSeek 官方 V3/R1 推論系統", "H800 節點（8 卡）", "73.7k in / 14.8k out tok/s 每節點", "prefill EP32、decode EP144"],
    ["同上 · 成本（自陳理論值）", "$2/GPU·hr 假設", "日成本 $87,072 → 日營收 $562,027", "理論成本利潤率 545%"],
    ["SGLang 開源復現", "96× H100（12 節點）", "52.3k in / 22.3k out tok/s 每節點", "相對同資源純 TP，輸出吞吐最高 5×"],
    ["SGLang on GB200 NVL72", "GB200 NVL72", "26,156 in / 13,386 out tok/s 每 GPU", "FP8 attn + NVFP4 MoE；vs H100 prefill 3.8× / decode 4.8×"],
  ], GOOD, 11);
  card(s, MX, 3.85, 11.9, 1.85, BG2, GOOD);
  s.addText("讀這張表的重點", { x: MX + 0.28, y: 3.96, w: 6, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: GOOD, margin: 0 });
  s.addText("「大規模 EP + PD 分離」相對「naive TP 部署」是數倍差距——同一份權重、同一批卡。\n第三列尤其重要：那是開源社群在沒有 DeepSeek 內部程式碼的情況下，用 SGLang 復現到接近官方數字。",
    { x: MX + 0.28, y: 4.38, w: 11.3, h: 1.1, valign: "top", fontFace: BODY, fontSize: 13, color: MUTE, lineSpacingMultiple: 1.4, margin: 0 });
  takeaway(s, "⚠️ 各家自報的最佳配置數字，工作負載（輸入/輸出長度比）不同結果差很多——當量級參考，不是保證。", COMP);
  footer(s, P5);
})();

// ============================================================ 10 問題⑥
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "09", "問題⑥：prefill 與 decode 的最佳配置根本不同", COMP);
  probStepper(s, 5);
  problemBanner(s, "⑥", "第三堂的 chunked prefill 只是緩解，沒有根治",
    "把長 prompt 切塊混批，確實讓 decode 不斷流。但兩者的硬體需求本質不同：prefill 要 FLOPs、decode 要頻寬與 KV 容量——擠在同一台機器上，只能取一個折衷點，兩邊都不是最佳。（掛回經典問題 5：通訊效率）", COMP);
  card(s, MX, 4.05, 5.4, 1.85, BG2, COMP);
  s.addText("Prefill 群 · compute-bound", { x: MX + 0.25, y: 4.15, w: 4.9, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: COMP, margin: 0 });
  [["最佳化目標", "FLOPs 利用率"], ["平行策略", "大 TP"], ["批次", "小批、追 TTFT"]].forEach(([k, v], i) => {
    const y = 4.55 + i * 0.42;
    s.addText(k, { x: MX + 0.25, y, w: 2.0, h: 0.38, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
    s.addText(v, { x: MX + 2.3, y, w: 2.8, h: 0.38, valign: "middle", fontFace: HEAD, fontSize: 11.5, bold: true, color: INK, margin: 0 });
  });
  card(s, 7.2, 4.05, 5.4, 1.85, BG2, MEM);
  s.addText("Decode 群 · memory-bound", { x: 7.45, y: 4.15, w: 4.9, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: MEM, margin: 0 });
  [["最佳化目標", "頻寬 + KV 容量"], ["平行策略", "大 EP"], ["批次", "大批、追吞吐"]].forEach(([k, v], i) => {
    const y = 4.55 + i * 0.42;
    s.addText(k, { x: 7.45, y, w: 2.0, h: 0.38, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
    s.addText(v, { x: 9.5, y, w: 2.8, h: 0.38, valign: "middle", fontFace: HEAD, fontSize: 11.5, bold: true, color: INK, margin: 0 });
  });
  arrow(s, 6.0, 4.95, 1.05, PURP);
  s.addText("傳 KV", { x: 5.8, y: 4.55, w: 1.45, h: 0.3, align: "center", fontFace: MONO, fontSize: 10.5, bold: true, color: PURP, margin: 0 });
  footer(s, P6);
})();

// ============================================================ 11 PD 分離的新問題
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "10", "PD 分離：好處、代價，與一個舊問題的影子", PURP);
  [["好處①", "兩邊各自開到最佳操作點", "prefill 群調 TP 追 TTFT、decode 群調 EP 追吞吐，不用再折衷", GOOD],
  ["好處②", "可以分開擴縮容", "prefill 與 decode 的負載比例會隨流量型態變動（長 prompt 多還是長輸出多）", GOOD],
  ["代價", "多一次 KV 跨機傳輸 + 一層調度複雜度", "實作：vLLM 的 KV connector 抽象（接 LMCache + NVIDIA NIXL）、SGLang 的 PD 分離、NVIDIA Dynamo", COMP]]
    .forEach(([t, d, r, c], i) => {
      const y = 1.4 + i * 1.15;
      card(s, MX, y, 11.9, 1.02, BG2, c);
      s.addText(t, { x: MX + 0.25, y: y + 0.08, w: 1.5, h: 0.42, valign: "middle", fontFace: HEAD, fontSize: 14, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 1.8, y: y + 0.08, w: 9.8, h: 0.42, valign: "middle", fontFace: HEAD, fontSize: 13.5, bold: true, color: INK, margin: 0 });
      s.addText(r, { x: MX + 1.8, y: y + 0.5, w: 9.8, h: 0.45, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
    });
  card(s, MX, 4.95, 11.9, 0.95, BG2, WARN);
  s.addText("🔍 舊問題的影子：KV 的「所有權轉移」", { x: MX + 0.28, y: 5.03, w: 5.5, h: 0.35, valign: "middle", fontFace: HEAD, fontSize: 14, bold: true, color: WARN, margin: 0 });
  s.addText("prefill 機算完把 KV 交給 decode 機——中途誰負責釋放？decode 機還沒接手就掛了怎麼辦？這是經典問題 6（並發／事務）在這個領域的最小形式。",
    { x: MX + 0.28, y: 5.38, w: 11.3, h: 0.45, valign: "middle", fontFace: BODY, fontSize: 11.8, color: MUTE, margin: 0 });
  takeaway(s, "接第二堂：KV 要跨機器搬，走 NVLink 域內還是 IB/Spectrum-X，直接決定 PD 分離划不划算。", PURP);
  footer(s, P6);
})();

// ============================================================ 12 問題⑦
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "11", "問題⑦：多副本各有自己的 radix tree", PURP);
  probStepper(s, 6);
  problemBanner(s, "⑦", "請求該送給誰？兩個目標互相打架",
    "每台機器維護獨立的 radix tree，彼此的 KV cache 不共享。想提高命中率就要把相關請求集中在同一台；想均衡負載就要把請求打散。（掛回經典問題 7 可擴展性 + 2 一致性 + 8 可觀測性）", PURP);
  card(s, MX, 4.05, 5.8, 1.9, BG2, WARN);
  s.addText("Round Robin（輪詢）", { x: MX + 0.25, y: 4.15, w: 5.3, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: WARN, margin: 0 });
  s.addText("完全不看內容 → 本該共享前綴、集中在同一台處理的請求被打散", { x: MX + 0.25, y: 4.52, w: 5.3, h: 0.6, valign: "top", fontFace: BODY, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  pill(s, MX + 0.25, 5.25, 5.3, 0.5, "命中率崩潰", WARN, WARNTINT, WARN, 12.5);

  card(s, 7.0, 4.05, 5.6, 1.9, BG2, COMP);
  s.addText("內容感知路由", { x: 7.25, y: 4.15, w: 5.1, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: COMP, margin: 0 });
  s.addText("同一份文件的請求固定送同一台 → 該機器負載暴漲、其他機器閒置", { x: 7.25, y: 4.52, w: 5.1, h: 0.6, valign: "top", fontFace: BODY, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  pill(s, 7.25, 5.25, 5.1, 0.5, "熱點（hot spot）", COMP, COMPTINT, COMP, 12.5);
  footer(s, P7);
})();

// ============================================================ 13 cache-aware router
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "12", "解法⑦：cache-aware load balancing", GOOD);
  s.addText("Router 同時看兩件事，並在必要時主動複製快取——這是本堂的核心機制。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  card(s, MX, 1.8, 5.8, 1.55, BG2, PURP);
  s.addText("看①：前綴命中程度", { x: MX + 0.25, y: 1.9, w: 5.3, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: PURP, margin: 0 });
  s.addText("把請求送去「已經有這段前綴」的機器（router 自己維護一份近似的 radix tree 視圖）",
    { x: MX + 0.25, y: 2.28, w: 5.3, h: 0.95, valign: "top", fontFace: BODY, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
  card(s, 7.0, 1.8, 5.6, 1.55, BG2, MEM);
  s.addText("看②：該機負載", { x: 7.25, y: 1.9, w: 5.1, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: MEM, margin: 0 });
  s.addText("佇列積壓、KV 池佔用、正在跑的序列數——超過閾值就不能再往那台塞",
    { x: 7.25, y: 2.28, w: 5.1, h: 0.95, valign: "top", fontFace: BODY, fontSize: 11.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });

  card(s, MX, 3.5, 11.9, 1.35, BG2, GOOD);
  s.addText("兩者衝突時：把高價值的快取複製出去", { x: MX + 0.28, y: 3.6, w: 6, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: GOOD, margin: 0 });
  s.addText("當某台負載超過閾值、且它上面有「被大量複用」的前綴時，主動把該段 KV 複製（replicate）到空閒機器 → 新請求可以分流，命中率不掉。",
    { x: MX + 0.28, y: 4.0, w: 11.3, h: 0.75, valign: "top", fontFace: BODY, fontSize: 12.5, color: MUTE, lineSpacingMultiple: 1.35, margin: 0 });

  card(s, MX, 5.0, 11.9, 0.9, BG2, WARN);
  s.addText("⚠️ 這裡藏著經典問題 2（一致性）：router 不可能即時知道每台的確切快取狀態，它手上永遠是一份「近似且過時」的視圖——決策品質的上限由此決定。",
    { x: MX + 0.28, y: 5.0, w: 11.3, h: 0.9, valign: "middle", fontFace: BODY, fontSize: 12.5, color: WARN, margin: 0 });
  takeaway(s, "業界對應：SGLang Router、Mooncake（Kimi）、vLLM 的 LMCache/NIXL、NVIDIA Dynamo 的 KV-aware routing。", GOOD);
  footer(s, P7);
})();

// ============================================================ 14 壓軸推導
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "13", "壓軸推導：該搬，還是該重算？", COMP);
  s.addText("複製 KV 不是免費的。這一頁把第二堂（頻寬階梯）、第三堂（KV 每 token 多少 bytes）與本堂的路由決策串成一條線。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 12.5, color: MUTE, margin: 0 });
  card(s, MX, 1.8, 11.9, 0.85, BG2, COMP);
  s.addText("搬的成本 ＝ KV 大小 ÷ 互連頻寬　　vs　　重算的成本 ＝ 該段前綴的 prefill 時間",
    { x: MX + 0.28, y: 1.8, w: 11.3, h: 0.85, align: "center", valign: "middle", fontFace: MONO, fontSize: 16, bold: true, color: COMP, margin: 0 });
  s.addText("以 Llama-3-8B（128 KB/token）估算", { x: MX, y: 2.82, w: 6, h: 0.32, fontFace: HEAD, fontSize: 14, bold: true, color: INK, margin: 0 });
  tableGrid(s, MX, 3.2, 11.9, [
    { t: "前綴長度", w: 2.2 }, { t: "KV 大小", w: 2.0 }, { t: "走 NVLink（~900 GB/s）", w: 2.9 }, { t: "走 IB（~50 GB/s）", w: 2.4 }, { t: "重新 prefill", w: 2.4 },
  ], [
    ["2,000 token", "256 MB", "~0.3 ms", "~5 ms", "數十 ms"],
    ["32,000 token", "4 GB", "~4.5 ms", "~80 ms", "數百 ms ~ 秒級"],
  ], MEM, 11.5);
  card(s, MX, 4.65, 11.9, 1.2, BG2, GOOD);
  s.addText("三條決策原則", { x: MX + 0.28, y: 4.73, w: 4, h: 0.3, fontFace: HEAD, fontSize: 14, bold: true, color: GOOD, margin: 0 });
  s.addText("① NVLink 域內幾乎永遠該搬　　② 跨節點走乙太要算清楚，短前綴常常不如重算　　③ 前綴越長越該搬——因為重算是 O(n²)、搬是 O(n)",
    { x: MX + 0.28, y: 5.08, w: 11.3, h: 0.7, valign: "top", fontFace: BODY, fontSize: 12.5, color: MUTE, lineSpacingMultiple: 1.35, margin: 0 });
  takeaway(s, "數字為量級估算（未計協定開銷與 prefill 的實際 FLOPs 曲線），用來建立決策直覺，不是設定閾值的依據。", COMP);
  footer(s, P7);
})();

// ============================================================ 15 問題⑧
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "14", "問題⑧：副本掛掉，請求與 KV 怎麼辦？", WARN);
  probStepper(s, 7);
  problemBanner(s, "⑧", "部分失效——分散式系統最本質的特徵，在這裡被放大",
    "一張卡掉，整個 TP group 就廢了（TP 內部沒有容錯）。而且分不清「掛了」還是「慢了」——straggler 與 dead node 在監控上長得很像。（掛回經典問題 3：部分失效 + 故障偵測）", WARN);
  card(s, MX, 4.05, 11.9, 1.85, BG2, GOOD);
  s.addText("🔑 鑰匙：KV cache 是「可重算的快取」，不是「不可回復的狀態」", { x: MX + 0.28, y: 4.15, w: 9, h: 0.38, fontFace: HEAD, fontSize: 16, bold: true, color: GOOD, margin: 0 });
  [["KV 完全由 prompt token 決定", "掉了可以重算，只是慢，不會錯"],
  ["所以預設策略是「重送 + 重新 prefill」", "而不是「想辦法把 KV 救回來」"],
  ["推論：KV 複製的動機不是可靠性", "而是省算力——問題⑦ 與 ⑧ 用同一套機制，動機完全不同"]]
    .forEach(([a, b], i) => {
      const y = 4.6 + i * 0.42;
      s.addText("· " + a, { x: MX + 0.35, y, w: 5.5, h: 0.38, valign: "middle", fontFace: BODY, fontSize: 11.8, color: MUTE, margin: 0 });
      s.addText("→ " + b, { x: MX + 6.0, y, w: 5.7, h: 0.38, valign: "middle", fontFace: HEAD, fontSize: 11.8, bold: true, color: GOOD, margin: 0 });
    });
  footer(s, P8);
})();

// ============================================================ 16 容錯的待答問題
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "15", "容錯：四個還沒有標準答案的問題", WARN);
  s.addText("這一頁誠實標示邊界——各家公開資訊都不多，適合當現場開放討論。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  [["1", "串流到一半的請求怎麼重試？", "使用者已經看到半句話了。重頭來過（畫面閃一下）還是接續（要把已輸出的 token 當 prompt 重新 prefill）？後者較好但成本更高。", WARN],
  ["2", "PD 分離下 decode 機掛掉", "prefill 已完成、KV 已傳過去、算力已經花了——這是最貴的失敗模式。KV 要不要在 prefill 端留一份？留多久？", COMP],
  ["3", "「掛了」與「慢了」怎麼分？", "健康檢查逾時設太短 → 誤判把好機器踢掉；設太長 → 請求卡死。這是經典的故障偵測難題，只能做到「疑似故障」。", MEM],
  ["4", "TP group 內部沒有容錯", "一張卡掉，整組廢掉。副本粒度越大（TP=8），單點故障的爆炸半徑越大——這會反過來影響你怎麼切模型。", PURP]]
    .forEach(([n, t, d, c], i) => {
      const y = 1.8 + i * 1.08;
      card(s, MX, y, 11.9, 0.95, BG2, c);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX + 0.2, y: y + 0.22, w: 0.5, h: 0.5, rectRadius: 0.09, fill: { color: c }, line: { type: "none" } });
      s.addText(n, { x: MX + 0.2, y: y + 0.22, w: 0.5, h: 0.5, align: "center", valign: "middle", fontFace: MONO, fontSize: 16, bold: true, color: BG, margin: 0 });
      s.addText(t, { x: MX + 0.85, y: y + 0.06, w: 10.8, h: 0.38, valign: "middle", fontFace: HEAD, fontSize: 14, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 0.85, y: y + 0.44, w: 10.8, h: 0.45, valign: "middle", fontFace: BODY, fontSize: 11.2, color: MUTE, margin: 0 });
    });
  takeaway(s, "第 4 點值得記：容錯需求會回頭改變你的平行策略——TP 切得越大，爆炸半徑越大。", WARN);
  footer(s, P8);
})();

// ============================================================ 17 換了名字的老問題
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "16", "回頭看：三個換了名字的老問題", MEM);
  s.addText("本堂的問題，其實都能在分散式系統的老領域裡找到同一個結構。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  [["專家負載不均（⑤）", "資料傾斜（data skew）", "分散式資料庫的分片熱點", COMP],
  ["局部性 vs 負載均衡（⑦）", "locality vs balance 的取捨", "CDN 邊緣快取放置、一致性雜湊、分散式快取", PURP],
  ["KV 該搬還是該重算（⑦）", "快取失效成本 vs 重建成本", "任何多層快取系統的核心決策", MEM]]
    .forEach(([a, b, c, col], i) => {
      const y = 1.85 + i * 1.28;
      card(s, MX, y, 11.9, 1.12, BG2, col);
      s.addText(a, { x: MX + 0.3, y: y + 0.12, w: 3.6, h: 0.45, valign: "middle", fontFace: HEAD, fontSize: 14.5, bold: true, color: INK, margin: 0 });
      s.addText("其實是", { x: MX + 4.0, y: y + 0.12, w: 0.9, h: 0.45, align: "center", valign: "middle", fontFace: BODY, fontSize: 10.5, color: FOOTC, margin: 0 });
      s.addText(b, { x: MX + 5.0, y: y + 0.12, w: 6.6, h: 0.45, valign: "middle", fontFace: HEAD, fontSize: 14.5, bold: true, color: col, margin: 0 });
      s.addText("老領域裡的同一個問題：" + c, { x: MX + 0.3, y: y + 0.6, w: 11.3, h: 0.42, valign: "middle", fontFace: BODY, fontSize: 11.8, color: MUTE, margin: 0 });
    });
  card(s, MX, 5.75, 11.9, 0.75, BG2, GOOD);
  s.addText("所以：如果你做過分散式資料庫或 CDN，這一堂大部分內容你們已經會了——只是名詞換了。",
    { x: MX + 0.28, y: 5.75, w: 11.3, h: 0.75, align: "center", valign: "middle", fontFace: HEAD, fontSize: 15, bold: true, color: GOOD, margin: 0 });
  footer(s, PZ);
})();

// ============================================================ 18 橫向比較
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "17", "橫向比較：四家的架構取向", PURP);
  s.addText("不比功能勾選表（大家都在快速補齊），比「把哪一件事當成一等公民」。",
    { x: MX, y: 1.35, w: 11.9, h: 0.32, fontFace: BODY, fontSize: 13, color: MUTE, margin: 0 });
  [["SGLang", "把「大規模 EP + PD 分離」做成開源實作", "首個在 96×H100 上復現 DeepSeek 級部署的開源方案；自帶 cache-aware router，把路由當成引擎的一部分", MEM],
  ["vLLM", "把 KV 傳輸抽象成 connector，其餘交給生態系", "KV connector + LMCache / NIXL；多副本路由由 production-stack / llm-d / NVIDIA Dynamo 承擔——分工而非全包", COMP],
  ["Mooncake（Kimi）", "把 KV cache pool 提升為一等公民", "KVCache-centric 架構：全域 KV 池（DRAM/SSD 分層），調度圍繞「KV 在哪」而不是「機器在哪」", PURP],
  ["DeepSeek", "模型與推論系統 co-design，零件全開源", "自家 EP32/EP144 推論系統 + FlashMLA / DeepEP / DeepGEMM / EPLB —— 讓別人跑得動自己的架構", GOOD]]
    .forEach(([t, tag, d, c], i) => {
      const y = 1.8 + i * 1.08;
      card(s, MX, y, 11.9, 0.95, BG2, c);
      s.addText(t, { x: MX + 0.25, y: y + 0.06, w: 2.6, h: 0.4, valign: "middle", fontFace: HEAD, fontSize: 15, bold: true, color: c, margin: 0 });
      s.addText(tag, { x: MX + 2.95, y: y + 0.06, w: 8.7, h: 0.4, valign: "middle", fontFace: HEAD, fontSize: 12.5, bold: true, color: INK, margin: 0 });
      s.addText(d, { x: MX + 0.25, y: y + 0.46, w: 11.4, h: 0.44, valign: "middle", fontFace: BODY, fontSize: 11.2, color: MUTE, margin: 0 });
    });
  s.addText("⚠️ 容錯這一欄四家公開資訊都不多，本表刻意留白——見第 15 頁的四個開放問題。生態演進很快，開講前請對一次官方文件。",
    { x: MX, y: 6.15, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 11, color: FOOTC, margin: 0 });
  footer(s, PZ);
})();

// ============================================================ 19 全系列位置
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "18", "這一堂在全系列的位置", MEM);
  const classes = [
    ["第一堂", "硬體本身", "roofline、記憶體階層、GPU 單元", FOOTC],
    ["第二堂", "一張卡 → 多張卡", "DP/TP/PP/EP 與 NVIDIA 互連（訓練式的 bulk-synchronous）", FOOTC],
    ["第三堂", "SGLang 單機篇", "問題 ①–④：把一台機器的 GPU 餵飽", MEM],
    ["第四堂", "SGLang 多機篇", "問題 ⑤–⑧：一群機器怎麼調度、怎麼不掛　← 本堂", PURP],
    ["第五堂", "中國開源模型", "模型從裡面改：五個旋鈕（壓 KV／少算／少看／一次多產／降精度）", COMP],
  ];
  classes.forEach(([n, t, d, c], i) => {
    const y = 1.6 + i * 0.86;
    const on = i === 3;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 11.9, h: 0.74, rectRadius: 0.07, fill: { color: on ? BG3 : BG2 }, line: { color: c, width: on ? 2 : 1 } });
    s.addText(n, { x: MX + 0.25, y, w: 1.5, h: 0.74, valign: "middle", fontFace: MONO, fontSize: 13, bold: true, color: c, margin: 0 });
    s.addText(t, { x: MX + 1.85, y, w: 3.0, h: 0.74, valign: "middle", fontFace: HEAD, fontSize: 15, bold: true, color: on ? INK : MUTE, margin: 0 });
    s.addText(d, { x: MX + 5.0, y, w: 6.7, h: 0.74, valign: "middle", fontFace: BODY, fontSize: 11.5, color: MUTE, margin: 0 });
  });
  card(s, MX, 6.0, 11.9, 0.75, BG2, GOOD);
  s.addText("三、四堂共用一條主幹（SGLang 的八個問題）；第五堂是另一個軸——框架從外面調 vs 模型從裡面改。",
    { x: MX + 0.28, y: 6.0, w: 11.3, h: 0.75, align: "center", valign: "middle", fontFace: HEAD, fontSize: 14, bold: true, color: GOOD, margin: 0 });
  footer(s, PZ);
})();

// ============================================================ 20 帶走三句話
(() => {
  const s = pres.addSlide(); base(s); runningHeader(s);
  header(s, "19", "帶走三句話", PURP);
  [["1", "GPU 叢集是「簡化過的」分散式系統", "它放棄了互不信任（→ 拜占庭免了）、放棄了執行期協商（→ 共識免了），代價是把全部複雜度壓到「效率」這一根軸上。經典系統問「怎麼保證正確」，它問「怎麼不浪費」。", MEM],
  ["2", "但推論服務把第三個放棄收回來了", "它是長期在線、有 SLO、狀態（KV）散在各機的服務——所以可用性、快取視圖不一致、故障偵測這些在訓練場景可忽略的事，全部回來了。", WARN],
  ["3", "多機的核心矛盾是局部性 vs 均衡", "集中提高命中率但製造熱點，打散均衡負載但命中率崩潰。cache-aware router 是折衷，KV 複製是緩解，而「該搬還是該重算」由 KV 大小 ÷ 互連頻寬 vs prefill 時間決定。", PURP]]
    .forEach(([n, t, d, c], i) => {
      const y = 1.85 + i * 1.45;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: 0.85, h: 0.85, rectRadius: 0.12, fill: { color: c }, line: { type: "none" } });
      s.addText(n, { x: MX, y, w: 0.85, h: 0.85, align: "center", valign: "middle", fontFace: MONO, fontSize: 30, bold: true, color: BG, margin: 0 });
      s.addText(t, { x: MX + 1.1, y, w: 11.0, h: 0.5, fontFace: HEAD, fontSize: 18, bold: true, color: c, margin: 0 });
      s.addText(d, { x: MX + 1.1, y: y + 0.52, w: 11.2, h: 0.9, valign: "top", fontFace: BODY, fontSize: 12.5, color: MUTE, lineSpacingMultiple: 1.3, margin: 0 });
    });
  s.addText("下一堂：模型端怎麼從裡面改——中國開源模型的五個旋鈕。",
    { x: MX, y: 6.25, w: 11.9, h: 0.4, fontFace: BODY, fontSize: 12.5, color: FOOTC, margin: 0 });
  footer(s, PZ);
})();

pres.writeFile({ fileName: "../class4_sglang_multi_node.pptx" }).then((f) => console.log("✅ 產生：" + f + "（" + PAGE + " 頁）")).catch((e) => console.error(e));
