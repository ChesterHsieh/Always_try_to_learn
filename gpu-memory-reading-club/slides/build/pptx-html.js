// pptx-html — 模擬 pptxgenjs 的子集合，把投影片腳本直接輸出成單一 HTML 檔。
//
// 用法：腳本裡 `const pptxgen = require("./pptx-html")`，其餘 API 照舊
// （addSlide / addText / addShape / addTable / addChart / writeFile）。
// 座標單位跟 pptxgenjs 一樣是英吋（LAYOUT_WIDE = 13.333 × 7.5），字級是 pt；
// 輸出時全部換算成舞台寬度的百分比與 cqw，所以投影片會隨視窗等比縮放。
// 只實作本系列腳本用到的功能：矩形／圓角矩形／橢圓／直線（含箭頭、虛線）、
// 文字段落（runs、bullet、breakLine、行距、段距）、表格、折線圖。
const fs = require("fs");
const path = require("path");

const SW = 13.333, SH = 7.5;
const DEFAULT_INSET = { l: 0.1, r: 0.1, t: 0.05, b: 0.05 };  // PowerPoint 文字框預設內距（英吋）
const CELL_INSET = { l: 0.1, r: 0.1, t: 0.05, b: 0.05 };
const BULLET_INDENT_PT = 20;
const FONT_STACKS = {
  "PingFang TC": "'PingFang TC','Noto Sans TC','Microsoft JhengHei',sans-serif",
  Menlo: "Menlo,'JetBrains Mono',Consolas,monospace",
};
// 講稿與測驗的對應：輸出檔名 → [堂次, 講稿檔名]
const DECKS = {
  full_series: [1, "full_series"],
  class2_transformer_gpu: [2, "class2_transformer_gpu"],
  class3_engine_single_node: [3, "class3_engine_single_node"],
  class4_sglang_multi_node: [4, "class4_sglang_multi_node"],
  class5_china_models: [5, "class5_china_models"],
  class6_multi_rack_inference: [6, "class6_multi_rack_inference"],
};

// ---------- 單位換算 ----------
const n = (v) => +(+v).toFixed(4);
const px = (inch, total) => `${n((inch / total) * 100)}%`;        // 位置／尺寸 → 舞台百分比
const cqIn = (inch) => `${n((inch / SW) * 100)}cqw`;               // 英吋 → cqw
const cqPt = (pt) => cqIn(pt / 72);                                // pt → cqw
const color = (c) => (c ? `#${String(c).replace(/^#/, "")}` : "transparent");
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const font = (f) => FONT_STACKS[f] || `'${f}',sans-serif`;  // 單引號：會被放進 style="..." 屬性

// 內文提到的互動教具檔名自動變成連結
const linkify = (html) => html.replace(/\b([a-z_]+_map\.html)\b/g, '<a href="../interactive/$1" target="_blank" rel="noopener">$1</a>');

function boxStyle(o) {
  return `left:${px(o.x || 0, SW)};top:${px(o.y || 0, SH)};width:${px(o.w || 0, SW)};height:${px(o.h || 0, SH)}`;
}

function lineCss(line) {
  if (!line || line.type === "none" || !line.color) return "";
  const w = line.width || line.pt || 0.75;
  return `border:${cqPt(w)} ${line.dashType === "dash" ? "dashed" : "solid"} ${color(line.color)};`;
}

function shadowCss(sh) {
  if (!sh) return "";
  const rad = ((sh.angle || 0) * Math.PI) / 180, off = sh.offset || 0;
  const hex = (sh.color || "000000").replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return `box-shadow:${cqPt(Math.cos(rad) * off)} ${cqPt(Math.sin(rad) * off)} ${cqPt(sh.blur || 0)} rgba(${r},${g},${b},${sh.opacity ?? 0.3});`;
}

// ---------- 文字 ----------
// pptxgenjs 的 runs → 段落：breakLine 結束一段；字串裡的 \n 也換段。
function toParagraphs(text) {
  const runs = Array.isArray(text) ? text : [{ text: String(text), options: {} }];
  const paras = [];
  let cur = { runs: [], opts: {} };
  const flush = () => { paras.push(cur); cur = { runs: [], opts: {} }; };
  runs.forEach((r) => {
    const o = r.options || {};
    const parts = String(r.text ?? "").split("\n");
    parts.forEach((part, i) => {
      if (i > 0) flush();
      if (!cur.runs.length) cur.opts = { ...o };
      cur.runs.push({ text: part, o });
    });
    if (o.breakLine) flush();
  });
  if (cur.runs.length) flush();
  return paras;
}

function runCss(o, base) {
  const css = [];
  if (o.color && o.color !== base.color) css.push(`color:${color(o.color)}`);
  if (o.fontSize && o.fontSize !== base.fontSize) css.push(`font-size:${cqPt(o.fontSize)}`);
  if (o.fontFace && o.fontFace !== base.fontFace) css.push(`font-family:${font(o.fontFace)}`);
  if (o.bold !== undefined && !!o.bold !== !!base.bold) css.push(`font-weight:${o.bold ? 700 : 400}`);
  if (o.italic) css.push("font-style:italic");
  return css.join(";");
}

function paragraphsHtml(text, base) {
  return toParagraphs(text).map((p) => {
    const po = { ...base, ...p.opts };
    const css = [];
    if (po.align && po.align !== base.align) css.push(`text-align:${po.align}`);
    if (po.paraSpaceAfter) css.push(`margin-bottom:${cqPt(po.paraSpaceAfter)}`);
    if (po.lineSpacingMultiple && po.lineSpacingMultiple !== base.lineSpacingMultiple) css.push(`line-height:${n(1.2 * po.lineSpacingMultiple)}`);
    const cls = po.bullet ? ' class="b"' : "";
    if (po.bullet) css.push(`padding-left:${cqPt(BULLET_INDENT_PT)}`);
    const inner = p.runs.map(({ text: t, o }) => {
      const rc = runCss(o, base);
      const h = linkify(esc(t));
      return rc ? `<span style="${rc}">${h}</span>` : h;
    }).join("") || "&#8203;";
    return `<p${cls}${css.length ? ` style="${css.join(";")}"` : ""}>${inner}</p>`;
  }).join("");
}

function textBoxHtml(text, o) {
  const m = o.margin === undefined ? DEFAULT_INSET : { l: o.margin / 72, r: o.margin / 72, t: o.margin / 72, b: o.margin / 72 };
  const valign = { top: "flex-start", middle: "center", bottom: "flex-end" }[o.valign || "middle"];
  const base = { color: o.color, fontSize: o.fontSize || 18, fontFace: o.fontFace, bold: !!o.bold, align: o.align || "left", lineSpacingMultiple: o.lineSpacingMultiple };
  const css = [
    boxStyle(o),
    `justify-content:${valign}`,
    `padding:${cqIn(m.t)} ${cqIn(m.r)} ${cqIn(m.b)} ${cqIn(m.l)}`,
    `color:${color(o.color || "000000")}`,
    `font-size:${cqPt(base.fontSize)}`,
    o.fontFace ? `font-family:${font(o.fontFace)}` : "",
    o.bold ? "font-weight:700" : "",
    o.italic ? "font-style:italic" : "",
    `text-align:${base.align}`,
    `line-height:${n(1.2 * (o.lineSpacingMultiple || 1))}`,
    o.charSpacing ? `letter-spacing:${cqPt(o.charSpacing)}` : "",
    o.fill && o.fill.color ? `background:${color(o.fill.color)}` : "",
    lineCss(o.line),
  ].filter(Boolean).join(";");
  return `<div class="t" style="${css}">${paragraphsHtml(text, base)}</div>`;
}

// ---------- 圖形 ----------
function shapeHtml(type, o) {
  if (type === "line") return lineHtml(o);
  const radius = type === "ellipse" ? "50%" : type === "roundRect" ? cqIn(o.rectRadius || 0) : "0";
  const css = [
    boxStyle(o),
    `background:${o.fill && o.fill.color ? color(o.fill.color) : "transparent"}`,
    `border-radius:${radius}`,
    lineCss(o.line),
    shadowCss(o.shadow),
  ].filter(Boolean).join(";");
  return `<div class="s" style="${css}"></div>`;
}

let markerSeq = 0;
function lineHtml(o) {
  const l = o.line || {};
  const x1 = o.x || 0, x2 = x1 + (o.w || 0);
  const [y1, y2] = o.flipV ? [(o.y || 0) + (o.h || 0), o.y || 0] : [o.y || 0, (o.y || 0) + (o.h || 0)];
  const sw = (l.width || 0.75) / 72, c = color(l.color || "FFFFFF");
  let defs = "", mEnd = "", mStart = "";
  const marker = (id) => `<marker id="${id}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="3.2" markerHeight="3.2" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="${c}"/></marker>`;
  if (l.endArrowType) { const id = `ah${++markerSeq}`; defs += marker(id); mEnd = ` marker-end="url(#${id})"`; }
  if (l.beginArrowType) { const id = `ah${++markerSeq}`; defs += marker(id); mStart = ` marker-start="url(#${id})"`; }
  const dash = l.dashType === "dash" ? ` stroke-dasharray="${n(sw * 4)} ${n(sw * 3)}"` : "";
  return `<svg class="ln" viewBox="0 0 ${SW} ${SH}" aria-hidden="true">${defs ? `<defs>${defs}</defs>` : ""}<line x1="${n(x1)}" y1="${n(y1)}" x2="${n(x2)}" y2="${n(y2)}" stroke="${c}" stroke-width="${n(sw)}"${dash}${mStart}${mEnd}/></svg>`;
}

// ---------- 表格 ----------
function tableHtml(rows, o) {
  const cols = o.colW || [];
  const b = o.border || {};
  const bcss = b.type === "none" ? "none" : `${cqPt(b.pt || 1)} solid ${color(b.color || "000000")}`;
  const colgroup = cols.map((w) => `<col style="width:${n((w / cols.reduce((a, c) => a + c, 0)) * 100)}%">`).join("");
  const body = rows.map((r) => `<tr style="height:${cqIn(o.rowH || 0.4)}">${r.map((cell) => {
    const c = typeof cell === "object" && cell !== null ? cell : { text: String(cell), options: {} };
    const co = c.options || {};
    const base = { color: o.color, fontSize: o.fontSize || 12, fontFace: o.fontFace, bold: false, align: o.align || "left" };
    const css = [
      `border:${bcss}`,
      co.fill && co.fill.color ? `background:${color(co.fill.color)}` : "",
      co.color ? `color:${color(co.color)}` : "",
      co.bold ? "font-weight:700" : "",
      co.fontSize ? `font-size:${cqPt(co.fontSize)}` : "",
      co.align ? `text-align:${co.align}` : "",
    ].filter(Boolean).join(";");
    return `<td style="${css}">${linkify(esc(c.text ?? ""))}</td>`;
  }).join("")}</tr>`).join("");
  const css = [
    `left:${px(o.x || 0, SW)}`, `top:${px(o.y || 0, SH)}`, `width:${px(o.w || 0, SW)}`,
    `font-size:${cqPt(o.fontSize || 12)}`, o.fontFace ? `font-family:${font(o.fontFace)}` : "",
    `color:${color(o.color || "000000")}`, `text-align:${o.align || "left"}`,
    `--pad:${cqIn(CELL_INSET.t)} ${cqIn(CELL_INSET.r)} ${cqIn(CELL_INSET.b)} ${cqIn(CELL_INSET.l)}`,
    `--va:${o.valign === "top" ? "top" : o.valign === "bottom" ? "bottom" : "middle"}`,
  ].filter(Boolean).join(";");
  return `<table class="tb" style="${css}"><colgroup>${colgroup}</colgroup>${body}</table>`;
}

// ---------- 折線圖（只支援單一或多條數列的 LINE chart） ----------
function niceMax(v) {
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((m) => m >= v);
}
function chartHtml(type, data, o) {
  const W = o.w, H = o.h;
  const fsIn = (pt) => pt / 72;
  const pad = { l: 0.85, r: 0.25, t: 0.25, b: 0.75 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const labels = data[0].labels;
  const vmax = niceMax(Math.max(...data.flatMap((d) => d.values)));
  const ticks = 5;
  const X = (i) => pad.l + (pw * (i + 0.5)) / labels.length;
  const Y = (v) => pad.t + ph * (1 - v / vmax);
  const lab = color(o.catAxisLabelColor || "888888"), grid = color((o.valGridLine && o.valGridLine.color) || "444444");
  let g = `<rect width="${W}" height="${H}" fill="${color(o.chartArea && o.chartArea.fill && o.chartArea.fill.color)}"/>`;
  for (let i = 0; i <= ticks; i++) {
    const v = (vmax / ticks) * i, y = Y(v);
    g += `<line x1="${pad.l}" x2="${n(pad.l + pw)}" y1="${n(y)}" y2="${n(y)}" stroke="${grid}" stroke-width="${n(((o.valGridLine && o.valGridLine.size) || 0.5) / 72)}"/>`;
    g += `<text x="${n(pad.l - 0.08)}" y="${n(y)}" fill="${lab}" font-size="${n(fsIn(o.valAxisLabelFontSize || 10))}" text-anchor="end" dominant-baseline="middle">${n(v)}</text>`;
  }
  labels.forEach((t, i) => {
    g += `<text x="${n(X(i))}" y="${n(pad.t + ph + 0.2)}" fill="${lab}" font-size="${n(fsIn(o.catAxisLabelFontSize || 10))}" text-anchor="middle">${esc(t)}</text>`;
  });
  data.forEach((d, di) => {
    const c = color((o.chartColors || [])[di] || "38BDF8");
    const pts = d.values.map((v, i) => `${n(X(i))},${n(Y(v))}`).join(" ");
    g += `<polyline points="${pts}" fill="none" stroke="${c}" stroke-width="${n((o.lineSize || 2) / 72)}" stroke-linejoin="round"/>`;
    if (o.lineDataSymbol) d.values.forEach((v, i) => { g += `<circle cx="${n(X(i))}" cy="${n(Y(v))}" r="${n((o.lineDataSymbolSize || 5) / 144)}" fill="${c}"/>`; });
  });
  if (o.showCatAxisTitle) g += `<text x="${n(pad.l + pw / 2)}" y="${n(H - 0.12)}" fill="${color(o.catAxisTitleColor)}" font-size="${n(fsIn(o.catAxisTitleFontSize || 11))}" text-anchor="middle">${esc(o.catAxisTitle)}</text>`;
  if (o.showValAxisTitle) g += `<text transform="translate(0.2 ${n(pad.t + ph / 2)}) rotate(-90)" fill="${color(o.valAxisTitleColor)}" font-size="${n(fsIn(o.valAxisTitleFontSize || 11))}" text-anchor="middle">${esc(o.valAxisTitle)}</text>`;
  return `<svg class="ch" style="${boxStyle(o)}" viewBox="0 0 ${n(W)} ${n(H)}" role="img" aria-label="${esc(data.map((d) => d.name).join("、"))}">${g}</svg>`;
}

// ---------- Slide / Presentation ----------
class Slide {
  constructor() { this.background = null; this.items = []; }
  addText(text, opts = {}) { this.items.push(textBoxHtml(text, opts)); return this; }
  addShape(type, opts = {}) { this.items.push(shapeHtml(type, opts)); return this; }
  addTable(rows, opts = {}) { this.items.push(tableHtml(rows, opts)); return this; }
  addChart(type, data, opts = {}) { this.items.push(chartHtml(type, data, opts)); return this; }
  addNotes() { return this; }
  html(i) {
    const bg = this.background && this.background.color ? `background:${color(this.background.color)}` : "";
    return `<section class="slide" id="s${i + 1}" aria-label="第 ${i + 1} 頁"><div class="stage" style="${bg}">${this.items.join("")}</div></section>`;
  }
}

class Presentation {
  constructor() {
    this.slides = [];
    this.layout = "LAYOUT_WIDE";
    this.title = "Slides";
    this.shapes = { RECTANGLE: "rect", ROUNDED_RECTANGLE: "roundRect", OVAL: "ellipse", LINE: "line" };
    this.charts = { LINE: "line" };
  }
  addSlide() { const s = new Slide(); this.slides.push(s); return s; }
  writeFile({ fileName }) {
    const out = path.resolve(__dirname, fileName);
    const key = path.basename(out, ".html");
    const [classNo, notes] = DECKS[key] || [null, null];
    const tpl = fs.readFileSync(path.join(__dirname, "deck-template.html"), "utf8");
    const html = tpl
      .replace(/{{TITLE}}/g, () => esc(this.title))
      .replace("{{NOTES_HREF}}", notes ? `../notes/view.html?doc=${notes}` : "../index.html")
      .replace("{{QUIZ_HREF}}", classNo ? `../quiz/index.html?class=${classNo}` : "../quiz/index.html")
      .replace("{{TOTAL}}", String(this.slides.length))
      .replace("{{SLIDES}}", () => this.slides.map((s, i) => s.html(i)).join("\n"));
    fs.writeFileSync(out, html);
    return Promise.resolve(out);
  }
}

module.exports = Presentation;
