export const C = {
  bg: "#F6F0E6",
  chalk: "#FFFDF8",
  ink: "#111827",
  muted: "#5F6B7A",
  faint: "#E8DFD2",
  line: "#D5CBBE",
  teal: "#1F6B67",
  orange: "#C46A2C",
  blue: "#315C9A",
  red: "#A33A2B",
  gold: "#B9933B",
  deep: "#0F172A",
  white: "#FFFFFF",
};

export function rect(slide, ctx, x, y, w, h, fill, opts = {}) {
  return ctx.addShape(slide, {
    left: x,
    top: y,
    width: w,
    height: h,
    fill,
    line: opts.line ?? ctx.line(opts.lineColor ?? "#00000000", opts.lineWidth ?? 0),
    geometry: opts.geometry ?? "rect",
    name: opts.name,
  });
}

export function text(slide, ctx, value, x, y, w, h, opts = {}) {
  return ctx.addText(slide, {
    text: String(value ?? ""),
    left: x,
    top: y,
    width: w,
    height: h,
    fontSize: opts.size ?? 22,
    color: opts.color ?? C.ink,
    bold: opts.bold ?? false,
    typeface: opts.face ?? (opts.serif ? "Aptos Display" : "Aptos"),
    align: opts.align ?? "left",
    valign: opts.valign ?? "top",
    fill: opts.fill ?? "#00000000",
    line: opts.line ?? ctx.line("#00000000", 0),
    insets: opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
    name: opts.name,
  });
}

export function rule(slide, ctx, x, y, w, color = C.line, weight = 1) {
  rect(slide, ctx, x, y, w, weight, color);
}

export function bg(slide, ctx) {
  rect(slide, ctx, 0, 0, ctx.W, ctx.H, C.bg);
}

export function footer(slide, ctx, n) {
  rule(slide, ctx, 56, 674, 1030, C.line, 1);
  text(slide, ctx, "Source: Albalak et al., A Survey on Data Selection for Language Models, TMLR 2024", 56, 684, 780, 20, {
    size: 10,
    color: C.muted,
  });
  text(slide, ctx, String(n).padStart(2, "0"), 1178, 678, 46, 26, {
    size: 14,
    color: C.muted,
    align: "right",
    face: "Aptos Mono",
  });
}

export function kicker(slide, ctx, label, color = C.teal) {
  rect(slide, ctx, 56, 42, 8, 22, color, { name: "kicker-marker" });
  text(slide, ctx, label.toUpperCase(), 76, 39, 300, 28, {
    size: 12,
    bold: true,
    color: C.muted,
    valign: "middle",
    name: "kicker-label",
  });
}

export function title(slide, ctx, claim) {
  text(slide, ctx, claim, 56, 78, 910, 82, {
    size: 34,
    bold: true,
    serif: true,
    color: C.ink,
  });
}

export function body(slide, ctx, value, x, y, w, h, opts = {}) {
  return text(slide, ctx, value, x, y, w, h, {
    size: opts.size ?? 17,
    color: opts.color ?? C.muted,
    ...opts,
  });
}

export function box(slide, ctx, x, y, w, h, fill, lineColor = C.line, name) {
  rect(slide, ctx, x, y, w, h, fill, { line: ctx.line(lineColor, 1), name });
}

export function labelBox(slide, ctx, label, detail, x, y, w, h, fill, accent = C.teal) {
  box(slide, ctx, x, y, w, h, fill, C.line);
  rect(slide, ctx, x, y, 6, h, accent);
  text(slide, ctx, label, x + 18, y + 14, w - 34, 24, { size: 17, bold: true, color: C.ink });
  text(slide, ctx, detail, x + 18, y + 44, w - 34, h - 56, { size: 13.5, color: C.muted });
}

export function chip(slide, ctx, value, x, y, w, fill = C.chalk, color = C.ink) {
  rect(slide, ctx, x, y, w, 30, fill, { line: ctx.line(C.line, 1) });
  text(slide, ctx, value, x + 12, y + 6, w - 24, 18, { size: 12, bold: true, color, valign: "middle" });
}

export function dot(slide, ctx, x, y, color = C.teal, size = 10) {
  rect(slide, ctx, x, y, size, size, color, { geometry: "ellipse" });
}
