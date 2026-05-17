import { C, bg, body, box, chip, dot, footer, kicker, labelBox, rect, rule, text, title } from "./common.mjs";

function arrow(slide, ctx, x1, y, w, color = C.muted) {
  rule(slide, ctx, x1, y, w - 15, color, 2);
  text(slide, ctx, ">", x1 + w - 16, y - 11, 18, 24, { size: 20, bold: true, color });
}

function stage(slide, ctx, label, sub, x, y, w, fill, accent) {
  box(slide, ctx, x, y, w, 80, fill, C.line);
  rect(slide, ctx, x, y, w, 6, accent);
  text(slide, ctx, label, x + 12, y + 16, w - 24, 22, { size: 16, bold: true, color: C.ink });
  text(slide, ctx, sub, x + 12, y + 42, w - 24, 28, { size: 11.8, color: C.muted });
}

function miniBar(slide, ctx, label, value, x, y, w, color) {
  text(slide, ctx, label, x, y, 132, 20, { size: 13, color: C.ink });
  rect(slide, ctx, x + 140, y + 5, w, 8, C.faint);
  rect(slide, ctx, x + 140, y + 5, Math.max(4, w * value), 8, color);
}

export async function renderSlide(presentation, ctx, n) {
  const slide = presentation.slides.add();
  bg(slide, ctx);

  if (n === 1) {
    rect(slide, ctx, 0, 0, 1280, 720, C.deep);
    rect(slide, ctx, 56, 46, 8, 42, C.orange);
    text(slide, ctx, "METHODS & TECHNIQUES", 78, 48, 360, 28, { size: 13, bold: true, color: "#B8C2D1" });
    text(slide, ctx, "Data selection is how we design the training distribution.", 56, 118, 760, 160, {
      size: 42,
      bold: true,
      serif: true,
      color: C.white,
    });
    text(slide, ctx, "20-minute section for Data Selection for LLMs", 58, 292, 520, 32, { size: 18, color: "#CBD5E1" });
    const nodes = [
      ["Raw data", 705, 150, C.line],
      ["Utility score", 905, 150, C.teal],
      ["Selection action", 905, 330, C.orange],
      ["Training set", 705, 330, C.blue],
    ];
    for (const [label, x, y, color] of nodes) {
      rect(slide, ctx, x, y, 160, 84, "#FFFFFF12", { line: ctx.line(color, 2) });
      text(slide, ctx, label, x + 18, y + 28, 124, 28, { size: 18, bold: true, color: C.white, align: "center" });
    }
    arrow(slide, ctx, 867, 192, 38, C.line);
    text(slide, ctx, "score", 884, 218, 60, 18, { size: 11, color: "#CBD5E1" });
    rule(slide, ctx, 985, 236, 2, 92, C.line, 2);
    text(slide, ctx, "act", 996, 264, 40, 18, { size: 11, color: "#CBD5E1" });
    arrow(slide, ctx, 867, 372, 38, C.line);
    text(slide, ctx, "sample", 860, 398, 74, 18, { size: 11, color: "#CBD5E1" });
    rule(slide, ctx, 783, 236, 2, 92, C.line, 2);
    text(slide, ctx, "train", 724, 264, 52, 18, { size: 11, color: "#CBD5E1" });
    text(slide, ctx, "Key message: most methods differ less by name than by what they score and how they use that score.", 56, 584, 940, 42, {
      size: 21,
      color: "#E5E7EB",
    });
    text(slide, ctx, "01", 1180, 676, 44, 26, { size: 14, color: "#94A3B8", align: "right", face: "Aptos Mono" });
    return slide;
  }

  if (n === 2) {
    kicker(slide, ctx, "Unified method grammar", C.teal);
    title(slide, ctx, "Every data selection method has a score and an action.");
    text(slide, ctx, "Candidate data point", 90, 225, 190, 22, { size: 16, bold: true });
    rect(slide, ctx, 76, 265, 220, 96, C.chalk, { line: ctx.line(C.line, 1) });
    text(slide, ctx, "x(i)\ntext, code, prompt,\nresponse pair, demo", 96, 286, 180, 52, { size: 17, color: C.ink, face: "Aptos Mono", align: "center" });
    arrow(slide, ctx, 320, 313, 105, C.muted);
    labelBox(slide, ctx, "Utility function", "Maps a data point to a usefulness score: language probability, quality score, domain similarity, toxicity, duplication.", 450, 235, 320, 150, C.chalk, C.teal);
    arrow(slide, ctx, 792, 313, 105, C.muted);
    labelBox(slide, ctx, "Selection mechanism", "Turns utility into an action: keep, remove, clean, oversample, downweight, reorder, or ask for annotation.", 922, 235, 300, 150, C.chalk, C.orange);
    rect(slide, ctx, 290, 470, 700, 72, "#EAF3F1", { line: ctx.line("#BCD7D3", 1) });
    text(slide, ctx, "A method is useful only relative to an objective: performance, data efficiency, evaluation integrity, safety, or selection cost.", 320, 492, 640, 32, { size: 18, color: C.ink, align: "center" });
    footer(slide, ctx, n);
    return slide;
  }

  if (n === 3) {
    kicker(slide, ctx, "Taxonomy", C.blue);
    title(slide, ctx, "The taxonomy explains what each method changes.");
    const cols = [260, 485, 710, 935];
    const rows = [215, 305, 395, 485];
    text(slide, ctx, "Axis", 86, 176, 130, 22, { size: 13, bold: true, color: C.muted });
    ["Goal", "Action target", "Output space", "Training stage"].forEach((h, i) => text(slide, ctx, h, cols[i], 176, 170, 22, { size: 13, bold: true, color: C.muted }));
    const data = [
      ["Distribution matching", "Match target domain, language, quality", "Dataset or data point", "Keep / remove"],
      ["Distribution diversification", "Reduce redundancy, expand coverage", "Mostly dataset", "Sample / weight"],
      ["Cleaning", "Remove bad spans inside examples", "Data point", "Rewrite / mask"],
      ["Mixing", "Control source proportions", "Dataset distribution", "Natural-number sampling"],
    ];
    data.forEach((r, i) => {
      const y = rows[i];
      rect(slide, ctx, 64, y - 12, 1120, 72, i % 2 ? "#FBF6EE" : C.chalk, { line: ctx.line(C.line, 1) });
      text(slide, ctx, r[0], 86, y + 6, 170, 30, { size: 16, bold: true, color: i === 1 ? C.blue : C.ink });
      text(slide, ctx, r[1], cols[0], y + 3, 190, 36, { size: 14, color: C.muted });
      text(slide, ctx, r[2], cols[1], y + 3, 170, 36, { size: 14, color: C.muted });
      text(slide, ctx, r[3], cols[2], y + 3, 170, 36, { size: 14, color: C.muted });
      text(slide, ctx, ["Pretraining", "Pretraining / tuning", "Pretraining", "Pretraining"][i], cols[3], y + 3, 170, 36, { size: 14, color: C.muted });
    });
    body(slide, ctx, "Speaker cue: use this taxonomy to keep later methods from feeling like a list of unrelated tricks.", 76, 595, 860, 32, { size: 17, color: C.ink });
    footer(slide, ctx, n);
    return slide;
  }

  if (n === 4) {
    kicker(slide, ctx, "Pretraining pipeline", C.orange);
    title(slide, ctx, "At web scale, filtering starts broad and becomes more selective.");
    const xs = [60, 220, 380, 540, 700, 860, 1020];
    const labels = [
      ["Language", "fastText, cld3,\nlangdetect"],
      ["Heuristics", "length, repetition,\nboilerplate"],
      ["Quality", "classifier or\nperplexity score"],
      ["Domain", "target similarity\nvia LM ratios"],
      ["Dedup", "URL, hash,\nMinHash, semantic"],
      ["Safety", "toxicity, NSFW,\nPII filters"],
      ["Mixing", "domain weights\nand sampling"],
    ];
    labels.forEach((d, i) => {
      stage(slide, ctx, d[0], d[1], xs[i], 260 + (i % 2) * 26, 128, i % 2 ? "#FBF6EE" : C.chalk, [C.teal, C.orange, C.blue, C.gold, C.red, C.teal, C.orange][i]);
      if (i < labels.length - 1) arrow(slide, ctx, xs[i] + 132, 304 + (i % 2) * 26, 64, C.muted);
    });
    rect(slide, ctx, 130, 500, 1020, 54, "#EBF1FA", { line: ctx.line("#C6D4EA", 1) });
    text(slide, ctx, "Ordering matters: run cheap high-recall filters first so expensive model-based selection sees fewer candidates.", 168, 517, 944, 20, { size: 18, bold: true, color: C.ink, align: "center" });
    footer(slide, ctx, n);
    return slide;
  }

  if (n === 5) {
    kicker(slide, ctx, "Heuristic filters", C.gold);
    title(slide, ctx, "Heuristics are fast assumptions about the target distribution.");
    const rows = [
      ["Item count", "characters, words, lines, sentences", "drop too-short web pages"],
      ["Repetition", "repeated chars, n-grams, sentences", "remove boilerplate loops"],
      ["Existence", "blacklisted terms, missing punctuation", "remove template fragments"],
      ["Ratios", "alpha, symbol, numeric, uppercase share", "drop non-language noise"],
      ["Statistics", "mean line length, variance", "flag malformed code or pages"],
    ];
    text(slide, ctx, "Signal family", 84, 178, 170, 20, { size: 13, bold: true, color: C.muted });
    text(slide, ctx, "Utility signal", 330, 178, 220, 20, { size: 13, bold: true, color: C.muted });
    text(slide, ctx, "Example action", 725, 178, 220, 20, { size: 13, bold: true, color: C.muted });
    rows.forEach((r, i) => {
      const y = 216 + i * 70;
      rect(slide, ctx, 64, y, 1110, 54, i % 2 ? "#FBF6EE" : C.chalk, { line: ctx.line(C.line, 1) });
      dot(slide, ctx, 86, y + 21, [C.teal, C.orange, C.blue, C.red, C.gold][i], 11);
      text(slide, ctx, r[0], 108, y + 14, 180, 20, { size: 16, bold: true });
      text(slide, ctx, r[1], 330, y + 14, 300, 20, { size: 15, color: C.muted });
      text(slide, ctx, r[2], 725, y + 14, 330, 20, { size: 15, color: C.muted });
    });
    body(slide, ctx, "Risk: high throughput comes with false positives, especially in legal, medical, minority-language, or identity-related text.", 80, 594, 980, 34, { size: 18, color: C.ink });
    footer(slide, ctx, n);
    return slide;
  }

  if (n === 6) {
    kicker(slide, ctx, "Quality and domain matching", C.teal);
    title(slide, ctx, "Target-aware filters estimate which data is closest to the desired distribution.");
    labelBox(slide, ctx, "Classifier quality filter", "Train a lightweight classifier: reference corpora are positives, raw web data is negative. Score candidates by probability of reference-like quality.", 70, 220, 340, 172, C.chalk, C.teal);
    labelBox(slide, ctx, "Perplexity filter", "Train an n-gram or small LM on reference data. Low perplexity suggests the candidate resembles the reference distribution.", 470, 220, 340, 172, C.chalk, C.blue);
    labelBox(slide, ctx, "Domain selection", "Compare in-domain likelihood against general-domain likelihood. Moore-Lewis style methods keep high ratio examples.", 870, 220, 340, 172, C.chalk, C.orange);
    rect(slide, ctx, 236, 468, 808, 70, "#F1E6D7", { line: ctx.line("#D9BE9F", 1) });
    text(slide, ctx, "utility(x) ~= log P(x | target) - log P(x | general)", 272, 488, 736, 28, { size: 22, color: C.ink, face: "Aptos Mono", align: "center" });
    text(slide, ctx, "Practical point: most systems use cheap proxies such as n-grams or hashed features before trying expensive LMs.", 190, 570, 900, 28, { size: 17, color: C.muted, align: "center" });
    footer(slide, ctx, n);
    return slide;
  }

  if (n === 7) {
    kicker(slide, ctx, "Deduplication", C.red);
    title(slide, ctx, "Deduplication buys coverage, efficiency, and lower memorization risk.");
    const ladder = [
      ["URL / metadata", "cheapest", "duplicate crawled pages", C.teal],
      ["Exact hash", "cheap", "identical documents or spans", C.blue],
      ["MinHash / LSH", "moderate", "near-duplicate documents", C.orange],
      ["Substring search", "moderate-high", "long copied spans", C.gold],
      ["Model-based semantic", "expensive", "meaning-level redundancy", C.red],
    ];
    ladder.forEach((r, i) => {
      const y = 180 + i * 82;
      rect(slide, ctx, 116 + i * 18, y, 840 - i * 18, 54, C.chalk, { line: ctx.line(C.line, 1) });
      rect(slide, ctx, 116 + i * 18, y, 8, 54, r[3]);
      text(slide, ctx, r[0], 142 + i * 18, y + 10, 230, 20, { size: 17, bold: true });
      chip(slide, ctx, r[1], 400 + i * 18, y + 12, 115, "#F4ECE0", C.ink);
      text(slide, ctx, r[2], 555 + i * 18, y + 13, 300, 20, { size: 14, color: C.muted });
    });
    rect(slide, ctx, 985, 220, 170, 240, "#1F2937", { line: ctx.line("#1F2937", 1) });
    text(slide, ctx, "Repeated samples\nincrease memorization\nwith scale.", 1008, 252, 124, 92, { size: 20, bold: true, color: C.white, align: "center" });
    rule(slide, ctx, 1016, 372, 108, "#94A3B8", 1);
    text(slide, ctx, "But not all memory is bad: facts, APIs, and syntax may need repetition.", 1006, 394, 128, 54, { size: 12.5, color: "#CBD5E1", align: "center" });
    footer(slide, ctx, n);
    return slide;
  }

  if (n === 8) {
    kicker(slide, ctx, "Safety and multilingual gates", C.orange);
    title(slide, ctx, "Safety filters and multilingual filters are necessary but distribution-shifting.");
    labelBox(slide, ctx, "Toxic / explicit content", "URL blocklists, lexicons, classifiers, harmful-content perplexity models.", 76, 220, 335, 140, C.chalk, C.red);
    labelBox(slide, ctx, "PII and secrets", "Regex or classifiers detect emails, phones, IPs, keys; common action is obfuscation.", 76, 390, 335, 140, C.chalk, C.orange);
    labelBox(slide, ctx, "Language-specific rules", "Length thresholds, script filters, language-ID confidence, and code/text splits need per-language tuning.", 478, 220, 335, 140, C.chalk, C.teal);
    labelBox(slide, ctx, "Low-resource languages", "Classifier noise rises as data gets rarer; manual inspection and native speakers become important.", 478, 390, 335, 140, C.chalk, C.blue);
    rect(slide, ctx, 885, 236, 270, 260, "#F7E6E2", { line: ctx.line("#D9AEA5", 1) });
    text(slide, ctx, "No free lunch", 920, 268, 200, 28, { size: 25, bold: true, serif: true, color: C.red, align: "center" });
    text(slide, ctx, "Filtering can reduce harmful generations, but it can also remove useful detection examples, minority dialects, or legitimate domain text.", 925, 320, 190, 118, { size: 17, color: C.ink, align: "center" });
    footer(slide, ctx, n);
    return slide;
  }

  if (n === 9) {
    kicker(slide, ctx, "Data mixing", C.blue);
    title(slide, ctx, "Data mixing turns source domains into sampling weights.");
    const x = 92;
    [["Web", .48, C.teal], ["Books", .18, C.orange], ["Code", .16, C.blue], ["Wikipedia", .10, C.gold], ["Domain data", .08, C.red]].forEach((r, i) => miniBar(slide, ctx, r[0], r[1], x, 225 + i * 45, 360, r[2]));
    text(slide, ctx, "alpha = domain weights", 96, 480, 360, 24, { size: 20, bold: true, face: "Aptos Mono", color: C.ink });
    text(slide, ctx, "Sample a domain first, then sample examples inside that domain.", 96, 512, 390, 40, { size: 16, color: C.muted });
    const methods = [
      ["Manual", "human priors: upweight books, code, or curated sets"],
      ["Empirical", "train small models and tune weights by downstream scores"],
      ["Principled offline", "DoReMi / DoGE update weights using proxy losses"],
      ["Online", "Skill-it / ODM adjust sampling during training"],
    ];
    methods.forEach((m, i) => {
      const y = 205 + i * 82;
      labelBox(slide, ctx, m[0], m[1], 610, y, 500, 60, i % 2 ? "#FBF6EE" : C.chalk, [C.teal, C.orange, C.blue, C.red][i]);
    });
    text(slide, ctx, "Core tradeoff: upweighting one domain downweights everything else.", 620, 548, 480, 24, { size: 18, bold: true, color: C.ink });
    footer(slide, ctx, n);
    return slide;
  }

  if (n === 10) {
    kicker(slide, ctx, "Instruction tuning", C.teal);
    title(slide, ctx, "Instruction-tuning selection shifts from cleaning web text to diversifying tasks.");
    const y = 292;
    stage(slide, ctx, "Task pools", "FLAN, P3, Natural Instructions", 84, y, 190, C.chalk, C.teal);
    arrow(slide, ctx, 286, y + 40, 92, C.muted);
    stage(slide, ctx, "Format unification", "instruction -> output", 390, y, 190, C.chalk, C.blue);
    arrow(slide, ctx, 592, y + 40, 92, C.muted);
    stage(slide, ctx, "Synthetic expansion", "Self-Instruct, Evol-Instruct", 696, y, 190, C.chalk, C.orange);
    arrow(slide, ctx, 898, y + 40, 92, C.muted);
    stage(slide, ctx, "Quality subset", "human or model judged", 1002, y, 190, C.chalk, C.red);
    rect(slide, ctx, 166, 492, 948, 54, "#EAF3F1", { line: ctx.line("#BCD7D3", 1) });
    text(slide, ctx, "The method objective becomes coverage over user intents, response styles, difficulty, and license/provenance constraints.", 196, 510, 888, 20, { size: 18, color: C.ink, align: "center" });
    footer(slide, ctx, n);
    return slide;
  }

  if (n === 11) {
    kicker(slide, ctx, "Preference fine-tuning", C.red);
    title(slide, ctx, "Alignment data selection is reward-guided, but the target is less explicit.");
    rect(slide, ctx, 80, 212, 290, 260, C.chalk, { line: ctx.line(C.line, 1) });
    text(slide, ctx, "Preference sample", 110, 238, 220, 22, { size: 20, bold: true });
    text(slide, ctx, "Prompt\nChosen response\nRejected response", 112, 292, 210, 92, { size: 21, face: "Aptos Mono", color: C.ink });
    arrow(slide, ctx, 396, 338, 110, C.muted);
    labelBox(slide, ctx, "Model-based evaluation", "GPT-4, critic models, or reward models score candidate responses for quality, helpfulness, or harm.", 530, 210, 300, 122, C.chalk, C.blue);
    labelBox(slide, ctx, "Reward re-weighting", "Best-of-N / rejection sampling selects high-reward outputs or weights examples during training.", 530, 372, 300, 122, C.chalk, C.orange);
    arrow(slide, ctx, 848, 338, 90, C.muted);
    rect(slide, ctx, 960, 254, 210, 190, "#1F2937", { line: ctx.line("#1F2937", 1) });
    text(slide, ctx, "Open issue", 990, 285, 150, 28, { size: 24, bold: true, serif: true, color: C.white, align: "center" });
    text(slide, ctx, "Helpful, harmless, and honest are qualitative targets, so utility functions inherit human/model value judgments.", 990, 334, 150, 78, { size: 14.5, color: "#CBD5E1", align: "center" });
    footer(slide, ctx, n);
    return slide;
  }

  if (n === 12) {
    kicker(slide, ctx, "Practical recipe", C.gold);
    title(slide, ctx, "Choose the cheapest utility that matches the objective.");
    const items = [
      ["1", "Define the target distribution", "General LLM, code model, domain model, chat assistant, or task-specific system."],
      ["2", "Pick the selection objective", "Performance, data efficiency, evaluation integrity, safety, or selection cost."],
      ["3", "Start with cheap filters", "Language, length, obvious noise, URL/hash deduplication."],
      ["4", "Add target-aware scoring", "Quality, domain, reward, gradient, or retriever-based utilities."],
      ["5", "Audit tradeoffs", "False positives, benchmark leakage, memorization, bias, multilingual loss."],
    ];
    items.forEach((it, i) => {
      const y = 172 + i * 86;
      rect(slide, ctx, 86, y, 78, 58, i % 2 ? C.blue : C.teal);
      text(slide, ctx, it[0], 105, y + 12, 40, 28, { size: 28, bold: true, color: C.white, align: "center" });
      text(slide, ctx, it[1], 192, y + 5, 320, 24, { size: 19, bold: true, color: C.ink });
      text(slide, ctx, it[2], 192, y + 34, 760, 24, { size: 15.5, color: C.muted });
    });
    rect(slide, ctx, 980, 206, 180, 250, "#F1E6D7", { line: ctx.line("#D9BE9F", 1) });
    text(slide, ctx, "Takeaway", 1008, 238, 122, 28, { size: 25, bold: true, serif: true, color: C.ink, align: "center" });
    text(slide, ctx, "Data selection is not cleanup afterthought. It is training-distribution design.", 1004, 298, 132, 82, { size: 18, bold: true, color: C.ink, align: "center" });
    footer(slide, ctx, n);
    return slide;
  }

  footer(slide, ctx, n);
  return slide;
}
