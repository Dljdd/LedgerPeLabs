import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = process.cwd();
const OUTPUT_DIR = process.env.APAR_DECK_OUTPUT_DIR;
if (!OUTPUT_DIR) throw new Error("APAR_DECK_OUTPUT_DIR is required");

const C = {
  paper: "#F8F7F3",
  white: "#FFFFFF",
  ink: "#171714",
  muted: "#686863",
  line: "#D7D6CF",
  soft: "#EEECE5",
  orange: "#E9652E",
  orangeSoft: "#FCE5DA",
  green: "#176A4B",
  greenSoft: "#DFF1E7",
  blue: "#255D73",
  blueSoft: "#DDEBF0",
  red: "#A43C2E",
};

const slides = Presentation.create({ slideSize: { width: 1280, height: 720 } });

function box(slide, x, y, w, h, fill = C.white, line = C.line, radius = "rounded-xl") {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width: 1 },
    borderRadius: radius,
  });
}

function rule(slide, x, y, w, color = C.line, height = 2) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: w, height },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
}

function txt(slide, text, x, y, w, h, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: opts.size ?? 24,
    typeface: opts.typeface ?? "Arial",
    color: opts.color ?? C.ink,
    bold: opts.bold ?? false,
    alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "top",
    autoFit: opts.autoFit ?? "shrinkText",
  };
  return shape;
}

function header(slide, index, title, eyebrow = "ADAPTIVE PAYMENT ASSURANCE RANGE") {
  slide.background.fill = C.paper;
  txt(slide, eyebrow, 42, 28, 780, 26, { size: 13, bold: true, color: C.orange });
  txt(slide, title, 42, 60, 1110, 70, { size: 38, bold: true });
  txt(slide, String(index).padStart(2, "0"), 1184, 36, 54, 22, {
    size: 13,
    color: C.muted,
    align: "right",
  });
  rule(slide, 42, 133, 1196, C.line, 1);
}

function note(slide, body, sources) {
  slide.speakerNotes.textFrame.setText(
    `${body}\n\n[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}\n[/Sources]`,
  );
  slide.speakerNotes.setVisible(true);
}

function imageFrame(slide, source, alt, x, y, w, h, fit = "cover") {
  box(slide, x - 8, y - 8, w + 16, h + 16, C.white, C.line);
  return slide.images.add({ blob: source, contentType: "image/png", alt, fit, position: { left: x, top: y, width: w, height: h }, geometry: "roundRect", borderRadius: "rounded-lg" });
}

function pill(slide, label, x, y, w, fill = C.orangeSoft, color = C.orange) {
  box(slide, x, y, w, 32, fill, fill, "rounded-full");
  txt(slide, label, x + 10, y + 6, w - 20, 20, { size: 13, bold: true, color, align: "center" });
}

const overview = path.join(ROOT, "docs/demo/screenshots/apar-console-overview-desktop.png");
const replay = path.join(ROOT, "docs/demo/screenshots/apar-console-replay-desktop.png");
const investigation = path.join(ROOT, "docs/demo/screenshots/apar-console-investigation-desktop.png");
const assurance = path.join(ROOT, "docs/demo/screenshots/apar-console-assurance-mobile.png");
const overviewBytes = await fs.readFile(overview);
const replayBytes = await fs.readFile(replay);
const investigationBytes = await fs.readFile(investigation);
const assuranceBytes = await fs.readFile(assurance);

// 1 — cover, adapted from Codex Grid slide 08.
{
  const s = slides.slides.add();
  s.background.fill = C.paper;
  txt(s, "MASTERCARD INNOVATION CHALLENGE 2026", 42, 36, 560, 28, { size: 13, bold: true, color: C.orange });
  txt(s, "Assure the campaign,\nnot just the transaction.", 42, 112, 570, 190, { size: 48, bold: true });
  txt(s, "APAR turns emerging GenAI payment threats into rail-correct synthetic campaigns, tests layered defenses, and preserves a human promotion boundary.", 42, 330, 540, 126, { size: 23, color: C.muted });
  pill(s, "SYNTHETIC • OFFLINE • REPLAYABLE", 42, 494, 288);
  txt(s, "Dylan Moraes  •  Competition submission", 42, 624, 480, 30, { size: 16, color: C.muted });
  imageFrame(s, overviewBytes, "APAR console overview", 664, 48, 574, 594, "cover");
  note(s, "Open with the problem shift: AI changes attacker capability, so the unit of assurance becomes the campaign.", ["docs/demo/screenshots/apar-console-overview-desktop.png", "SOLUTION_SPEC.md"]);
}

// 2 — problem shift.
{
  const s = slides.slides.add();
  header(s, 2, "GenAI changes the operating model of fraud");
  const stages = [
    ["PERSONALIZE", "Victim-specific\ncontent at scale"],
    ["ITERATE", "Faster adaptation\nto feedback"],
    ["COORDINATE", "Many entities, rails,\nand time windows"],
    ["DELEGATE", "Agents can act\non user intent"],
  ];
  stages.forEach(([a, b], i) => {
    const x = 42 + i * 298;
    box(s, x, 182, 260, 210, i === 2 ? C.orangeSoft : C.white, i === 2 ? C.orange : C.line);
    txt(s, `0${i + 1}`, x + 20, 202, 42, 28, { size: 15, bold: true, color: C.orange });
    txt(s, a, x + 20, 250, 220, 34, { size: 22, bold: true });
    txt(s, b, x + 20, 304, 220, 72, { size: 19, color: C.muted });
    if (i < 3) txt(s, "→", x + 264, 268, 32, 44, { size: 28, bold: true, color: C.orange, align: "center" });
  });
  box(s, 42, 448, 1196, 138, C.ink, C.ink);
  txt(s, "Transaction classifier", 68, 476, 260, 32, { size: 19, color: "#BDBDB7" });
  txt(s, "→", 344, 472, 54, 36, { size: 26, color: C.orange, bold: true, align: "center" });
  txt(s, "Campaign-aware assurance system", 418, 468, 470, 68, { size: 28, color: C.white, bold: true });
  txt(s, "Behavior + lifecycle + economics + authority + governance", 418, 546, 650, 26, { size: 16, color: "#DADAD4" });
  note(s, "Most entries stop at a fraud classifier. APAR models the operating capability GenAI adds and tests the downstream control system.", ["SOLUTION_SPEC.md", "https://www.bis.org/speeches/20260608-strengthening-security-european-payments"]);
}

// 3 — architecture.
{
  const s = slides.slides.add();
  header(s, 3, "One assurance loop connects five control planes");
  const cards = [
    ["IDENTIFY", "Evidence + sources\nthreat registry", C.blueSoft, C.blue],
    ["GENERATE", "Rail + ledger-correct\ncampaigns", C.orangeSoft, C.orange],
    ["DEFEND", "Rules + calibrated\ngraph ensemble", C.greenSoft, C.green],
    ["VERIFY", "Identity + intent\nintegrity", C.blueSoft, C.blue],
    ["GOVERN", "Controls + human\npromotion", C.soft, C.ink],
  ];
  cards.forEach(([a, b, fill, color], i) => {
    const x = 42 + i * 242;
    box(s, x, 188, 214, 236, fill, color);
    txt(s, a, x + 18, 210, 178, 26, { size: 14, bold: true, color });
    txt(s, b, x + 18, 272, 178, 82, { size: 22, bold: true });
    if (i < 4) txt(s, "→", x + 212, 286, 30, 34, { size: 24, bold: true, color: C.orange, align: "center" });
  });
  rule(s, 146, 482, 988, C.orange, 3);
  [146, 388, 630, 872, 1114].forEach((x) => {
    s.shapes.add({ geometry: "ellipse", position: { left: x - 7, top: 475, width: 17, height: 17 }, fill: C.orange, line: { style: "solid", fill: C.orange, width: 0 } });
  });
  txt(s, "Bounded feedback returns failed controls to the research backlog—never directly to deployment.", 180, 525, 920, 58, { size: 22, color: C.muted, align: "center" });
  note(s, "Walk left to right. Emphasize that verification and promotion are separate from probabilistic scoring.", ["docs/02-system-architecture.md", "docs/08-security-and-governance.md"]);
}

// 4 — experiment journey.
{
  const s = slides.slides.add();
  header(s, 4, "Failed experiments shaped the winning architecture");
  const points = [
    ["V1", "Workload cap failed", "1.7857% review floor\nvs 1% cap"],
    ["V4", "Missing metrics failed closed", "Calibration + alert time\nwere not invented"],
    ["V5a", "Perfect score rejected", "Future graph leakage\nremoved and recorded"],
    ["V5b", "Real rail evidence", "Generator → simulator →\nrail → ledger → row"],
    ["V5c", "Graph arm selected", "Best frontier;\nfull hybrid rejected"],
  ];
  rule(s, 88, 316, 1100, C.line, 3);
  points.forEach(([v, title, body], i) => {
    const x = 58 + i * 236;
    box(s, x, i % 2 ? 348 : 166, 210, 144, i === 4 ? C.greenSoft : C.white, i === 4 ? C.green : C.line);
    txt(s, v, x + 16, (i % 2 ? 364 : 182), 50, 24, { size: 14, bold: true, color: i === 4 ? C.green : C.orange });
    txt(s, title, x + 16, (i % 2 ? 397 : 215), 178, 38, { size: 18, bold: true });
    txt(s, body, x + 16, (i % 2 ? 440 : 258), 178, 52, { size: 15, color: C.muted });
    s.shapes.add({ geometry: "ellipse", position: { left: x + 97, top: 307, width: 18, height: 18 }, fill: i === 4 ? C.green : C.orange, line: { style: "solid", fill: C.paper, width: 3 } });
  });
  txt(s, "Novelty: not a single algorithm—an evidence-governed loop that learns from its own failure modes.", 94, 566, 1080, 42, { size: 22, bold: true, align: "center" });
  note(s, "Tell the story of intellectual honesty: the architecture improved because infeasible, incomplete, and leaky results were rejected.", ["docs/submission/RESEARCH_AND_EXPERIMENT_JOURNEY.md", "docs/experiments/defense-v5-rejected-7410a64-record.json", "docs/experiments/defense-v5-rejected-7f3b78d-record.json"]);
}

// 5 — quantitative evidence, adapted from Codex Grid slide 14/20.
{
  const s = slides.slides.add();
  header(s, 5, "Graph context wins the operational frontier");
  txt(s, "Verified recovered diagnostics — synthetic and non-authoritative", 42, 142, 760, 26, { size: 15, bold: true, color: C.orange });
  const rows = [
    ["Rules only", "85.95", "14.88", "25.37", "87.44", "4.06"],
    ["Ensemble • no graph", "99.74", "94.76", "97.19", "0.000", "3.38"],
    ["Ensemble • graph", "99.87", "95.88", "97.83", "0.0037", "3.54"],
    ["Full hybrid", "99.93", "16.85", "28.83", "87.44", "19.74"],
  ];
  const x0 = 42, y0 = 200;
  const widths = [300, 150, 150, 150, 190, 150];
  const headers = ["ARM", "RECALL %", "PRECISION %", "F1 %", "FALSE-DECLINE %", "P95 MS"];
  let x = x0;
  headers.forEach((h, i) => { box(s, x, y0, widths[i], 48, C.ink, C.ink, "rounded-none"); txt(s, h, x + 10, y0 + 14, widths[i] - 20, 20, { size: 13, bold: true, color: C.white }); x += widths[i]; });
  rows.forEach((row, r) => {
    let rx = x0; const fill = r === 2 ? C.greenSoft : (r % 2 ? C.white : C.soft);
    row.forEach((v, i) => { box(s, rx, y0 + 48 + r * 66, widths[i], 66, fill, r === 2 ? C.green : C.line, "rounded-none"); txt(s, v, rx + 10, y0 + 69 + r * 66, widths[i] - 20, 25, { size: i === 0 ? 17 : 16, bold: r === 2 || i === 0, color: r === 3 && i > 0 ? C.red : C.ink }); rx += widths[i]; });
  });
  box(s, 42, 548, 1196, 76, C.orangeSoft, C.orange);
  txt(s, "Precision  +1.12 pts", 70, 570, 260, 28, { size: 20, bold: true, color: C.orange });
  txt(s, "Challenge  −0.23 pts", 380, 570, 280, 28, { size: 20, bold: true, color: C.orange });
  txt(s, "Latency  +0.16 ms p95", 716, 570, 390, 28, { size: 20, bold: true, color: C.orange });
  note(s, "The graph arm improves the no-graph frontier at a small latency cost. The full hybrid proves why recall alone is unsafe.", ["evidence/sentinel-v5-recovered-metrics/verified-report.json", "docs/submission/EVALUATION_AND_LIMITATIONS.md"]);
}

// 6 — portable model.
{
  const s = slides.slides.add();
  header(s, 6, "A small, portable model powers the live demo");
  imageFrame(s, replayBytes, "APAR live model replay", 42, 168, 746, 444, "cover");
  const stats = [
    ["3 ×", "CatBoost members"],
    ["46", "frozen causal features"],
    ["12", "hash-bound scenarios"],
    ["0.0", "max replay error"],
  ];
  stats.forEach(([n, l], i) => {
    const y = 168 + i * 104;
    box(s, 830, y, 408, 86, i === 3 ? C.greenSoft : C.white, i === 3 ? C.green : C.line);
    txt(s, n, 850, y + 17, 100, 42, { size: 31, bold: true, color: i === 3 ? C.green : C.orange });
    txt(s, l, 958, y + 23, 250, 32, { size: 18, bold: true });
  });
  txt(s, "Approve • Challenge • Review hold • Decline hold", 830, 592, 408, 28, { size: 15, bold: true, color: C.muted, align: "center" });
  note(s, "This is the accepted Stage 30 ensemble_with_graph bundle. The scorer is real; the scenarios are curated replay checks.", ["demo/sentinel-v5/manifest.json", "demo/sentinel-v5/spec.json", "docs/demo/screenshots/apar-console-replay-desktop.png"]);
}

// 7 — investigation.
{
  const s = slides.slides.add();
  header(s, 7, "Investigators see one campaign—not twelve alerts");
  imageFrame(s, investigationBytes, "APAR campaign investigation", 394, 160, 844, 478, "cover");
  pill(s, "CASE-CENTRIC", 42, 174, 150, C.greenSoft, C.green);
  txt(s, "14", 42, 234, 110, 50, { size: 42, bold: true, color: C.orange });
  txt(s, "entities", 42, 284, 190, 26, { size: 17, color: C.muted });
  txt(s, "10", 42, 342, 110, 50, { size: 42, bold: true, color: C.orange });
  txt(s, "payment edges", 42, 392, 200, 26, { size: 17, color: C.muted });
  txt(s, "$500", 42, 452, 160, 50, { size: 42, bold: true, color: C.orange });
  txt(s, "conserved synthetic ledger", 42, 502, 270, 52, { size: 17, color: C.muted });
  box(s, 42, 574, 304, 64, C.soft, C.line);
  txt(s, "Analyst-time benefit: evidence pending", 58, 594, 272, 24, { size: 15, bold: true });
  note(s, "Point to the linked mule node and genuine deterministic payment edges. Do not invent an analyst productivity number.", ["docs/demo/screenshots/apar-console-investigation-desktop.png", "web/public/data/console-evidence.json"]);
}

// 8 — agentic trust.
{
  const s = slides.slides.add();
  header(s, 8, "Agentic commerce needs proof before probability");
  const assuranceView = imageFrame(s, assuranceBytes, "APAR TrustVerifier assurance view", 932, 162, 270, 480, "cover");
  assuranceView.crop = { left: 0, top: 0.465, right: 0, bottom: 0.30636 };
  const checks = ["Agent identity", "User mandate", "Scope + amount", "Merchant + cart", "Expiry + nonce", "Replay rejection"];
  checks.forEach((label, i) => {
    const col = i < 3 ? 0 : 1, row = i % 3;
    const x = 42 + col * 410, y = 178 + row * 120;
    box(s, x, y, 374, 88, C.white, C.line);
    s.shapes.add({ geometry: "ellipse", position: { left: x + 18, top: y + 25, width: 34, height: 34 }, fill: C.greenSoft, line: { style: "solid", fill: C.green, width: 1 } });
    txt(s, "✓", x + 25, y + 29, 20, 22, { size: 16, bold: true, color: C.green, align: "center" });
    txt(s, label, x + 68, y + 27, 280, 30, { size: 19, bold: true });
  });
  box(s, 42, 554, 784, 88, C.ink, C.ink);
  txt(s, "DEFINITIVE FAILURE", 64, 574, 220, 24, { size: 13, bold: true, color: C.orange });
  txt(s, "DECLINE_HOLD before statistical risk", 294, 570, 500, 36, { size: 24, bold: true, color: C.white });
  note(s, "TrustVerifier is a separate authorization-integrity claim—not graph model performance. A probabilistic score cannot repair invalid authority.", ["src/apar/trust/verifier.py", "tests/trust/test_verifier.py", "https://www.mastercard.com/us/en/news-and-trends/stories/2026/mastercard-agentic-commerce-vision.html", "https://www.nccoe.nist.gov/sites/default/files/2026-02/accelerating-the-adoption-of-software-and-ai-agent-identity-and-authorization-concept-paper.pdf"]);
}

// 9 — evidence governance.
{
  const s = slides.slides.add();
  header(s, 9, "Evidence governance is part of the product");
  const layers = [
    ["01", "Causal features", "Future append + equal-time isolation"],
    ["02", "Executed controls", "Shuffle, rename, leakage, single-class"],
    ["03", "Immutable evidence", "Models, traces, metrics, manifests hashed"],
    ["04", "Independent replay", "Fresh environment re-scores all 12 cases"],
    ["05", "Human promotion", "The model cannot approve itself"],
  ];
  layers.forEach(([n, a, b], i) => {
    const x = 42 + i * 238, y = 178 + i * 28;
    box(s, x, y, 214, 276 - i * 10, i === 4 ? C.ink : C.white, i === 4 ? C.ink : C.line);
    txt(s, n, x + 18, y + 20, 44, 24, { size: 14, bold: true, color: C.orange });
    txt(s, a, x + 18, y + 70, 178, 56, { size: 22, bold: true, color: i === 4 ? C.white : C.ink });
    txt(s, b, x + 18, y + 148, 178, 76, { size: 16, color: i === 4 ? "#D8D8D2" : C.muted });
  });
  txt(s, "Portable bundle", 68, 536, 160, 22, { size: 14, bold: true, color: C.muted });
  txt(s, "52ed4c31…dbc900", 68, 568, 310, 28, { size: 18, bold: true, typeface: "Menlo" });
  txt(s, "Promotion boundary", 486, 536, 160, 22, { size: 14, bold: true, color: C.muted });
  txt(s, "Human approval required", 486, 568, 280, 28, { size: 18, bold: true });
  pill(s, "NO SELF-PROMOTION", 912, 550, 250);
  note(s, "Show that every evidence step stays auditable and that promotion remains a human decision; recovered metrics remain diagnostics.", ["demo/sentinel-v5/manifest.json", "evidence/sentinel-v5-recovered-metrics/verified-report.json", "docs/submission/RELEASE_CHECKLIST.md"]);
}

// 10 — claims and limitations.
{
  const s = slides.slides.add();
  header(s, 10, "A sharp claim boundary makes the demo credible");
  box(s, 42, 170, 568, 438, C.greenSoft, C.green);
  box(s, 632, 170, 606, 438, C.orangeSoft, C.orange);
  txt(s, "WHAT WE PROVE", 70, 196, 300, 32, { size: 17, bold: true, color: C.green });
  txt(s, "WHAT WE DO NOT CLAIM", 660, 196, 340, 32, { size: 17, bold: true, color: C.orange });
  const yes = ["Portable model loads and replays exactly", "Four rail-backed synthetic families", "Graph/no-graph architecture comparison", "Deterministic TrustVerifier integrity", "Offline console + hash-bound fallback"];
  const no = ["Production or real-cardholder performance", "Production-readiness or scale evidence", "External or multi-institution validation", "Autonomous model promotion", "Claims beyond the demonstrated model"];
  yes.forEach((t, i) => { txt(s, "✓", 72, 258 + i * 62, 28, 28, { size: 18, bold: true, color: C.green }); txt(s, t, 112, 258 + i * 62, 460, 38, { size: 18, bold: true }); });
  no.forEach((t, i) => { txt(s, "×", 662, 258 + i * 62, 28, 28, { size: 22, bold: true, color: C.red }); txt(s, t, 704, 258 + i * 62, 490, 38, { size: 18, bold: true }); });
  txt(s, "Synthetic evidence before assertion.", 42, 634, 1196, 34, { size: 24, bold: true, align: "center" });
  note(s, "This slide is the trust moment. Be explicit about what remains future work and why that does not diminish the working prototype.", ["docs/submission/EVALUATION_AND_LIMITATIONS.md", "docs/submission/MODEL_CARD.md"]);
}

// 11 — deployment / close.
{
  const s = slides.slides.add();
  header(s, 11, "From competition prototype to governed payment assurance");
  const phases = [
    ["1", "OFFLINE REPLAY", "Authorized historical samples"],
    ["2", "SHADOW MODE", "Live features; no authority"],
    ["3", "ASSISTED OPS", "Human-controlled friction"],
    ["4", "CHALLENGER", "Limited traffic + rollback"],
    ["5", "GOVERNED SCALE", "Continuous adversarial regression"],
  ];
  phases.forEach(([n, a, b], i) => {
    const x = 42 + i * 238;
    box(s, x, 184, 212, 218, i === 0 ? C.orangeSoft : C.white, i === 0 ? C.orange : C.line);
    txt(s, n, x + 18, 204, 32, 28, { size: 14, bold: true, color: C.orange });
    txt(s, a, x + 18, 258, 176, 48, { size: 18, bold: true });
    txt(s, b, x + 18, 326, 176, 58, { size: 16, color: C.muted });
    if (i < 4) txt(s, "→", x + 210, 282, 28, 32, { size: 22, bold: true, color: C.orange, align: "center" });
  });
  box(s, 42, 460, 1196, 144, C.ink, C.ink);
  txt(s, "APAR", 68, 484, 180, 52, { size: 40, bold: true, color: C.white });
  txt(s, "Assure the campaign. Verify the intent.\nPromote only the evidence.", 272, 480, 730, 82, { size: 30, bold: true, color: C.white });
  pill(s, "DEMO READY", 1020, 506, 166, C.greenSoft, C.green);
  txt(s, "Adaptive Payment Assurance Range  •  Dylan Moraes", 42, 638, 660, 26, { size: 15, color: C.muted });
  note(s, "Close on the product promise and transition into the live console walkthrough.", ["docs/submission/COMMERCIAL_AND_DEPLOYMENT_PLAN.md", "docs/submission/FIVE_MINUTE_WALKTHROUGH.md"]);
}

async function writeBlob(target, blob) {
  await fs.writeFile(target, new Uint8Array(await blob.arrayBuffer()));
}

await fs.mkdir(OUTPUT_DIR, { recursive: true });
for (const [index, slide] of slides.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  await writeBlob(path.join(OUTPUT_DIR, `${stem}.png`), await slides.export({ slide, format: "png", scale: 1 }));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(OUTPUT_DIR, `${stem}.layout.json`), await layout.text());
}
await writeBlob(path.join(OUTPUT_DIR, "deck-montage.webp"), await slides.export({ format: "webp", montage: true, scale: 1 }));
const pptx = await PresentationFile.exportPptx(slides);
await pptx.save(path.join(OUTPUT_DIR, "APAR_COMPETITION_DECK.pptx"));
