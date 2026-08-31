import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const SOURCE_PPTX = process.env.APAR_DECK_SOURCE;
const OUTPUT_DIR = process.env.APAR_DECK_OUTPUT_DIR;
const FINAL_PPTX = process.env.APAR_DECK_FINAL_PPTX;

if (!SOURCE_PPTX) throw new Error("APAR_DECK_SOURCE is required");
if (!OUTPUT_DIR) throw new Error("APAR_DECK_OUTPUT_DIR is required");
if (!FINAL_PPTX) throw new Error("APAR_DECK_FINAL_PPTX is required");

const C = {
  bg: "#0A0907",
  raised: "#100E0B",
  surface: "#14110D",
  surface2: "#19150F",
  surface3: "#211B14",
  line: "#3F382E",
  lineStrong: "#6B5F50",
  text: "#F1E9DC",
  textSoft: "#C8BDAE",
  muted: "#9C9183",
  orange: "#E8793D",
  orangeHot: "#F19A68",
  red: "#E35E54",
  redSoft: "#F28A80",
  redDark: "#2B1512",
  amber: "#D4A359",
  amberDark: "#251F12",
  green: "#75B78E",
  greenDark: "#20382A",
};

const presentation = await PresentationFile.importPptx(await FileBlob.load(SOURCE_PPTX));
const inspection = await presentation.inspect({
  kind: "slide,textbox,shape,image,notes,layout",
  include: "id,slide,name,title,text,textPreview,textChars,textLines,bbox,bboxUnit,alt,isPlaceholder,placeholders",
  maxChars: 500000,
});
const records = inspection.ndjson
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line));

const bySlide = new Map();
for (const record of records) {
  if (!Number.isInteger(record.slide) || !record.id || record.kind === "slide") continue;
  const target = presentation.resolve(record.id);
  const text = record.kind === "textbox" ? target.text.toString().trim() : "";
  if (!bySlide.has(record.slide)) bySlide.set(record.slide, []);
  bySlide.get(record.slide).push({ ...record, target, text });
}

function items(slideNumber, kind) {
  return (bySlide.get(slideNumber) ?? []).filter((item) => !kind || item.kind === kind);
}

function within(value, expected, tolerance = 3) {
  return Math.abs(Number(value) - expected) <= tolerance;
}

function at(item, x, y, w, h, tolerance = 4) {
  const box = item.bbox ?? [];
  return (
    box.length === 4 &&
    within(box[0], x, tolerance) &&
    within(box[1], y, tolerance) &&
    (w === undefined || within(box[2], w, tolerance)) &&
    (h === undefined || within(box[3], h, tolerance))
  );
}

function textItem(slideNumber, exact) {
  return items(slideNumber, "textbox").find((item) => item.text === exact);
}

function textsAt(slideNumber, predicate) {
  return items(slideNumber, "textbox").filter(predicate);
}

function shapesAt(slideNumber, predicate) {
  return items(slideNumber, "shape").filter(predicate);
}

function setPanel(item, fill = C.surface, line = C.line, width = 1) {
  if (!item) return;
  item.target.fill = fill;
  item.target.line = { style: "solid", fill: line, width };
  if (["rect", "roundRect", "textbox"].includes(String(item.target.geometry))) {
    item.target.borderRadius = 0;
  }
}

function setRule(item, position, color = C.lineStrong) {
  if (!item) return;
  setPosition(item, position);
  setPanel(item, color, color, 0);
}

function clearShape(item) {
  if (!item) return;
  setPanel(item, "none", C.bg, 0);
}

function setText(item, options = {}) {
  if (!item) return;
  item.target.text.typeface = options.typeface ?? "Arial";
  item.target.text.color = options.color ?? C.text;
  if (options.size !== undefined) item.target.text.fontSize = options.size;
  if (options.bold !== undefined) item.target.text.bold = options.bold;
  if (options.align !== undefined) item.target.text.alignment = options.align;
  if (options.valign !== undefined) item.target.text.verticalAlignment = options.valign;
  item.target.text.autoFit = options.autoFit ?? "shrinkText";
}

function setPosition(item, position) {
  if (!item) return;
  item.target.position = { ...item.target.position, ...position };
}

function rewriteText(slideNumber, previousText, nextText) {
  const item = textItem(slideNumber, previousText) ?? textItem(slideNumber, nextText);
  if (!item) throw new Error(`Unable to find text on slide ${slideNumber}: ${previousText}`);
  if (item.text === previousText && previousText !== nextText) {
    item.target.text.replace(previousText, nextText);
  }
  return item;
}

function styleHeader(slideNumber, titleSize = 39) {
  const slide = presentation.slides.items[slideNumber - 1];
  slide.background.fill = C.bg;
  for (const item of items(slideNumber, "textbox")) {
    const [x, y] = item.bbox ?? [];
    if (y <= 34 && x < 900) {
      setText(item, { typeface: "Menlo", size: 11, bold: true, color: C.orangeHot });
    } else if (y >= 54 && y <= 80 && x < 1160) {
      setText(item, { typeface: "Georgia", size: titleSize, bold: false, color: C.text });
    } else if (x >= 1160 && y < 70) {
      setText(item, { typeface: "Menlo", size: 11, bold: false, color: C.muted, align: "right" });
    } else {
      setText(item, { color: C.textSoft });
    }
  }
  for (const item of shapesAt(slideNumber, (entry) => {
    const box = entry.bbox ?? [];
    return box[1] >= 128 && box[1] <= 138 && box[3] <= 4;
  })) {
    setPanel(item, C.line, C.line, 0);
  }
}

// 01 — Opening thesis: restrained editorial cover with the real console as proof.
{
  const slide = presentation.slides.items[0];
  slide.background.fill = C.bg;
  const all = items(1);
  for (const item of all.filter((entry) => entry.kind === "textbox")) setText(item, { color: C.textSoft });
  setText(textItem(1, "MASTERCARD INNOVATION CHALLENGE 2026"), { typeface: "Menlo", size: 11, bold: true, color: C.orangeHot });
  setText(textItem(1, "Assure the campaign,\nnot just the transaction."), { typeface: "Georgia", size: 50, bold: false, color: C.text });
  setText(textItem(1, "APAR turns emerging GenAI payment threats into rail-correct synthetic campaigns, tests layered defenses, and preserves a human promotion boundary."), { size: 21, color: C.textSoft });
  setText(textItem(1, "SYNTHETIC • OFFLINE • REPLAYABLE"), { typeface: "Menlo", size: 11, bold: true, color: C.orangeHot, align: "center" });
  setText(textItem(1, "Dylan Moraes  •  Competition submission"), { size: 14, color: C.muted });
  const pill = shapesAt(1, (entry) => at(entry, 42, 494, 288, 32))[0];
  setRule(pill, { left: 42, top: 494, width: 108, height: 2 }, C.orange);
  const proofLabel = textItem(1, "SYNTHETIC • OFFLINE • REPLAYABLE");
  setPosition(proofLabel, { left: 42, top: 508, width: 390, height: 24 });
  setText(proofLabel, { typeface: "Menlo", size: 11, bold: true, color: C.orangeHot, align: "left" });
  const frame = shapesAt(1, (entry) => at(entry, 656, 40, 590, 610))[0];
  setPanel(frame, "none", C.lineStrong, 1);
  const image = items(1, "image")[0];
  if (image) {
    image.target.borderRadius = "rounded-none";
    image.target.crop = { left: 0, top: 0, right: 0, bottom: 0.32694 };
  }
}

// 02 — Threat capability delta as a flat editorial sequence.
{
  styleHeader(2, 39);
  const cards = shapesAt(2, (entry) => {
    const box = entry.bbox ?? [];
    return within(box[1], 182) && within(box[2], 260) && within(box[3], 210);
  });
  cards.forEach((card, index) => {
    const x = [42, 340, 638, 936][index];
    setRule(card, { left: x, top: 184, width: 2, height: 206 }, index === 2 ? C.orange : C.line);
  });
  const slideNumber = textsAt(2, (entry) => entry.text === "02" && (entry.bbox ?? [])[0] >= 1160)[0];
  setPosition(slideNumber, { left: 1184, top: 36, width: 54, height: 22 });
  setText(slideNumber, { typeface: "Menlo", size: 11, bold: false, color: C.muted, align: "right" });
  for (const item of textsAt(2, (entry) => /^0[1-4]$/.test(entry.text) && (entry.bbox ?? [])[0] < 1100)) {
    setPosition(item, { top: 184, height: 30 });
    setText(item, { typeface: "Georgia", size: 20, bold: false, color: C.orangeHot });
  }
  for (const item of textsAt(2, (entry) => ["PERSONALIZE", "ITERATE", "COORDINATE", "DELEGATE"].includes(entry.text))) {
    setPosition(item, { top: 232, height: 38 });
    setText(item, { typeface: "Menlo", size: 16, bold: true, color: C.text });
  }
  for (const item of textsAt(2, (entry) => entry.text.includes("\n") && !entry.text.includes("Campaign-aware"))) {
    setPosition(item, { top: 292, height: 72 });
    setText(item, { size: 16, color: C.muted });
  }
  for (const item of textsAt(2, (entry) => entry.text === "→")) setText(item, { size: 18, bold: false, color: C.lineStrong, align: "center" });
  const band = shapesAt(2, (entry) => at(entry, 42, 448, 1196, 138))[0];
  setRule(band, { left: 42, top: 454, width: 1196, height: 2 }, C.lineStrong);
  setText(textItem(2, "Transaction classifier"), { size: 16, color: C.muted });
  setText(textItem(2, "Campaign-aware assurance system"), { typeface: "Georgia", size: 28, bold: false, color: C.text });
  setText(textItem(2, "Behavior + lifecycle + economics + authority + governance"), { typeface: "Menlo", size: 12, color: C.textSoft });
}

// 03 — Assurance loop with open typographic planes.
{
  styleHeader(3, 39);
  const cards = shapesAt(3, (entry) => {
    const box = entry.bbox ?? [];
    return within(box[1], 188) && within(box[2], 214) && within(box[3], 236);
  });
  cards.forEach((card, index) => {
    const x = [42, 284, 526, 768, 1010][index];
    setRule(card, { left: x, top: 188, width: 214, height: index === 2 ? 3 : 1 }, index === 2 ? C.green : C.lineStrong);
  });
  for (const item of textsAt(3, (entry) => ["IDENTIFY", "GENERATE", "DEFEND", "VERIFY", "GOVERN"].includes(entry.text))) {
    setText(item, { typeface: "Menlo", size: 12, bold: true, color: item.text === "DEFEND" ? C.green : C.orangeHot });
  }
  for (const item of textsAt(3, (entry) => entry.text.includes("\n"))) setText(item, { typeface: "Georgia", size: 19, bold: false, color: C.text });
  for (const item of textsAt(3, (entry) => entry.text === "→")) setText(item, { size: 22, bold: true, color: C.orange, align: "center" });
  for (const line of shapesAt(3, (entry) => (entry.bbox ?? [])[1] >= 470)) {
    const box = line.bbox ?? [];
    if (box[2] < 30) {
      setRule(line, { left: box[0] + 4, top: 479, width: 9, height: 9 }, box[0] > 1000 ? C.green : C.orange);
    } else {
      setRule(line, { left: 146, top: 483, width: 988, height: 1 }, C.lineStrong);
    }
  }
  setText(textItem(3, "Bounded feedback returns failed controls to the research backlog—never directly to deployment."), { size: 19, color: C.textSoft, align: "center" });
}

// 04 — Experiment record as an alternating evidence timeline.
{
  styleHeader(4, 39);
  const cards = shapesAt(4, (entry) => {
    const box = entry.bbox ?? [];
    return within(box[2], 210) && within(box[3], 144);
  });
  cards.forEach((card, index) => {
    const x = [58, 294, 530, 766, 1002][index];
    const top = [166, 348, 166, 348, 166][index];
    setRule(card, { left: x, top, width: 2, height: 144 }, index === 4 ? C.green : C.lineStrong);
  });
  const rail = shapesAt(4, (entry) => at(entry, 88, 316, 1100, 3))[0];
  setPanel(rail, C.lineStrong, C.lineStrong, 0);
  const dots = shapesAt(4, (entry) => within((entry.bbox ?? [])[2], 18) && within((entry.bbox ?? [])[3], 18));
  dots.forEach((dot, index) => {
    const x = [159, 395, 631, 867, 1103][index];
    setRule(dot, { left: x, top: 311, width: 10, height: 10 }, index === dots.length - 1 ? C.green : C.orange);
  });
  for (const item of textsAt(4, (entry) => /^V/.test(entry.text))) setText(item, { typeface: "Georgia", size: 18, bold: false, color: item.text === "V5c" ? C.green : C.orangeHot });
  for (const item of textsAt(4, (entry) => ["Workload cap failed", "Missing metrics failed closed", "Perfect score rejected", "Real rail evidence", "Graph arm selected"].includes(entry.text))) setText(item, { typeface: "Georgia", size: 17, bold: false, color: C.text });
  for (const item of textsAt(4, (entry) => entry.text.includes("\n"))) setText(item, { size: 14, color: C.muted });
  setText(textItem(4, "Novelty: not a single algorithm—an evidence-governed loop that learns from its own failure modes."), { typeface: "Georgia", size: 20, bold: false, color: C.text, align: "center" });
}

// 05 — Recovered diagnostic comparison; keep the data table, remove dashboard chrome.
{
  styleHeader(5, 39);
  setText(textItem(5, "Verified recovered diagnostics — synthetic and non-authoritative"), { typeface: "Menlo", size: 12, bold: true, color: C.orangeHot });
  const tableColumns = [
    { left: 42, width: 300 },
    { left: 342, width: 170 },
    { left: 512, width: 170 },
    { left: 682, width: 170 },
    { left: 852, width: 216 },
    { left: 1068, width: 170 },
  ];
  const tableRows = [
    { panelTop: 200, panelHeight: 48, textTop: 214, textHeight: 20 },
    { panelTop: 248, panelHeight: 66, textTop: 269, textHeight: 25 },
    { panelTop: 314, panelHeight: 66, textTop: 335, textHeight: 25 },
    { panelTop: 380, panelHeight: 66, textTop: 401, textHeight: 25 },
    { panelTop: 446, panelHeight: 66, textTop: 467, textHeight: 25 },
  ];
  for (const row of tableRows) {
    const panels = shapesAt(5, (entry) => within((entry.bbox ?? [])[1], row.panelTop) && within((entry.bbox ?? [])[3], row.panelHeight)).sort((a, b) => a.bbox[0] - b.bbox[0]);
    const labels = textsAt(5, (entry) => within((entry.bbox ?? [])[1], row.textTop) && within((entry.bbox ?? [])[3], row.textHeight)).sort((a, b) => a.bbox[0] - b.bbox[0]);
    tableColumns.forEach((column, index) => {
      setPosition(panels[index], { left: column.left, width: column.width });
      setPosition(labels[index], { left: column.left + 10, width: column.width - 20 });
    });
  }
  for (const shape of shapesAt(5, () => true)) {
    const box = shape.bbox ?? [];
    if (within(box[1], 200) && within(box[3], 48)) setPanel(shape, "none", C.lineStrong, 0.6);
    else if (within(box[1], 248) || within(box[1], 314)) setPanel(shape, "none", C.line, 0.5);
    else if (within(box[1], 380)) setPanel(shape, C.surface2, C.green, 0.8);
    else if (within(box[1], 446)) setPanel(shape, "none", C.line, 0.5);
    else if (within(box[1], 548)) setRule(shape, { left: 42, top: 548, width: 1196, height: 2 }, C.orange);
  }
  for (const item of items(5, "textbox")) {
    const box = item.bbox ?? [];
    if (within(box[1], 214, 8)) setText(item, { typeface: "Menlo", size: 11, bold: true, color: C.text });
    else if (box[1] >= 260 && box[1] <= 500) setText(item, { typeface: box[0] > 330 ? "Menlo" : "Arial", size: 15, bold: box[1] >= 390 && box[1] < 440, color: box[1] >= 450 ? C.redSoft : (box[1] >= 390 && box[1] < 440 ? C.green : C.textSoft) });
    else if (box[1] >= 560) setText(item, { typeface: "Menlo", size: 16, bold: true, color: C.orangeHot, align: "center" });
  }
}

// 06 — Portable model proof with documentary screenshot and typeset facts.
{
  styleHeader(6, 39);
  const frame = shapesAt(6, (entry) => at(entry, 34, 160, 762, 460))[0];
  setPanel(frame, "none", C.lineStrong, 1);
  const image = items(6, "image")[0];
  if (image) image.target.borderRadius = "rounded-none";
  const stats = shapesAt(6, (entry) => {
    const box = entry.bbox ?? [];
    return within(box[0], 830) && within(box[2], 408) && within(box[3], 86);
  });
  stats.forEach((stat, index) => {
    const top = [168, 272, 376, 480][index];
    setRule(stat, { left: 830, top, width: 408, height: index === 3 ? 2 : 1 }, index === 3 ? C.green : C.lineStrong);
  });
  for (const item of textsAt(6, (entry) => ["3 ×", "46", "12", "0.0"].includes(entry.text))) setText(item, { typeface: "Georgia", size: 31, bold: false, color: item.text === "0.0" ? C.green : C.orangeHot });
  for (const item of textsAt(6, (entry) => ["CatBoost members", "frozen causal features", "hash-bound scenarios", "max replay error"].includes(entry.text))) setText(item, { size: 17, bold: true, color: C.text });
  const actionLine = textItem(6, "Approve • Challenge • Review hold • Decline hold");
  setPosition(actionLine, { top: 584, height: 28 });
  setText(actionLine, { typeface: "Menlo", size: 12, bold: true, color: C.muted, align: "center" });
}

// 07 — Case-centric investigation with a contained evidence card.
{
  styleHeader(7, 39);
  const frame = shapesAt(7, (entry) => {
    const box = entry.bbox ?? [];
    return within(box[0], 386) && box[1] >= 150 && box[1] <= 162 && within(box[2], 860) && box[3] >= 484 && box[3] <= 496;
  })[0];
  setPosition(frame, { left: 386, top: 160, width: 860, height: 486 });
  setPanel(frame, "none", C.lineStrong, 1);
  const image = items(7, "image")[0];
  if (image) {
    setPosition(image, { left: 394, top: 168, width: 844, height: 470 });
    image.target.borderRadius = "rounded-none";
  }
  const pill = shapesAt(7, (entry) => at(entry, 42, 174, 150, 32))[0];
  setPosition(pill, { left: 42, top: 168, width: 304, height: 386 });
  setPanel(pill, "none", C.lineStrong, 0.8);
  const caseLabel = textItem(7, "CASE-CENTRIC");
  setPosition(caseLabel, { left: 62, top: 188, width: 250, height: 22 });
  setText(caseLabel, { typeface: "Menlo", size: 10, bold: true, color: C.green, align: "left" });
  const metricLayout = [
    ["14", 232, "entities", 278],
    ["10", 332, "payment edges", 378],
    ["$500", 432, "conserved synthetic ledger", 478],
  ];
  for (const [value, valueTop, label, labelTop] of metricLayout) {
    const valueItem = textItem(7, value);
    const labelItem = textItem(7, label);
    setPosition(valueItem, { left: 62, top: valueTop, width: 250, height: 46 });
    setPosition(labelItem, { left: 62, top: labelTop, width: 250, height: 34 });
    setText(valueItem, { typeface: "Georgia", size: 38, bold: false, color: C.orangeHot });
    setText(labelItem, { size: 15, color: C.muted });
  }
  const pending = shapesAt(7, (entry) => at(entry, 42, 574, 304, 64))[0];
  setPosition(pending, { left: 42, top: 570, width: 304, height: 68 });
  setPanel(pending, "none", C.lineStrong, 0.8);
  const pendingText = textItem(7, "Analyst-time benefit: evidence pending");
  setPosition(pendingText, { left: 58, top: 590, width: 272, height: 28 });
  setText(pendingText, { typeface: "Menlo", size: 10, bold: true, color: C.textSoft, align: "left" });
}

// 08 — Deterministic authority proof as a verification ledger.
{
  styleHeader(8, 39);
  const checks = shapesAt(8, (entry) => {
    const box = entry.bbox ?? [];
    return [42, 452].some((x) => within(box[0], x)) && within(box[2], 374) && within(box[3], 88);
  });
  checks.forEach((check) => {
    const [left, top] = check.bbox;
    setRule(check, { left, top, width: 374, height: 1 }, C.lineStrong);
  });
  const circles = shapesAt(8, (entry) => within((entry.bbox ?? [])[2], 34) && within((entry.bbox ?? [])[3], 34));
  circles.forEach((circle) => {
    const [left, top] = circle.bbox;
    setRule(circle, { left, top: top + 12, width: 8, height: 8 }, C.green);
  });
  for (const item of textsAt(8, (entry) => entry.text === "✓")) setText(item, { size: 1, bold: false, color: C.bg, align: "center" });
  for (const item of textsAt(8, (entry) => ["Agent identity", "User mandate", "Scope + amount", "Merchant + cart", "Expiry + nonce", "Replay rejection"].includes(entry.text))) {
    const [left, top] = item.bbox;
    setPosition(item, { left: left - 28, top: top - 4, width: 308, height: 36 });
    setText(item, { typeface: "Georgia", size: 19, bold: false, color: C.text });
  }
  const failure = shapesAt(8, (entry) => at(entry, 42, 554, 784, 88))[0];
  setRule(failure, { left: 42, top: 554, width: 784, height: 2 }, C.red);
  setText(textItem(8, "DEFINITIVE FAILURE"), { typeface: "Menlo", size: 11, bold: true, color: C.redSoft });
  setText(textItem(8, "DECLINE_HOLD before statistical risk"), { typeface: "Menlo", size: 20, bold: true, color: C.text });
  const frame = shapesAt(8, (entry) => at(entry, 924, 154, 286, 496))[0];
  setPanel(frame, "none", C.lineStrong, 1);
  const image = items(8, "image")[0];
  if (image) {
    image.target.borderRadius = "rounded-none";
    image.target.crop = { left: 0, top: 0.465, right: 0, bottom: 0.30636 };
  }
}

// 09 — Evidence governance as a restrained staircase of evidence cards.
{
  styleHeader(9, 39);
  const cards = shapesAt(9, (entry) => {
    const box = entry.bbox ?? [];
    return box[1] >= 170 && box[1] <= 300 && box[2] >= 200 && box[2] <= 220 && box[3] >= 230;
  }).sort((a, b) => a.bbox[0] - b.bbox[0]);
  const cardLefts = [42, 286.5, 531, 775.5, 1020];
  cards.forEach((card, index) => {
    setPosition(card, { left: cardLefts[index], width: 218 });
    if (index === 0) setPanel(card, "none", C.orange, 1.2);
    else if (index === 4) setPanel(card, "none", C.green, 1.2);
    else setPanel(card, "none", C.line, 0.8);
  });
  const cardCopy = [
    ["01", "Causal features", "Future append + equal-time isolation"],
    ["02", "Executed controls", "Shuffle, rename, leakage, single-class"],
    ["03", "Immutable evidence", "Models, traces, metrics, manifests hashed"],
    ["04", "Independent replay", "Fresh environment re-scores all 12 cases"],
    ["05", "Human promotion", "The model cannot approve itself"],
  ];
  cardCopy.forEach(([number, titleText, bodyText], index) => {
    const left = cardLefts[index] + 18;
    setPosition(textItem(9, number), { left });
    setPosition(textItem(9, titleText), { left, width: 182 });
    setPosition(textItem(9, bodyText), { left, width: 182 });
  });
  for (const item of textsAt(9, (entry) => /^0[1-5]$/.test(entry.text))) setText(item, { typeface: "Georgia", size: 18, bold: false, color: C.orangeHot });
  for (const item of textsAt(9, (entry) => ["Causal features", "Executed controls", "Immutable evidence", "Independent replay", "Human promotion"].includes(entry.text))) setText(item, { typeface: "Georgia", size: 20, bold: false, color: C.text });
  for (const item of textsAt(9, (entry) => entry.text.includes("\n") && !entry.text.includes("Evidence governance"))) setText(item, { size: 14, color: C.muted });
  const promotionLabel = rewriteText(9, "Official chain", "Promotion boundary");
  const promotionValue = rewriteText(9, "Stops at Stage 60", "Human approval required");
  const promotionPill = rewriteText(9, "STAGE 70 NOT CLAIMED", "NO SELF-PROMOTION");
  for (const item of [textItem(9, "Portable bundle"), promotionLabel]) setText(item, { typeface: "Menlo", size: 10, bold: true, color: C.muted });
  setText(textItem(9, "52ed4c31…dbc900"), { typeface: "Menlo", size: 16, bold: true, color: C.text });
  setText(promotionValue, { typeface: "Menlo", size: 16, bold: true, color: C.amber });
  const pill = shapesAt(9, (entry) => at(entry, 912, 550, 250, 32))[0];
  setRule(pill, { left: 912, top: 584, width: 250, height: 2 }, C.amber);
  setPosition(promotionPill, { left: 912, top: 552, width: 250, height: 22 });
  setText(promotionPill, { typeface: "Menlo", size: 10, bold: true, color: C.amber, align: "left" });
  const notes = presentation.slides.items[8].speakerNotes;
  notes.textFrame.setText("Show that every evidence step stays auditable and that promotion remains a human decision; recovered metrics remain diagnostics.\n\n[Sources]\n- demo/sentinel-v5/manifest.json\n- evidence/sentinel-v5-recovered-metrics/verified-report.json\n- docs/submission/RELEASE_CHECKLIST.md\n[/Sources]");
  notes.setVisible(true);
}

// 10 — Explicit proof/claim boundary without colored container panels.
{
  styleHeader(10, 39);
  const proof = shapesAt(10, (entry) => at(entry, 42, 170, 568, 438))[0];
  const boundary = shapesAt(10, (entry) => at(entry, 632, 170, 606, 438))[0];
  setRule(proof, { left: 620, top: 188, width: 1, height: 378 }, C.lineStrong);
  setRule(boundary, { left: 42, top: 594, width: 1196, height: 1 }, C.lineStrong);
  setText(textItem(10, "WHAT WE PROVE"), { typeface: "Menlo", size: 14, bold: true, color: C.green });
  setText(textItem(10, "WHAT WE DO NOT CLAIM"), { typeface: "Menlo", size: 14, bold: true, color: C.redSoft });
  for (const item of textsAt(10, (entry) => entry.text === "✓")) setText(item, { size: 17, bold: true, color: C.green });
  for (const item of textsAt(10, (entry) => entry.text === "×")) setText(item, { size: 20, bold: true, color: C.redSoft });
  for (const item of textsAt(10, (entry) => !["WHAT WE PROVE", "WHAT WE DO NOT CLAIM", "✓", "×"].includes(entry.text) && (entry.bbox ?? [])[1] >= 250 && (entry.bbox ?? [])[1] < 590)) setText(item, { size: 17, bold: false, color: C.textSoft });
  setText(rewriteText(10, "Accepted official Stage 70 capacity evidence", "Production-readiness or scale evidence"), { size: 17, bold: false, color: C.textSoft });
  setText(rewriteText(10, "Full Sentinel as the champion", "Claims beyond the demonstrated model"), { size: 17, bold: false, color: C.textSoft });
  setText(textItem(10, "Synthetic evidence before assertion."), { typeface: "Georgia", size: 22, bold: false, color: C.text, align: "center" });
}

// 11 — Deployment path and close with restrained roadmap cards.
{
  styleHeader(11, 35);
  const title = textItem(11, "From competition prototype to governed payment assurance");
  setPosition(title, { top: 58, width: 1120, height: 72 });
  setText(title, { typeface: "Georgia", size: 35, bold: false, color: C.text });
  const rule = shapesAt(11, (entry) => {
    const box = entry.bbox ?? [];
    return box[1] >= 128 && box[1] <= 138 && box[3] <= 4;
  })[0];
  setPosition(rule, { top: 148 });
  const cards = shapesAt(11, (entry) => {
    const box = entry.bbox ?? [];
    return within(box[1], 184) && box[2] >= 210 && box[2] <= 220 && within(box[3], 218);
  }).sort((a, b) => a.bbox[0] - b.bbox[0]);
  const cardLefts = [42, 286.5, 531, 775.5, 1020];
  cards.forEach((card, index) => {
    setPosition(card, { left: cardLefts[index], width: 218 });
    if (index === 0) setPanel(card, "none", C.orange, 1.2);
    else if (index === 4) setPanel(card, "none", C.green, 1.2);
    else setPanel(card, "none", C.line, 0.8);
  });
  const cardCopy = [
    ["1", "OFFLINE REPLAY", "Authorized historical samples"],
    ["2", "SHADOW MODE", "Live features; no authority"],
    ["3", "ASSISTED OPS", "Human-controlled friction"],
    ["4", "CHALLENGER", "Limited traffic + rollback"],
    ["5", "GOVERNED SCALE", "Continuous adversarial regression"],
  ];
  cardCopy.forEach(([number, titleText, bodyText], index) => {
    const left = cardLefts[index] + 18;
    setPosition(textItem(11, number), { left });
    setPosition(textItem(11, titleText), { left, width: 182 });
    setPosition(textItem(11, bodyText), { left, width: 182 });
  });
  const arrowLefts = [260, 504.5, 749, 993.5];
  textsAt(11, (entry) => entry.text === "→").sort((a, b) => a.bbox[0] - b.bbox[0]).forEach((arrow, index) => setPosition(arrow, { left: arrowLefts[index], width: 26.5 }));
  for (const item of textsAt(11, (entry) => /^[1-5]$/.test(entry.text))) setText(item, { typeface: "Georgia", size: 19, bold: false, color: C.orangeHot });
  for (const item of textsAt(11, (entry) => ["OFFLINE REPLAY", "SHADOW MODE", "ASSISTED OPS", "CHALLENGER", "GOVERNED SCALE"].includes(entry.text))) setText(item, { typeface: "Menlo", size: 13, bold: true, color: C.text });
  for (const item of textsAt(11, (entry) => entry.text.includes("\n") && (entry.bbox ?? [])[1] < 430)) setText(item, { size: 14, color: C.muted });
  const close = shapesAt(11, (entry) => at(entry, 42, 460, 1196, 144))[0];
  setRule(close, { left: 42, top: 460, width: 1196, height: 2 }, C.lineStrong);
  setText(textItem(11, "APAR"), { typeface: "Georgia", size: 38, bold: false, color: C.text });
  setText(textItem(11, "Assure the campaign. Verify the intent.\nPromote only the evidence."), { typeface: "Georgia", size: 28, bold: false, color: C.text });
  const pill = shapesAt(11, (entry) => at(entry, 1020, 506, 166, 32))[0];
  clearShape(pill);
  setText(textItem(11, "DEMO READY"), { typeface: "Menlo", size: 10, bold: true, color: C.green, align: "center" });
  setText(textItem(11, "Adaptive Payment Assurance Range  •  Dylan Moraes"), { typeface: "Menlo", size: 11, color: C.muted });
}

async function writeBlob(target, blob) {
  await fs.writeFile(target, new Uint8Array(await blob.arrayBuffer()));
}

await fs.mkdir(OUTPUT_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });
for (const [index, slide] of presentation.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  await writeBlob(path.join(OUTPUT_DIR, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(OUTPUT_DIR, `${stem}.layout.json`), await layout.text());
}
await writeBlob(path.join(OUTPUT_DIR, "deck-montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(FINAL_PPTX);
