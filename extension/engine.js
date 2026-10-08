// Runs the exported model in the browser. Mirrors src/model.py (TicketModel.predict).

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

export function prepare(blob) {
  const v = blob.vectorizer;
  const m = blob.matrix;
  const rows = [];
  for (let i = 0; i < blob.tickets.ids.length; i++) {
    const a = m.indptr[i], b = m.indptr[i + 1];
    rows.push({ idx: m.indices.slice(a, b), val: m.data.slice(a, b) });
  }
  return {
    blob, rows, cfg: blob.config,
    vocab: new Map(Object.entries(v.vocab)),
    size: v.idf.length,
    scratch: new Float64Array(v.idf.length),
    words: new Array(rows.length).fill(null),
    pii: ["EID_PATTERN", "EMAIL_PATTERN", "PHONE_PATTERN"].map((k) => new RegExp(blob.config[k])),
  };
}

function vectorize(model, text) {
  const v = model.blob.vectorizer;
  let s = text.replace(/\s+/g, " ").trim().slice(0, model.cfg.MAX_TEXT_CHARS);
  if (v.lowercase) s = s.toLowerCase();
  const toks = s.match(/[\p{L}\p{N}_]{2,}/gu) || [];
  const counts = new Map();
  for (let n = v.ngram_range[0]; n <= v.ngram_range[1]; n++) {
    for (let i = 0; i + n <= toks.length; i++) {
      const j = model.vocab.get(n === 1 ? toks[i] : toks.slice(i, i + n).join(" "));
      if (j !== undefined) counts.set(j, (counts.get(j) || 0) + 1);
    }
  }
  const q = new Float64Array(model.size);
  let norm = 0;
  for (const [j, c] of counts) {
    q[j] = (v.sublinear_tf ? 1 + Math.log(c) : c) * v.idf[j];
    norm += q[j] * q[j];
  }
  norm = Math.sqrt(norm) || 1;
  for (const j of counts.keys()) q[j] /= norm;
  return { q, nz: [...counts.keys()] };
}

const dotRow = (row, q) => {
  let s = 0;
  for (let i = 0; i < row.idx.length; i++) s += row.val[i] * q[row.idx[i]];
  return s;
};

function quantile(sorted, p) {
  const pos = p * (sorted.length - 1), lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

// Median, quartiles and best case of incidents at least SIMILARITY_CUTOFF similar (up to SIMILAR_MAX);
// fewer than SIMILAR_MIN close matches: the nearest SIMILAR_TICKETS_K, flagged rough.
function timeEstimate(cfg, hours, sims) {
  const k = Math.min(cfg.SIMILAR_TICKETS_K, hours.length);
  let pick = [];
  for (let i = 0; i < Math.min(cfg.SIMILAR_MAX, hours.length); i++) {
    if (sims[i] >= cfg.SIMILARITY_CUTOFF) pick.push(hours[i]);
  }
  const rough = pick.length < cfg.SIMILAR_MIN;
  if (rough) pick = hours.slice(0, k);
  pick.sort((a, b) => a - b);
  return {
    median: quantile(pick, 0.5), low: quantile(pick, 0.25), high: quantile(pick, 0.75),
    best: quantile(pick, cfg.BEST_CASE_QUANTILE), matches: pick.length, rough,
  };
}

function wordsOf(model, i) {
  if (!model.words[i]) model.words[i] = new Set(model.blob.tickets.texts[i].toLowerCase().match(/[a-z0-9]+/g) || []);
  return model.words[i];
}

// Most typical wording among close matches, built only from words that recur across them.
function suggest(model, nb) {
  const cfg = model.cfg, texts = model.blob.tickets.texts;
  if (nb.length < cfg.SUGGEST_MIN) return null;
  const seen = new Map();
  for (const i of nb) for (const w of wordsOf(model, i)) seen.set(w, (seen.get(w) || 0) + 1);
  const need = Math.max(cfg.SUGGEST_WORD_MIN_COUNT, Math.ceil(cfg.SUGGEST_WORD_SHARE * (nb.length - 1)));
  const dense = model.scratch;
  const typical = nb.map((i) => {
    const r = model.rows[i];
    r.idx.forEach((j, t) => { dense[j] = r.val[t]; });
    let total = 0;
    for (const o of nb) total += dotRow(model.rows[o], dense);
    r.idx.forEach((j) => { dense[j] = 0; });
    return (total - 1) / (nb.length - 1);
  });
  const order = nb.map((_, j) => j).sort((a, b) =>
    (-Math.round(typical[a] * 1e4) / 1e4) - (-Math.round(typical[b] * 1e4) / 1e4) ||
    texts[nb[a]].length - texts[nb[b]].length);
  for (const j of order) {
    const i = nb[j], w = wordsOf(model, i);
    if (w.size && [...w].every((x) => seen.get(x) - 1 >= need) && !model.pii.some((p) => p.test(texts[i]))) return texts[i];
  }
  return null;
}

function fieldGuess(spec, nz, q) {
  if (spec.default !== undefined) return [[spec.default, 1]];
  const z = spec.coef.map((w, c) => nz.reduce((s, j) => s + w[j] * q[j], spec.intercept[c]));
  const p = z.length === 1 ? [1 - sigmoid(z[0]), sigmoid(z[0])]
    : (() => { const m = Math.max(...z), e = z.map((x) => Math.exp(x - m)), t = e.reduce((a, b) => a + b); return e.map((x) => x / t); })();
  return p.map((x, c) => [spec.classes[c], x]).sort((a, b) => b[1] - a[1]).slice(0, 3);
}

export function predict(model, text) {
  const { blob, cfg } = model, t = blob.tickets;
  const { q, nz } = vectorize(model, text);
  const pInquiry = sigmoid(nz.reduce((s, j) => s + blob.type.coef[0][j] * q[j], blob.type.intercept[0]));

  const order = model.rows.map((r, i) => [i, dotRow(r, q)]).sort((a, b) => b[1] - a[1]);
  const close = order.slice(0, cfg.SUGGEST_MAX);
  const inc = order.filter(([i]) => t.is_incident[i]).slice(0, Math.max(cfg.SIMILAR_TICKETS_K, cfg.SIMILAR_MAX));

  return {
    p_inquiry: pInquiry,
    inquiry: pInquiry >= cfg.INQUIRY_CUTOFF,
    time: timeEstimate(cfg, inc.map(([i]) => t.hours[i]), inc.map(([, s]) => s)),
    similar: order.slice(0, cfg.SIMILAR_SHOWN).map(([i, s]) => ({ id: t.ids[i], text: t.texts[i], sim: s })),
    fields: Object.fromEntries(Object.entries(blob.fields).map(([n, spec]) => [n, fieldGuess(spec, nz, q)])),
    suggestion: suggest(model, close.filter(([, s]) => s >= cfg.SUGGEST_CUTOFF).map(([i]) => i)),
  };
}

// "8 h-24 h", "1-3 days" style label for a time in hours
export function bucketText(hours, edges) {
  const fmt = (h) => (h <= 24 ? `${h} h` : `${Math.round(h / 24)} days`);
  const i = edges.findIndex((e) => hours < e);
  if (i === 0) return `under ${fmt(edges[0])}`;
  if (i === -1) return `over ${fmt(edges[edges.length - 1])}`;
  return `${fmt(edges[i - 1])} to ${fmt(edges[i])}`;
}

export function hoursText(h) {
  if (h < 1) return `${Math.max(1, Math.round(h * 60))} min`;
  if (h < 48) return `${Math.round(h)} h`;
  return `${Math.round(h / 24)} days`;
}
