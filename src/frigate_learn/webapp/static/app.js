const POLL_MS = 5000;
const QUALITIES = ["useful", "bad", "duplicate", "ignore"];
const MAP50_KEY = "metrics/mAP50(B)";
const RECALL_KEY = "metrics/recall(B)";

const QUALITY_LABELS = {
  useful: "Useful",
  bad: "Bad",
  duplicate: "Duplicate",
  ignore: "Ignore",
};

const QUALITY_DESC = {
  useful: "keep — usable detection, good for training",
  bad: "reject — no usable object or corrupt frame",
  duplicate: "reject — same detection as another sample",
  ignore: "exclude — not relevant to this model",
};

const STEP_LABELS = {
  collect: ["Collect", "pull new detections and snapshots from Frigate"],
  verify: ["Verify", "label unverified samples with the VLM and write verified annotations"],
  build: ["Build", "emit a versioned YOLO dataset with deterministic splits"],
  train: ["Train", "fine-tune the configured YOLO model on the latest dataset"],
  benchmark: ["Benchmark", "score candidates against the golden dataset"],
  gate: ["Gate", "compare candidates with the baseline and record the verdict"],
  deploy: ["Deploy", "compile the winning weights to HEF for Hailo"],
};

const JOB_KIND_LABELS = {
  pipeline: "Pipeline",
  audit: "Audit",
  collect: "Collect",
  verify: "Verify",
};

function jobKindLabel(type) {
  return JOB_KIND_LABELS[type] || type;
}

const $ = (sel) => document.querySelector(sel);
const section = (name) => $("#view-" + name);

async function api(path, opts) {
  const res = await fetch(path, opts);
  let data = null;
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    try { data = await res.json(); } catch (e) { data = null; }
  }
  if (!res.ok) {
    const msg = (data && (data.error || data.detail)) || "HTTP " + res.status;
    throw new Error(String(msg));
  }
  return data;
}

function h(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}

function esc(s) {
  if (s === null || s === undefined) return "";
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function num(v, d) {
  if (v === null || v === undefined || v === "") return "";
  const n = typeof v === "number" ? v : Number(v);
  if (!Number.isFinite(n)) return "";
  return n.toFixed(d === undefined ? 3 : d);
}

function fmtTs(v) {
  if (v === null || v === undefined || v === "") return "";
  const d = typeof v === "number" && Number.isFinite(v) ? new Date(v * 1000) : new Date(v);
  if (Number.isNaN(d.getTime())) return String(v);
  return d.toLocaleString();
}

function fmtBytes(n) {
  if (!Number.isFinite(n)) return "";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i += 1; }
  return v.toFixed(v >= 100 || i === 0 ? 0 : 1) + " " + u[i];
}

function fmtDur(start, end) {
  if (!start) return "";
  const s = new Date(start).getTime();
  if (!Number.isFinite(s)) return "";
  const e = end ? new Date(end).getTime() : Date.now();
  const sec = Math.max(0, Math.round((e - s) / 1000));
  if (sec < 1) return "0s";
  if (sec < 60) return sec + "s";
  const min = Math.floor(sec / 60);
  if (min < 60) return min + "m " + (sec % 60) + "s";
  const hr = Math.floor(min / 60);
  return hr + "h " + (min % 60) + "m";
}

function pill(cls, text) {
  return `<span class="pill ${esc(cls)}">${esc(text)}</span>`;
}

function pillNode(text, quality, title) {
  const span = document.createElement("span");
  span.className = "pill " + (quality || "");
  span.textContent = text;
  if (title) span.title = title;
  return span;
}

function labelBadge(src, value, tone) {
  const b = h("div", "label-badge" + (tone ? " badge-" + tone : ""));
  b.appendChild(h("span", "label-src", src));
  b.appendChild(h("span", "label-name", value));
  return b;
}

function countCard(title, number, sub) {
  const card = h("div", "card");
  card.appendChild(h("h2", null, title));
  card.appendChild(h("div", "number", number));
  card.appendChild(h("div", "sub", sub));
  return card;
}

function tableCaption(title) {
  return h("h2", "card-title", title);
}

function errorBox(container, message) {
  container.appendChild(h("div", "placeholder", String(message)));
}

function renderPipelinePanel(data) {
  const panel = h("div", "pipeline-panel");
  panel.appendChild(h("h2", "card-title", "Run pipeline"));
  const steps = data.steps || [];
  const enabled = new Set(data.auto_enable || []);
  const selected = new Set(steps.length ? (enabled.size ? enabled : steps) : []);
  const grid = h("div", "step-grid");
  const checks = new Map();
  for (const step of steps) {
    const [label, desc] = STEP_LABELS[step] || [step, ""];
    const lab = h("label", null);
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = selected.has(step);
    cb.value = step;
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(label));
    lab.title = step + (desc ? " — " + desc : "");
    grid.appendChild(lab);
    checks.set(step, cb);
  }
  panel.appendChild(grid);
  panel.appendChild(h("div", "status-line",
    "selected stages run in order as one job; only one job runs at a time"));

  const collect = data.collection || {};
  const runOptions = h("div", "run-options");
  const daysWrap = h("div", "option");
  daysWrap.appendChild(h("span", "option-label", "Collect lookback"));
  const daysInput = document.createElement("input");
  daysInput.type = "number";
  daysInput.min = "1";
  daysInput.step = "1";
  daysInput.value = String(collect.default_days || 7);
  daysInput.title = "How many days of Frigate history the collect stage pulls (defaults to collection.default_days in config).";
  daysWrap.appendChild(daysInput);
  daysWrap.appendChild(h("span", "option-unit", "days"));
  runOptions.appendChild(daysWrap);
  const limitWrap = h("div", "option option-wide");
  limitWrap.appendChild(h("span", "option-label", "Max items to import"));
  const limitInput = document.createElement("input");
  limitInput.type = "number";
  limitInput.min = "1";
  limitInput.step = "1";
  limitInput.placeholder = "unlimited";
  limitInput.title = "Cap on how many segments the collect stage imports (empty = collection.max_reviews / max_events in config).";
  limitWrap.appendChild(limitInput);
  limitWrap.appendChild(h("span", "option-unit", "items"));
  runOptions.appendChild(limitWrap);
  panel.appendChild(runOptions);

  if (!data.vlm_enabled) {
    panel.appendChild(h("div", "audit-banner",
      "VLM is disabled (vlm.enabled=false) — the Verify stage will be skipped until it is enabled."));
  }

  const controls = h("div", "buttons-row");
  const dryLab = h("label", null);
  const dry = document.createElement("input");
  dry.type = "checkbox";
  dryLab.appendChild(dry);
  dryLab.appendChild(document.createTextNode(" Dry run"));
  dryLab.title = "validate and report without writing data";
  const keepLab = h("label", null);
  const keep = document.createElement("input");
  keep.type = "checkbox";
  keepLab.appendChild(keep);
  keepLab.appendChild(document.createTextNode(" Keep going"));
  keepLab.title = "continue past a failed stage instead of halting the pipeline";
  const runBtn = h("button", "btn btn-run", "Run pipeline");
  const status = h("span", "status-line");
  const updateEnabled = () => {
    runBtn.disabled = hasRunning || ![...checks.values()].some((c) => c.checked);
  };
  for (const cb of checks.values()) cb.addEventListener("change", updateEnabled);
  runBtn.addEventListener("click", async () => {
    const chosen = [...checks.entries()]
      .filter(([, cb]) => cb.checked)
      .map(([s]) => s);
    const daysRaw = parseInt(daysInput.value, 10);
    const days = Number.isInteger(daysRaw) && daysRaw >= 1 ? daysRaw : null;
    const limitRaw = parseInt(limitInput.value, 10);
    const limit = Number.isInteger(limitRaw) && limitRaw >= 1 ? limitRaw : null;
    runBtn.disabled = true;
    status.textContent = "starting job…";
    try {
      const res = await api("/api/jobs/run", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          steps: chosen,
          dry_run: dry.checked,
          days,
          limit,
          keep_going: keep.checked,
        }),
      });
      status.textContent = `job ${res.job_id} started${dry.checked ? " (dry run)" : ""} — live log open below`;
      hasRunning = true;
      renderBanner({ id: res.job_id, status: "running" });
      openJobDrawer(res.job_id);
      views.overview.stale = true;
      render("overview");
    } catch (e) {
      status.textContent = e.message;
      updateEnabled();
    }
  });
  dry.addEventListener("change", updateEnabled);
  keep.addEventListener("change", updateEnabled);
  controls.appendChild(runBtn);
  controls.appendChild(dryLab);
  controls.appendChild(keepLab);
  controls.appendChild(status);
  panel.appendChild(controls);

  const auditPanel = h("div", "audit-run");
  auditPanel.appendChild(h("h2", "card-title", "Audit dataset"));
  auditPanel.appendChild(h("div", "status-line",
    "SAM + VLM reconciliation of collected samples — independent of the pipeline above"));
  const auditOpts = h("div", "run-options");
  const resumeLab = h("label", null);
  const resume = document.createElement("input");
  resume.type = "checkbox";
  resume.checked = true;
  resumeLab.appendChild(resume);
  resumeLab.appendChild(document.createTextNode(" Resume"));
  resumeLab.title = "skip samples with a valid cached KEEP/DROP decision";
  auditOpts.appendChild(resumeLab);
  const samOnlyLab = h("label", null);
  const samOnly = document.createElement("input");
  samOnly.type = "checkbox";
  samOnlyLab.appendChild(samOnly);
  samOnlyLab.appendChild(document.createTextNode(" SAM only"));
  samOnlyLab.title = "run SAM only; do not call the VLM";
  auditOpts.appendChild(samOnlyLab);
  const auditLimitWrap = h("div", "option option-wide");
  auditLimitWrap.appendChild(h("span", "option-label", "Max samples to audit"));
  const auditLimit = document.createElement("input");
  auditLimit.type = "number";
  auditLimit.min = "1";
  auditLimit.step = "1";
  auditLimit.placeholder = "unlimited";
  auditLimit.title = "process at most N new samples (empty = all)";
  auditLimitWrap.appendChild(auditLimit);
  auditLimitWrap.appendChild(h("span", "option-unit", "samples"));
  auditOpts.appendChild(auditLimitWrap);
  auditPanel.appendChild(auditOpts);
  const auditControls = h("div", "buttons-row");
  const auditBtn = h("button", "btn btn-run", "Run Audit");
  const auditStatus = h("span", "status-line");
  auditControls.appendChild(auditBtn);
  auditControls.appendChild(auditStatus);
  auditPanel.appendChild(auditControls);
  auditBtn.addEventListener("click", async () => {
    const limitRaw = parseInt(auditLimit.value, 10);
    const limit = Number.isInteger(limitRaw) && limitRaw >= 1 ? limitRaw : null;
    auditBtn.disabled = true;
    auditStatus.textContent = "starting audit…";
    try {
      const res = await api("/api/jobs/audit", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          resume: resume.checked,
          limit,
          sam_only: samOnly.checked,
        }),
      });
      const opts = [];
      if (!resume.checked) opts.push("no resume");
      if (limit !== null) opts.push(`max ${limit} samples`);
      if (samOnly.checked) opts.push("SAM only");
      auditStatus.textContent = `audit job ${res.job_id} started` +
        (opts.length ? " (" + opts.join(", ") + ")" : "") + " — live log open below";
      hasRunning = true;
      renderBanner({ id: res.job_id, status: "running", type: "audit" });
      openJobDrawer(res.job_id);
      views.audit.stale = true;
      views.overview.stale = true;
      render("overview");
    } catch (e) {
      auditStatus.textContent = e.message;
      auditBtn.disabled = hasRunning;
    }
  });
  panel.appendChild(auditPanel);
  panel.appendChild(h("div", "status-line",
    "note: stages run in this server's process; stopping the server stops a running job"));
  return panel;
}

let drawerJobId = null;
let drawerTimer = null;
let drawerRendering = false;

function closeDrawer() {
  drawerJobId = null;
  if (drawerTimer) { clearInterval(drawerTimer); drawerTimer = null; }
  const d = $("#drawer");
  d.hidden = true;
  d.innerHTML = "";
}

function openJobDrawer(jobId) {
  drawerJobId = jobId;
  renderJobDrawer(jobId);
  if (drawerTimer) { clearInterval(drawerTimer); drawerTimer = null; }
  drawerTimer = setInterval(() => renderJobDrawer(jobId), 3000);
}

async function renderJobDrawer(jobId) {
  if (drawerRendering) return;
  drawerRendering = true;
  try {
    let job;
    try {
      job = await api("/api/jobs/" + encodeURIComponent(jobId));
    } catch (e) {
      return;
    }
    const d = $("#drawer");
    const wasOpen = !d.hidden;
    const prevScroll = wasOpen ? (d.querySelector(".drawer-body") || {}).scrollTop || 0 : 0;
    d.hidden = false;
    const head = h("div", "drawer-head");
    head.appendChild(h("h2", null, "job " + job.id.slice(0, 8)));
    head.appendChild(pillNode(job.status || "—", job.status || ""));
    const closeBtn = h("button", "drawer-close", "✕");
    closeBtn.title = "close";
    closeBtn.addEventListener("click", closeDrawer);
    head.appendChild(closeBtn);

    const body = h("div", "drawer-body");
    const meta = h("div", "drawer-meta");
    const fields = [
      ["id", job.id],
      ["kind", jobKindLabel(job.type)],
      ["started", fmtTs(job.started_at)],
      ["finished", job.finished_at ? fmtTs(job.finished_at) : "running"],
      ["duration", fmtDur(job.started_at, job.finished_at)],
    ];
    if (job.days) fields.push(["lookback", job.days + " days"]);
    if (job.limit) fields.push(["max items", job.limit]);
    if (job.keep_going) fields.push(["keep going", "yes"]);
    if (job.resume) fields.push(["resume", "yes"]);
    if (job.sam_only) fields.push(["sam only", "yes"]);
    for (const [k, v] of fields) {
      meta.appendChild(h("div", "k", k));
      meta.appendChild(h("div", "v", v === null || v === undefined || v === "" ? "—" : String(v)));
    }
    body.appendChild(meta);
    if (job.error) {
      body.appendChild(h("div", "drawer-error", job.error));
    }
    const reports = job.reports || [];
    if (reports.length) {
      body.appendChild(h("h2", "card-title", "Stage reports"));
      const repBox = h("div", "drawer-reports");
      for (const r of reports) {
        const rep = h("div", "drawer-rep");
        rep.appendChild(h("span", "rep-name", r.name));
        rep.appendChild(pillNode(r.status || "—", r.status || ""));
        rep.appendChild(h("span", "rep-msg", r.message || ""));
        repBox.appendChild(rep);
      }
      body.appendChild(repBox);
    }
    const log = job.log_tail || [];
    body.appendChild(h("h2", "card-title", "Log tail"));
    if (!log.length) {
      body.appendChild(h("div", "drawer-empty",
        job.status === "running" ? "waiting for log lines…" : "no log captured"));
    } else {
      body.appendChild(h("pre", "drawer-log", log.join("\n")));
    }
    d.innerHTML = "";
    d.appendChild(head);
    d.appendChild(body);
    const db = d.querySelector(".drawer-body");
    if (job.status === "running") {
      const logEl = body.querySelector(".drawer-log");
      if (logEl) logEl.scrollTop = logEl.scrollHeight;
      if (db) db.scrollTop = db.scrollHeight;
    } else if (wasOpen && db) {
      db.scrollTop = prevScroll;
    }
    if (job.status !== "running" && drawerTimer) {
      clearInterval(drawerTimer);
      drawerTimer = null;
    }
  } finally {
    drawerRendering = false;
  }
}

function plotBox(container) {
  const box = h("div", "plot");
  container.appendChild(box);
  return box;
}

function chartExtent(values) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) {
    if (v === null || v === undefined || !Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  if (lo === hi) { lo -= 0.05; hi += 0.05; }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

function chartTicks(lo, hi, n) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const mag = Math.pow(10, Math.floor(Math.log10(span / n)));
  const step = [1, 2, 2.5, 5, 10].find((m) => m * mag >= span / n - 1e-9) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

function chartNum(v) {
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(Math.abs(v) < 1 ? 2 : 1);
}

function chartLegend(series) {
  const wrap = h("div", "chart-legend");
  for (const s of series) {
    const item = h("span", "chart-legend-item");
    const sw = h("span", "chart-legend-swatch");
    sw.style.background = s.color;
    item.appendChild(sw);
    item.appendChild(h("span", null, s.label));
    wrap.appendChild(item);
  }
  return wrap;
}

function lineChartSVG(width, height, xs, series, xLabel) {
  const pl = 50, pr = 14, pt = 14, pb = 32;
  const iw = width - pl - pr, ih = height - pt - pb;
  const yExt = chartExtent(series.flatMap((s) => s.data));
  if (!yExt) return "";
  const x0 = xs[0], x1 = xs[xs.length - 1];
  const xExt = x0 === x1 ? [x0 - 0.5, x0 + 0.5] : [x0, x1];
  const xSpan = xExt[1] - xExt[0], ySpan = yExt[1] - yExt[0];
  const px = (x) => pl + ((x - xExt[0]) / xSpan) * iw;
  const py = (y) => pt + ((yExt[1] - y) / ySpan) * ih;
  let g = "";
  for (const y of chartTicks(yExt[0], yExt[1], 5)) {
    g += `<line x1="${pl}" y1="${py(y).toFixed(1)}" x2="${width - pr}" y2="${py(y).toFixed(1)}" class="chart-grid"/>`;
    g += `<text x="${pl - 6}" y="${(py(y) + 4).toFixed(1)}" text-anchor="end">${chartNum(y)}</text>`;
  }
  for (const x of chartTicks(xExt[0], xExt[1], 8)) {
    g += `<text x="${px(x).toFixed(1)}" y="${height - 12}" text-anchor="middle">${chartNum(x)}</text>`;
  }
  if (xLabel) {
    g += `<text x="${(pl + iw / 2).toFixed(1)}" y="${height - 2}" text-anchor="middle" class="chart-axis">${esc(xLabel)}</text>`;
  }
  for (const s of series) {
    let d = "", pen = false;
    s.data.forEach((v, i) => {
      if (v === null || v === undefined || !Number.isFinite(v)) { pen = false; return; }
      d += (pen ? "L" : "M") + px(xs[i]).toFixed(1) + " " + py(v).toFixed(1) + " ";
      pen = true;
    });
    const dash = (s.dash || []).map(String).join(",");
    const dashAttr = dash ? ";stroke-dasharray:" + dash : "";
    g += `<path d="${d}" style="stroke:${s.color};stroke-width:${s.width || 2}${dashAttr};fill:none;"/>`;
  }
  return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${g}</svg>`;
}

function scatterSVG(width, height, points) {
  const pl = 54, pr = 14, pt = 14, pb = 34;
  const iw = width - pl - pr, ih = height - pt - pb;
  const xExt = chartExtent(points.map((p) => p.x));
  const yExt = chartExtent(points.map((p) => p.y));
  if (!xExt || !yExt) return "";
  const xSpan = xExt[1] - xExt[0], ySpan = yExt[1] - yExt[0];
  const px = (x) => pl + ((x - xExt[0]) / xSpan) * iw;
  const py = (y) => pt + ((yExt[1] - y) / ySpan) * ih;
  let g = "";
  for (const y of chartTicks(yExt[0], yExt[1], 5)) {
    g += `<line x1="${pl}" y1="${py(y).toFixed(1)}" x2="${width - pr}" y2="${py(y).toFixed(1)}" class="chart-grid"/>`;
    g += `<text x="${pl - 6}" y="${(py(y) + 4).toFixed(1)}" text-anchor="end">${chartNum(y)}</text>`;
  }
  for (const x of chartTicks(xExt[0], xExt[1], 6)) {
    g += `<text x="${px(x).toFixed(1)}" y="${height - 12}" text-anchor="middle">${chartNum(x)}</text>`;
  }
  for (const p of points) {
    g += `<circle cx="${px(p.x).toFixed(1)}" cy="${py(p.y).toFixed(1)}" r="5" class="chart-dot"/>`;
  }
  g += `<text x="${(pl + iw / 2).toFixed(1)}" y="${height - 2}" text-anchor="middle" class="chart-axis">latency ms</text>`;
  g += `<text transform="rotate(-90 ${12} ${(pt + ih / 2).toFixed(1)})" x="${12}" y="${(pt + ih / 2).toFixed(1)}" text-anchor="middle" class="chart-axis">mAP50</text>`;
  return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${g}</svg>`;
}

function renderScatter(container, points) {
  if (!points.length) {
    container.innerHTML = "";
    container.appendChild(h("div", "plot-missing", "no valid data points"));
    return;
  }
  const width = Math.max(320, (container.clientWidth || 600) - 20);
  container.innerHTML = scatterSVG(width, 340, points);
}

function renderLineChart(container, xs, series) {
  if (!xs.length) {
    container.innerHTML = "";
    container.appendChild(h("div", "plot-missing", "no numeric epoch data"));
    return;
  }
  if (!chartExtent(series.flatMap((s) => s.data))) {
    container.innerHTML = "";
    container.appendChild(h("div", "plot-missing", "no valid data points"));
    return;
  }
  const width = Math.max(320, (container.clientWidth || 600) - 20);
  container.innerHTML = lineChartSVG(width, 340, xs, series, "epoch");
  container.appendChild(chartLegend(series));
}

const views = {
  overview: { fn: renderOverview, loaded: false, stale: false },
  benchmark: { fn: renderBenchmark, loaded: false, stale: false },
  quality: { fn: renderQuality, loaded: false, stale: false },
  audit: { fn: renderAudit, loaded: false, stale: false },
  datasets: { fn: renderDatasets, loaded: false, stale: false },
  training: { fn: renderTraining, loaded: false, stale: false },
  triage: { fn: renderTriage, loaded: false, stale: false },
};

let currentView = "overview";

async function render(name) {
  const v = views[name];
  if (v.inFlight) return;
  v.inFlight = true;
  try {
    await v.fn(section(name));
  } catch (e) {
    section(name).innerHTML = "";
    errorBox(section(name), e.message);
  } finally {
    v.inFlight = false;
    v.loaded = true;
    v.stale = false;
  }
}

function activate(name) {
  currentView = name;
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.view === name);
  });
  Object.keys(views).forEach((k) => {
    section(k).hidden = k !== name;
  });
  if (!views[name].loaded || views[name].stale) render(name);
}

document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => activate(t.dataset.view));
});

document.addEventListener("keydown", (ev) => {
  if (currentView !== "audit" || !auditState.detailId) return;
  const t = ev.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
  if (ev.key === "ArrowLeft") {
    auditNav(-1);
    ev.preventDefault();
  } else if (ev.key === "ArrowRight") {
    auditNav(1);
    ev.preventDefault();
  }
});

async function renderOverview(sec) {
  const data = await api("/api/overview");
  refreshStageStrip(data);
  sec.innerHTML = "";
  const cards = h("div", "cards");
  cards.appendChild(countCard("Samples", data.samples.total,
    `verified ${data.samples.verified} / unverified ${data.samples.unverified}`));
  cards.appendChild(countCard("Cameras", data.cameras.length, data.cameras.join(", ") || "none"));
  cards.appendChild(countCard("Disk free", fmtBytes(data.disk_free_bytes), "dataset root"));
  cards.appendChild(countCard("VLM", data.vlm_enabled ? "enabled" : "disabled",
    "labels samples in the Verify stage"));
  const quals = Object.keys(data.samples)
    .filter((k) => k.startsWith("quality_"))
    .filter((k) => data.samples[k] > 0)
    .map((k) => `${QUALITY_LABELS[k.slice(8)] || k.slice(8)}=${data.samples[k]}`)
    .join(", ");
  cards.appendChild(countCard("Quality verdicts", quals || "none", "per-verdict sample counts"));
  cards.appendChild(countCard("Auto pipeline", data.auto_enable.length,
    data.auto_enable.map((s) => (STEP_LABELS[s] || [s])[0]).join(", ") || "none"));
  cards.appendChild(countCard("Next dataset", data.next_build_version || "—", "version the next build will emit"));
  sec.appendChild(cards);

  sec.appendChild(renderPipelinePanel(data));

  sec.appendChild(tableCaption("Recent jobs"));
  const jobs = data.pipeline || [];
  if (!jobs.length) {
    sec.appendChild(h("div", "placeholder",
      "no jobs yet — pick the stages above and run the pipeline"));
  } else {
    const table = h("table");
    table.className = "jobs-table";
    table.innerHTML = `
      <thead><tr><th>id</th><th>kind</th><th>status</th><th>started</th>
      <th>duration</th><th>error</th><th></th></tr></thead>
      <tbody></tbody>`;
    const tbody = table.querySelector("tbody");
    for (const j of jobs) {
      const tr = h("tr");
      tr.innerHTML = `
        <td class="mono">${esc(j.id.slice(0, 8))}</td>
        <td>${esc(jobKindLabel(j.type))}</td>
        <td>${pill(j.status, j.status)}</td>
        <td>${esc(fmtTs(j.started_at))}</td>
        <td>${esc(fmtDur(j.started_at, j.finished_at))}</td>
        <td>${esc(j.error || "")}</td>`;
      const tdLog = h("td");
      const logBtn = h("button", "btn", j.status === "running" ? "live" : "log");
      logBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        openJobDrawer(j.id);
      });
      tdLog.appendChild(logBtn);
      tr.appendChild(tdLog);
      tr.addEventListener("click", () => openJobDrawer(j.id));
      tbody.appendChild(tr);
    }
    sec.appendChild(table);
    sec.appendChild(h("div", "status-line", "click a row or “log” to open the job with stage reports and log tail"));
  }
}

function refreshStageStrip(data) {
  const strip = $("#stage-strip");
  strip.innerHTML = "";
  const latest = data.pipeline && data.pipeline[0];
  if (!latest) {
    drawChips(strip, data.steps || [], []);
    return;
  }
  api("/api/jobs/" + latest.id)
    .then((detail) => drawChips(strip, data.steps || [], detail.reports || []))
    .catch(() => drawChips(strip, data.steps || [], []));
}

function drawChips(strip, steps, reports) {
  strip.innerHTML = "";
  const byStep = {};
  for (const r of reports) byStep[r.name] = r;
  for (const step of steps) {
    const rep = byStep[step];
    let cls = "chip";
    if (rep) {
      if (rep.status === "executed") cls = "chip chip-ok";
      else if (rep.status === "failed") cls = "chip chip-fail";
      else cls = "chip chip-skip";
    }
    const chip = h("span", cls, (STEP_LABELS[step] || [step])[0]);
    if (rep && rep.message) chip.title = rep.message;
    strip.appendChild(chip);
  }
}

async function renderBenchmark(sec) {
  const data = await api("/api/benchmark");
  const deps = await api("/api/deployments");
  sec.innerHTML = "";
  const latestByModel = {};
  for (const d of deps.deployments || []) {
    if (!(d.model_name in latestByModel)) latestByModel[d.model_name] = d;
  }

  sec.appendChild(h("div", "status-line",
    `golden dataset ${data.golden} · baseline ${data.baseline} · ` +
    `max latency ${num(data.max_latency_ms, 1)} ms · results updated ${fmtTs(data.results_updated_at) || "—"}`));

  const btnRow = h("div", "buttons-row");
  const runBtn = h("button", "btn", "Re-benchmark + gate");
  runBtn.disabled = hasRunning;
  const btnStatus = h("span", "status-line");
  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    btnStatus.textContent = "starting job…";
    try {
      const res = await api("/api/jobs/run", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ steps: ["benchmark", "gate"], dry_run: false }),
      });
      btnStatus.textContent = "job " + res.job_id + " started";
    } catch (e) {
      btnStatus.textContent = e.message;
      runBtn.disabled = hasRunning;
    }
  });
  btnRow.appendChild(runBtn);
  btnRow.appendChild(btnStatus);
  sec.appendChild(btnRow);

  sec.appendChild(tableCaption("Candidates"));
  const results = data.results || [];
  if (!results.length) {
    sec.appendChild(h("div", "placeholder",
      "no benchmark results yet — run “Re-benchmark + gate” above to seed the golden comparison"));
  } else {
    const table = h("table");
    table.innerHTML = `
      <thead><tr><th>model</th><th>version</th><th>mAP50</th><th>mAP50-95</th>
      <th>recall</th><th>fp rate</th><th>latency ms</th><th>cpu %</th><th>verdict</th></tr></thead>
      <tbody></tbody>`;
    const tbody = table.querySelector("tbody");
    for (const r of results) {
      const m = r.metrics || {};
      const dep = latestByModel[r.name];
      const verdict = dep ? dep.verdict : null;
      const pareto = (data.pareto || []).includes(r.name);
      const tr = h("tr");
      tr.innerHTML = `
        <td>${esc(r.name)}${pareto ? ' <span class="pill" title="pareto frontier">★</span>' : ""}</td>
        <td class="mono">${esc(r.version)}</td>
        <td>${num(m.map50)}</td>
        <td>${num(m.map50_95)}</td>
        <td>${num(m.recall)}</td>
        <td>${num(m.false_positive_rate)}</td>
        <td>${num(m.latency_ms, 1) || "—"}</td>
        <td>${num(m.cpu_percent, 1) || "—"}</td>
        <td><span class="verdict ${(verdict || "none").toLowerCase()}">${esc(verdict || "—")}</span></td>`;
      tbody.appendChild(tr);
    }
    sec.appendChild(table);
  }

  sec.appendChild(tableCaption("Latency vs mAP50"));
  const points = results
    .map((r) => {
      const m = r.metrics || {};
      const x = m.latency_ms;
      const y = m.map50;
      if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
      return { x, y, name: r.name };
    })
    .filter((p) => p !== null);
  renderScatter(plotBox(sec), points);

  sec.appendChild(tableCaption("Deployments ledger"));
  const ledger = deps.deployments || [];
  if (!ledger.length) {
    sec.appendChild(h("div", "placeholder", "no deployment records yet"));
  } else {
    const table = h("table");
    table.innerHTML = `
      <thead><tr><th>model</th><th>version</th><th>verdict</th><th>reasons</th>
      <th>deployed</th><th>deployed at</th></tr></thead>
      <tbody></tbody>`;
    const tbody = table.querySelector("tbody");
    for (const d of ledger) {
      const tr = h("tr");
      const reasons = Array.isArray(d.reasons) && d.reasons.length
        ? `<ul class="reasons">${d.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>`
        : "—";
      tr.innerHTML = `
        <td>${esc(d.model_name)}</td>
        <td class="mono">${esc(d.version)}</td>
        <td><span class="verdict ${(d.verdict || "none").toLowerCase()}">${esc(d.verdict || "—")}</span></td>
        <td>${reasons}</td>
        <td>${d.deployed ? "yes" : "no"}</td>
        <td>${esc(fmtTs(d.deployed_at))}</td>`;
      tbody.appendChild(tr);
    }
    sec.appendChild(table);
  }
}

async function renderQuality(sec) {
  const [data, backlog] = await Promise.all([
    api("/api/quality"),
    api("/api/samples?verified=0&limit=20"),
  ]);
  sec.innerHTML = "";
  const cards = h("div", "cards");
  cards.appendChild(countCard("Samples", data.total, `with verified boxes: ${data.one_class_boxes}`));
  cards.appendChild(countCard("Verified", data.verified, `bad or duplicate: ${data.problematic}`));
  cards.appendChild(countCard("Unverified", data.unverified, "awaiting a verdict"));
  for (const q of QUALITIES) {
    cards.appendChild(countCard(QUALITY_LABELS[q] || q, data.by_quality[q] ?? 0,
      QUALITY_DESC[q] || "verdict count"));
  }
  sec.appendChild(cards);

  sec.appendChild(tableCaption("Verified boxes per class"));
  const entries = Object.entries(data.per_class || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    sec.appendChild(h("div", "placeholder", "no verified annotations yet"));
  } else {
    const max = Math.max(...entries.map((e) => e[1]), 1);
    const bars = h("div", "bars");
    for (const [label, count] of entries) {
      const row = h("div", "bar-row");
      row.appendChild(h("span", null, label));
      const track = h("div", "bar-track");
      const fill = h("div", "bar-fill");
      fill.style.width = (count / max * 100) + "%";
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(h("span", "bar-count", count));
      bars.appendChild(row);
    }
    sec.appendChild(bars);
  }

  sec.appendChild(tableCaption("Verdict backlog"));
  const samples = backlog.samples || [];
  if (!samples.length) {
    sec.appendChild(h("div", "placeholder", "no unverified samples awaiting a verdict"));
  } else {
    const table = h("table");
    table.innerHTML = `
      <thead><tr><th>id</th><th>camera</th><th>label</th><th>timestamp</th><th></th></tr></thead>
      <tbody></tbody>`;
    const tbody = table.querySelector("tbody");
    samples.forEach((s, i) => {
      const tr = h("tr");
      const link = h("a", null, "verdict");
      link.addEventListener("click", () => openLightbox(samples, i));
      tr.appendChild(h("td", "mono", s.id.slice(0, 12)));
      tr.appendChild(h("td", null, s.camera || "—"));
      tr.appendChild(h("td", null, s.label || "—"));
      tr.appendChild(h("td", null, fmtTs(s.timestamp)));
      const td = h("td");
      td.appendChild(link);
      tr.appendChild(td);
      tbody.appendChild(tr);
    });
    sec.appendChild(table);
  }
}

const auditState = {
  status: "", label: "", offset: 0, limit: 100,
  labels: [], detailId: null, updatedAt: null, viewSamples: [],
};

const AUDIT_CHECKS = [
  ["bbox_covers_object", "bbox covers object"],
];

async function renderAudit(sec) {
  const params = new URLSearchParams();
  if (auditState.status) params.set("status", auditState.status);
  if (auditState.label) params.set("label", auditState.label);
  params.set("limit", auditState.limit);
  params.set("offset", auditState.offset);
  const data = await api("/api/audit?" + params.toString());
  auditState.labels = data.labels || [];
  auditState.updatedAt = data.updated_at;
  auditState.viewSamples = data.samples || [];
  sec.innerHTML = "";
  const s = data.stats || {};
  const updated = fmtTs(data.updated_at) || "—";
  sec.appendChild(h("div", "status-line",
    `audit root ${esc(data.root)} · pipeline v${num(data.pipeline_version, 0)} · updated ${updated}`));
  if (!data.exists || !s.total) {
    sec.appendChild(h("div", "placeholder",
      "no audit output yet — run Audit from the Run pipeline panel on Overview, or `.venv/bin/frigate-learn audit run` to seed SAM + VLM reconciliation decisions"));
    return;
  }

  const cards = h("div", "cards");
  cards.appendChild(countCard("Samples", s.total, `processed ${num(s.processed, 0)}`));
  cards.appendChild(countCard("SAM ok", s.sam_ok, `failures ${num(s.sam_fail, 0)}`));
  cards.appendChild(countCard("VLM verdicts", s.vlm_calls, `failed ${num(s.vlm_fail, 0)} · pending ${num(s.vlm_transport, 0)}`));
  cards.appendChild(countCard("SAM + VLM agree", s.agreements, `disagreements ${num(s.disagreements, 0)}`));
  cards.appendChild(countCard("Accepted", s.accepted, "kept for training"));
  cards.appendChild(countCard("Dropped", s.dropped, "rejected by a stage"));
  cards.appendChild(countCard("Pending", s.pending, "VLM transport — retry with --resume"));
  cards.appendChild(countCard("Acceptance", (s.acceptance_rate * 100).toFixed(1) + "%",
    `positives ${num(s.positives_written, 0)} · hard negatives ${num(s.hard_negatives_written, 0)}`));
  sec.appendChild(cards);

  const banner = auditFailureBanner(s);
  if (banner) sec.appendChild(banner);

  sec.appendChild(tableCaption("Decision funnel · one object per row, width = share of candidates"));
  sec.appendChild(renderAuditFunnel(s));

  sec.appendChild(tableCaption("Per class"));
  sec.appendChild(renderAuditClassBars(s.by_class || {}));

  const bar = h("div", "filter-bar");
  const statusSel = filterSelect("Verdict", ["KEEP", "DROP", "PENDING"], auditState.status,
    (v) => { auditState.status = v; auditState.offset = 0; auditState.detailId = null; render("audit"); },
    { KEEP: "Accepted", DROP: "Dropped", PENDING: "Pending" });
  const labelSel = filterSelect("Label", auditState.labels.slice().sort(), auditState.label,
    (v) => { auditState.label = v; auditState.offset = 0; auditState.detailId = null; render("audit"); });
  bar.appendChild(statusSel);
  bar.appendChild(labelSel);
  bar.appendChild(h("span", "status-line", `${data.total} samples`));
  const prev = h("button", "btn", "← prev");
  prev.disabled = auditState.offset <= 0;
  prev.addEventListener("click", () => {
    auditState.offset = Math.max(0, auditState.offset - auditState.limit);
    render("audit");
  });
  const next = h("button", "btn", "next →");
  next.disabled = auditState.offset + (data.samples || []).length >= data.total;
  next.addEventListener("click", () => {
    auditState.offset += auditState.limit;
    render("audit");
  });
  bar.appendChild(prev);
  bar.appendChild(next);
  sec.appendChild(bar);

  if (auditState.detailId) {
    await renderAuditDetail(sec, null, auditState.detailId);
  }

  sec.appendChild(tableCaption("Samples · SAM + VLM reconciliation"));
  const samples = data.samples || [];
  if (!samples.length) {
    sec.appendChild(h("div", "placeholder", "no audit samples match the current filters"));
  } else {
    const table = h("table");
    table.innerHTML = `
      <thead><tr><th>sample</th><th>camera</th><th>Frigate</th><th>SAM</th><th>verdict</th><th>reason</th>
      <th>failing</th><th>artifacts</th><th></th></tr></thead>
      <tbody></tbody>`;
    const tbody = table.querySelector("tbody");
    for (const smp of samples) {
      const tr = h("tr");
      const failTxt = (smp.failing_conditions || []).join(", ") || "—";
      const artifacts = [];
      if (smp.has_reconciliation) artifacts.push("recon");
      if (smp.has_mask) artifacts.push("mask");
      tr.innerHTML = `
        <td class="mono">${esc(smp.sample_id.slice(0, 12))}</td>
        <td>${esc((smp.source_extra && smp.source_extra.camera) || "—")}</td>
        <td>${esc(smp.frigate_label || "—")}</td>
        <td>${esc(smp.sam_class || "—")}</td>
        <td>${pill(smp.status || "—", auditPillClass(smp.status))}${smp.override ? " *" : ""}</td>
        <td>${esc(smp.reason || "—")}</td>
        <td>${esc(failTxt)}</td>
        <td>${esc(artifacts.join(" · ") || "—")}</td>`;
      const td = h("td");
      const btn = h("button", "btn", smp.sample_id === auditState.detailId ? "hide" : "detail");
      btn.dataset.sample = smp.sample_id;
      btn.addEventListener("click", () => {
        auditState.detailId = smp.sample_id === auditState.detailId ? null : smp.sample_id;
        refreshAuditButtons(null);
        if (!auditState.detailId) {
          const p = document.querySelector("#view-audit .detail-panel");
          if (p) p.remove();
          return;
        }
        renderAuditDetail(section("audit"), null, auditState.detailId).then(() => {
          const p = document.querySelector("#view-audit .detail-panel");
          if (p) p.scrollIntoView({ block: "nearest" });
        });
      });
      td.appendChild(btn);
      tr.appendChild(td);
      tbody.appendChild(tr);
    }
    sec.appendChild(table);
    refreshAuditButtons(table);
  }
}

function refreshAuditButtons(table) {
  const scope = table || document.querySelector("#view-audit");
  if (!scope) return;
  for (const btn of scope.querySelectorAll("button[data-sample]")) {
    btn.textContent = btn.dataset.sample === auditState.detailId ? "hide" : "detail";
  }
}

function auditPillClass(status) {
  if (status === "KEEP") return "pass";
  if (status === "DROP") return "fail";
  if (status === "PENDING") return "running";
  return "";
}

function auditFailureBanner(s) {
  if (!s.total) return null;
  if (s.sam_ok === 0 && s.sam_fail > 0) {
    return h("div", "audit-banner",
      `SAM 3.1 produced no segmentations (${num(s.sam_fail, 0)}/${num(s.total, 0)} sam_failure). ` +
      `The sam3-mlx extra is not installed or the checkpoint failed, so no SAM mask/box, reconciliation image, ` +
      `or VLM verdict exists for any sample. Install it with .venv/bin/pip install -e ".[audit]" ` +
      `and re-run .venv/bin/frigate-learn audit run --resume`);
  }
  if (s.vlm_transport > 0 && s.pending > 0) {
    return h("div", "audit-banner",
      `${num(s.pending, 0)} samples are PENDING (VLM transport failure). Retry them with ` +
      `.venv/bin/frigate-learn audit run --resume once the audit VLM endpoint is reachable`);
  }
  return null;
}

function renderAuditFunnel(s) {
  const stages = [
    ["all candidates", s.total, "var(--gray)"],
    ["SAM segment ok", s.sam_ok, "var(--blue)"],
    ["VLM verdict parsed", s.vlm_calls + s.vlm_fail, "var(--amber)"],
    ["KEEP (accepted)", s.accepted, "var(--lime)"],
  ];
  const wrap = h("div", "funnel");
  const max = Math.max(s.total, 1);
  for (const [label, count, color] of stages) {
    const row = h("div", "funnel-row");
    row.appendChild(h("span", "funnel-label", label));
    const track = h("div", "funnel-track");
    const fill = h("div", "funnel-bar");
    fill.style.width = (count / max * 100).toFixed(1) + "%";
    fill.style.background = color;
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(h("span", "funnel-count", `${num(count, 0)} · ${(count / max * 100).toFixed(1)}%`));
    wrap.appendChild(row);
  }
  wrap.appendChild(h("div", "status-line",
    "grey = all candidates · blue = SAM produced a mask+box · amber = VLM answered · green = conservative rule kept it"));
  return wrap;
}

function renderAuditClassBars(byClass) {
  const entries = Object.entries(byClass).sort((a, b) => b[1].processed - a[1].processed);
  if (!entries.length) {
    return h("div", "placeholder", "no per-class data yet");
  }
  const wrap = h("div", "bars");
  for (const [label, cs] of entries) {
    const row = h("div", "bar-row");
    row.appendChild(h("span", null, label + ` (${num(cs.processed, 0)})`));
    const track = h("div", "bar-track audit-stack");
    const total = Math.max(cs.processed, 1);
    const acc = h("div", "bar-fill-frag accept");
    acc.style.width = (cs.accepted / total * 100) + "%";
    const drp = h("div", "bar-fill-frag drop");
    drp.style.width = (cs.dropped / total * 100) + "%";
    const pen = h("div", "bar-fill-frag pending");
    pen.style.width = (cs.pending / total * 100) + "%";
    track.appendChild(acc);
    track.appendChild(drp);
    track.appendChild(pen);
    row.appendChild(track);
    row.appendChild(h("span", "bar-count",
      `✓${num(cs.accepted, 0)} ✗${num(cs.dropped, 0)} ?${num(cs.pending, 0)}`));
    wrap.appendChild(row);
  }
  return wrap;
}

function auditNav(delta) {
  const list = auditState.viewSamples || [];
  if (!auditState.detailId || list.length < 2) return;
  const i = list.findIndex((x) => x.sample_id === auditState.detailId);
  if (i === -1) return;
  auditState.detailId = list[(i + delta + list.length) % list.length].sample_id;
  renderAuditDetail(section("audit"), null, auditState.detailId);
  refreshAuditButtons(null);
}

async function setAuditOverride(sampleId, status, noteInput) {
  try {
    await api("/api/audit/overrides/" + encodeURIComponent(sampleId), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, note: noteInput.value }),
    });
    render("audit");
  } catch (e) {
    errorBox(section("audit").querySelector(".detail-override"), e.message);
  }
}

async function renderAuditDetail(sec, table, sampleId) {
  const seq = (auditState.detailSeq = (auditState.detailSeq || 0) + 1);
  if (!sampleId) {
    const prev = sec.querySelector(".detail-panel");
    if (prev) prev.remove();
    return;
  }
  const det = await api("/api/audit/" + encodeURIComponent(sampleId));
  if (seq !== auditState.detailSeq) return;
  const panel = h("div", "detail-panel");
  const navList = auditState.viewSamples || [];
  const navIdx = navList.findIndex((x) => x.sample_id === sampleId);
  const nav = h("div", "detail-nav");
  const prevBtn = h("button", "btn", "← prev");
  prevBtn.disabled = navList.length < 2;
  prevBtn.addEventListener("click", () => auditNav(-1));
  const nextBtn = h("button", "btn", "next →");
  nextBtn.disabled = navList.length < 2;
  nextBtn.addEventListener("click", () => auditNav(1));
  const navCam = navIdx >= 0 && navList[navIdx].source_extra ? navList[navIdx].source_extra.camera : null;
  const navBtns = h("div", "detail-nav-btns");
  navBtns.appendChild(prevBtn);
  navBtns.appendChild(nextBtn);
  const closeNav = h("button", "btn", "close");
  closeNav.addEventListener("click", () => {
    auditState.detailId = null;
    const p = document.querySelector("#view-audit .detail-panel");
    if (p) p.remove();
    refreshAuditButtons(null);
  });
  navBtns.appendChild(closeNav);
  nav.appendChild(navBtns);
  nav.appendChild(h("span", "detail-nav-info",
    navIdx >= 0 ? `sample ${navIdx + 1} of ${navList.length}${navCam ? " · " + esc(navCam) : ""}` : ""));
  panel.appendChild(nav);

  const d = det.decision || {};
  const prov = det.provenance || {};
  const source = prov.source || {};
  const sam = prov.sam || {};
  const geometry = prov.geometry || {};
  const vlm = prov.vlm || {};
  const samFailed = d.reason === "sam_failure";

  const head = h("div", "detail-head");
  head.appendChild(h("h2", null, `audit ${esc(sampleId)}`));
  head.appendChild(pillNode(d.status || "—", auditPillClass(d.status)));
  head.appendChild(pillNode(d.reason || "ok", "ignored"));
  for (const cond of d.failing_conditions || []) {
    head.appendChild(pillNode("✗ " + cond, "fail"));
  }
  if (det.override) head.appendChild(pillNode("override → " + det.override.status, "warn"));
  panel.appendChild(head);

  const failing = d.failing_conditions || [];
  const labelsWrap = h("div", "detail-labels");
  labelsWrap.appendChild(labelBadge("frigate", source.class_name || "—"));
  labelsWrap.appendChild(labelBadge("sam", samFailed ? "failed" : (sam.class_name || "—"), samFailed ? "fail" : ""));
  if (det.vlm_error) {
    labelsWrap.appendChild(labelBadge("vlm", "error", "fail"));
  } else if (!det.has_vlm) {
    labelsWrap.appendChild(labelBadge("vlm", "not reached", ""));
  } else {
    const vlmLabel = vlm.class_label || "";
    labelsWrap.appendChild(labelBadge("vlm", vlmLabel ? vlmLabel : "no object", vlmLabel ? "ok" : "warn"));
  }
  const pendingDec = !d.status || String(d.status) === "PENDING";
  const agreeOk = !pendingDec && !failing.includes("class_mismatch");
  labelsWrap.appendChild(labelBadge(
    "sam ❯ vlm",
    pendingDec ? "pending" : (agreeOk ? "agree" : "mismatch"),
    pendingDec ? "" : (agreeOk ? "ok" : "warn"),
  ));
  panel.appendChild(labelsWrap);

  const imgs = h("div", "detail-imgs");
  if (det.artifacts && det.artifacts.reconciliation) {
    const fig = h("div", "audit-figure");
    const img = document.createElement("img");
    img.src = det.artifacts.reconciliation;
    img.alt = "reconciliation";
    fig.appendChild(img);
    fig.appendChild(h("div", "audit-caption",
      "reconciliation overlay — green = Frigate box, blue = SAM box, red = SAM mask."));
    imgs.appendChild(fig);
  } else {
    imgs.appendChild(h("div", "placeholder", "no reconciliation image (SAM failed before rendering)"));
  }
  if (det.artifacts && det.artifacts.vlm) {
    const fig = h("div", "audit-figure");
    const img = document.createElement("img");
    img.src = det.artifacts.vlm;
    img.alt = "VLM input";
    fig.appendChild(img);
    fig.appendChild(h("div", "audit-caption",
      "blind VLM input — exactly what the VLM saw: the crop with only the Frigate box, no labels."));
    imgs.appendChild(fig);
  }
  if (det.artifacts && det.artifacts.mask) {
    const fig = h("div", "audit-figure");
    const img = document.createElement("img");
    img.src = det.artifacts.mask;
    img.alt = "SAM mask";
    fig.appendChild(img);
    fig.appendChild(h("div", "audit-caption", "SAM binary mask (crop space)"));
    imgs.appendChild(fig);
  }
  panel.appendChild(imgs);

  const ovr = det.override || null;
  const ovrSection = h("div", "detail-override");
  ovrSection.appendChild(h("h2", "card-title", "Human override"));
  ovrSection.appendChild(h("div", "status-line", ovr
    ? `effective ${esc(ovr.status)} · ${esc(ovr.note || "no note")}${d.pipeline_status ? " · pipeline was " + esc(d.pipeline_status) : ""}`
    : `pipeline verdict ${esc(d.status || "—")}${d.reason ? " (" + esc(d.reason) + ")" : ""}`));
  const noteInput = document.createElement("input");
  noteInput.className = "override-note";
  noteInput.placeholder = "note (optional)";
  if (ovr && ovr.note) noteInput.value = ovr.note;
  const ovrBtns = h("div", "detail-nav-btns");
  const keepBtn = h("button", "btn", "force KEEP");
  keepBtn.addEventListener("click", () => setAuditOverride(sampleId, "KEEP", noteInput));
  const dropBtn = h("button", "btn", "force DROP");
  dropBtn.addEventListener("click", () => setAuditOverride(sampleId, "DROP", noteInput));
  ovrBtns.appendChild(keepBtn);
  ovrBtns.appendChild(dropBtn);
  if (ovr) {
    const clearBtn = h("button", "btn", "remove override");
    clearBtn.addEventListener("click", () => {
      api("/api/audit/overrides/" + encodeURIComponent(sampleId), { method: "DELETE" })
        .then(() => render("audit"))
        .catch((e) => errorBox(ovrSection, e.message));
    });
    ovrBtns.appendChild(clearBtn);
  }
  ovrSection.appendChild(noteInput);
  ovrSection.appendChild(ovrBtns);
  panel.appendChild(ovrSection);

  const sourceMeta = [
    ["sample id", sampleId],
    ["hash", (prov.sample_hash || "").slice(0, 12)],
    ["class", `${source.class_name || "—"}${source.class_id !== null && source.class_id !== undefined ? " · id " + source.class_id : ""}`],
    ["source box", Array.isArray(source.bbox) ? source.bbox.map((v) => num(v)).join(", ") : "—"],
    ...Object.entries(d.source_extra || {}).map(([k, v]) => [k, typeof v === "object" ? "…" : String(v)]),
  ];
  const meta = det.meta || {};
  panel.appendChild(renderKvSection("Source", sourceMeta));
  panel.appendChild(renderKvSection("Models", [
    ["sam key", meta.sam_key || "—"],
    ["vlm key", meta.vlm_key || "—"],
    ["pipeline version", num(meta.pipeline_version, 0) || "—"],
  ]));
  panel.appendChild(renderKvSection("SAM result", samFailed ? [
    ["stage", "failed (sam_failure)"],
    ["result", "no mask, no box — sample dropped before geometry/VLM"],
  ] : [
    ["hypothesis", sam.class_name || "—"],
    ["confidence", num(sam.confidence, 3)],
    ["refined box", Array.isArray(sam.bbox) ? sam.bbox.map((v) => num(v)).join(", ") : "—"],
    ["mask area (px)", num(sam.mask_area, 0)],
  ]));
  panel.appendChild(renderKvSection("Geometry", [
    ["bbox IoU", num(geometry.bbox_iou, 3)],
    ["mask / box ratio", num(geometry.mask_bbox_ratio, 3)],
    ["mask in Frigate box", num(geometry.mask_frigate_containment, 3)],
    ["mask in SAM box", num(geometry.mask_sam_containment, 3)],
    ["Frigate box area", num(geometry.frigate_bbox_area, 0)],
    ["SAM box area", num(geometry.sam_bbox_area, 0)],
    ["edge deltas L/T/R/B",
      ["left_delta", "top_delta", "right_delta", "bottom_delta"]
        .map((k) => num(geometry.edge_deltas && geometry.edge_deltas[k], 1))
        .join(" / ") || "—"],
  ]));

  const checkBox = h("div", "audit-check-list");
  if (det.vlm_error) {
    checkBox.appendChild(h("div", "audit-error", "VLM error: " + det.vlm_error));
  } else if (!det.has_vlm) {
    checkBox.appendChild(h("div", "placeholder",
      "VLM never reached — an earlier stage (SAM) failed, so no independent verdict exists"));
  } else {
    const failing = d.failing_conditions || [];
    const presentOk = !failing.includes("object_present");
    const presentRow = h("div", "audit-check " + (presentOk ? "check-ok" : "check-fail"));
    presentRow.appendChild(h("span", null, "object present" + (vlm.class_label ? " · VLM sees " + esc(vlm.class_label) : "")));
    presentRow.appendChild(pillNode(presentOk ? "✓ pass" : "✗ fail", presentOk ? "pass" : "fail"));
    checkBox.appendChild(presentRow);
    for (const [key, label] of AUDIT_CHECKS) {
      const val = vlm[key];
      const row = h("div", "audit-check " + (val ? "check-ok" : "check-fail"));
      row.appendChild(h("span", null, label));
      row.appendChild(pillNode(val ? "✓ pass" : "✗ fail", val ? "pass" : "fail"));
      checkBox.appendChild(row);
    }
    const agreeOk = !failing.includes("class_mismatch");
    const agreeRow = h("div", "audit-check " + (agreeOk ? "check-ok" : "check-fail"));
    agreeRow.appendChild(h("span", null, "SAM + VLM class agree (independent)"));
    agreeRow.appendChild(pillNode(agreeOk ? "✓ pass" : "✗ fail", agreeOk ? "pass" : "fail"));
    checkBox.appendChild(agreeRow);
  }
  const decisionBox = renderKvSection("Decision", [
    ["verdict", d.status || "—"],
    ["reason", d.reason || "none (all stages agreed)"],
    ["training label", d.training_label || "—"],
    ["bbox source", d.bbox_source || "—"],
    ["mask source", d.mask_source || "—"],
    ["failing conditions", (d.failing_conditions || []).join(", ") || "none"],
  ]);
  const right = h("div", "detail-right");
  right.appendChild(h("h2", "card-title", "VLM independent verdict"));
  right.appendChild(checkBox);
  right.appendChild(decisionBox);
  panel.appendChild(right);
  const prevPanel = sec.querySelector(".detail-panel");
  if (prevPanel) prevPanel.remove();
  const tableEl = sec.querySelector("table");
  if (tableEl) sec.insertBefore(panel, tableEl);
  else sec.appendChild(panel);
}

function renderKvSection(title, pairs) {
  const wrap = h("div", "kv-section");
  wrap.appendChild(h("h3", "card-title", title));
  const grid = h("div", "kv-grid");
  for (const [k, v] of pairs) {
    grid.appendChild(h("div", "k", k));
    grid.appendChild(h("div", "v", v === null || v === undefined || v === "" ? "—" : String(v)));
  }
  wrap.appendChild(grid);
  return wrap;
}

async function renderDatasets(sec) {
  const data = await api("/api/datasets");
  sec.innerHTML = "";
  const versions = data.versions || [];
  sec.appendChild(tableCaption("Dataset versions"));
  if (!versions.length) {
    sec.appendChild(h("div", "placeholder",
      "no dataset versions built yet — run collect → verify → build from the Overview pipeline controls"));
  } else {
    const table = h("table");
    table.innerHTML = `
      <thead><tr><th>version</th><th>built at</th><th>written</th><th>total</th>
      <th>verified</th><th>skipped class</th><th>train</th><th>val</th></tr></thead>
      <tbody></tbody>`;
    const tbody = table.querySelector("tbody");
    for (const v of versions) {
      const tr = h("tr");
      if (v.error) {
        tr.innerHTML = `<td class="mono">${esc(v.version)}</td>` +
          `<td colspan="7" style="color:var(--red)">${esc(v.error)}</td>`;
        tbody.appendChild(tr);
        continue;
      }
      const split = v.split_counts || {};
      tr.innerHTML = `
        <td class="mono">${esc(v.version)}</td>
        <td>${esc(fmtTs(v.built_at))}</td>
        <td>${esc(num(v.images_written, 0))}</td>
        <td>${esc(num(v.total, 0))}</td>
        <td>${esc(num(v.verified, 0))}</td>
        <td>${esc(num(v.skipped_class, 0))}</td>
        <td>${esc(num(split.train, 0))}</td>
        <td>${esc(num(split.val, 0))}</td>`;
      tbody.appendChild(tr);
    }
    sec.appendChild(table);
  }

  sec.appendChild(tableCaption("Golden dataset"));
  const g = data.golden || {};
  const card = h("div", "card");
  card.appendChild(h("div", "status-line", `${g.version || "—"} · ${g.exists ? "valid" : "missing"}`));
  if (g.exists) {
    card.appendChild(h("div", "sub",
      `classes: ${(g.classes || []).join(", ") || "—"} · images: ${g.image_count}`));
    card.appendChild(h("div", "sub",
      "per-class: " + (Object.entries(g.per_class || {})
        .map(([k, v]) => `${k}=${v}`).join(", ") || "—")));
  }
  sec.appendChild(card);
}

const trainingState = { run: "", showTable: false };

async function renderTraining(sec) {
  const listData = await api("/api/training/list");
  sec.innerHTML = "";
  const runs = listData.runs || [];
  sec.appendChild(tableCaption("Training runs"));
  if (!runs.length) {
    sec.appendChild(h("div", "placeholder",
      "no training runs found — run the train stage from the Overview pipeline controls"));
    return;
  }
  if (!trainingState.run || !runs.some((r) => r.run === trainingState.run)) {
    trainingState.run = runs[0].run;
  }
  const bar = h("div", "filter-bar");
  bar.appendChild(h("span", null, "run"));
  const select = h("select");
  for (const r of runs) {
    const opt = h("option", null, `${r.run} (best ${num(r.best_map50)})`);
    opt.value = r.run;
    select.appendChild(opt);
  }
  select.value = trainingState.run;
  const tableBtn = h("button", "btn", trainingState.showTable ? "hide table" : "show table");
  tableBtn.addEventListener("click", () => {
    trainingState.showTable = !trainingState.showTable;
    render("training");
  });
  select.addEventListener("change", () => {
    trainingState.run = select.value;
    render("training");
  });
  bar.appendChild(select);
  bar.appendChild(tableBtn);
  sec.appendChild(bar);

  const detail = await renderTrainingDetail(sec, trainingState.run);
  if (detail && trainingState.showTable) {
    const table = h("table");
    const thead = h("thead");
    const headRow = h("tr");
    for (const c of detail.columns) headRow.appendChild(h("th", null, c));
    thead.appendChild(headRow);
    const tbody = h("tbody");
    for (const row of detail.rows) {
      const tr = h("tr");
      for (const c of detail.columns) {
        const v = row[c];
        tr.appendChild(h("td", null, v === null || v === undefined ? "" : String(v)));
      }
      tbody.appendChild(tr);
    }
    table.appendChild(thead);
    table.appendChild(tbody);
    sec.appendChild(table);
  }
}

async function renderTrainingDetail(sec, run) {
  let data;
  try {
    data = await api("/api/training/" + encodeURIComponent(run));
  } catch (e) {
    errorBox(sec, e.message);
    return null;
  }
  const cards = h("div", "cards");
  cards.appendChild(countCard("Best mAP50", num(data.best_map50) || "—", `dataset ${data.dataset}`));
  cards.appendChild(countCard("Latest mAP50", num(data.latest_map50) || "—", `${data.epochs} epochs`));
  cards.appendChild(countCard("Weights", data.weights_exists ? "present" : "absent", "weights/best.pt"));
  const ba = data.before_after;
  const beforeMap50 = ba && ba.before ? ba.before.map50 : null;
  const afterMap50 = ba && ba.after ? ba.after.map50 : null;
  cards.appendChild(countCard("Golden before", num(beforeMap50) || "—", "pretrained · golden"));
  cards.appendChild(countCard("Golden after", num(afterMap50) || "—", "fine-tuned · golden"));
  if (beforeMap50 !== null && afterMap50 !== null) {
    const delta = afterMap50 - beforeMap50;
    cards.appendChild(countCard("Golden Δ", (delta >= 0 ? "+" : "") + num(delta), "after − before"));
  }
  sec.appendChild(cards);

  sec.appendChild(tableCaption("Epoch curves · val set per epoch · dashed = golden before/after"));
  const xs = [];
  const map = [];
  const rec = [];
  for (const row of data.rows || []) {
    const e = Number(row.epoch);
    if (!Number.isFinite(e)) continue;
    xs.push(e);
    const m = row[MAP50_KEY];
    const r = row[RECALL_KEY];
    map.push((m === null || m === undefined || m === "" || !Number.isFinite(Number(m))) ? null : Number(m));
    rec.push((r === null || r === undefined || r === "" || !Number.isFinite(Number(r))) ? null : Number(r));
  }
  const series = [
    { label: "mAP50", color: "var(--blue)", data: map },
    { label: "recall", color: "var(--lime)", data: rec },
  ];
  if (beforeMap50 !== null) series.push({ label: "golden before", color: "var(--muted)", width: 1, dash: [6, 4], data: xs.map(() => beforeMap50) });
  if (afterMap50 !== null) series.push({ label: "golden after", color: "var(--amber)", width: 1, dash: [6, 4], data: xs.map(() => afterMap50) });
  renderLineChart(plotBox(sec), xs, series);
  return data;
}

const triageState = {
  quality: "", status: "", camera: "", verified: false,
  samples: [], statuses: new Set(), cameras: [], seeded: false, list: [], idx: 0,
};

async function renderTriage(sec) {
  if (!triageState.seeded) {
    try {
      const ov = await api("/api/overview");
      triageState.cameras = ov.cameras || [];
      const seed = await api("/api/samples?limit=200");
      for (const s of seed.samples || []) {
        if (s.status) triageState.statuses.add(s.status);
      }
      triageState.seeded = true;
    } catch (e) {}
  }
  sec.innerHTML = "";
  const bar = h("div", "filter-bar");
  bar.appendChild(filterSelect("Verdict", QUALITIES, triageState.quality,
    (v) => { triageState.quality = v; loadTriageGrid(); }, QUALITY_LABELS));
  bar.appendChild(filterSelect("Status", [...triageState.statuses].sort(), triageState.status,
    (v) => { triageState.status = v; loadTriageGrid(); }));
  bar.appendChild(filterSelect("Camera", triageState.cameras.slice().sort(), triageState.camera,
    (v) => { triageState.camera = v; loadTriageGrid(); }));
  const vbox = h("label", null);
  const vcheck = document.createElement("input");
  vcheck.type = "checkbox";
  vcheck.checked = triageState.verified;
  vcheck.addEventListener("change", () => {
    triageState.verified = vcheck.checked;
    loadTriageGrid();
  });
  vbox.appendChild(vcheck);
  vbox.appendChild(document.createTextNode(" verified only"));
  bar.appendChild(vbox);
  sec.appendChild(bar);
  sec.appendChild(h("div", "status-line",
    "click a thumbnail to open the lightbox — set a verdict, or navigate with ←/→ and close with Esc"));
  triageState.grid = h("div", "thumb-grid");
  sec.appendChild(triageState.grid);
  await loadTriageGrid();
}

function filterSelect(label, options, current, onChange, display) {
  const lab = h("span", null, label);
  const select = h("select");
  const empty = h("option", null, "All");
  empty.value = "";
  select.appendChild(empty);
  for (const o of options) {
    const opt = h("option", null, display ? (display[o] || o) : o);
    opt.value = o;
    select.appendChild(opt);
  }
  select.value = current || "";
  select.addEventListener("change", () => onChange(select.value));
  const wrap = h("div", "filter-bar");
  wrap.style.margin = "0";
  wrap.appendChild(lab);
  wrap.appendChild(select);
  return wrap;
}

function samplesQuery() {
  const p = new URLSearchParams();
  if (triageState.quality) p.set("quality", triageState.quality);
  if (triageState.status) p.set("status", triageState.status);
  if (triageState.camera) p.set("camera", triageState.camera);
  if (triageState.verified) p.set("verified", 1);
  p.set("limit", 200);
  return "/api/samples?" + p.toString();
}

async function loadTriageGrid() {
  if (!triageState.grid) return;
  triageState.grid.innerHTML = "";
  triageState.grid.appendChild(h("div", "status-line", "loading…"));
  try {
    const data = await api(samplesQuery());
    triageState.samples = data.samples || [];
    triageState.grid.innerHTML = "";
    triageState.grid.appendChild(h("div", "status-line",
      `${data.total} samples, showing ${triageState.samples.length}`));
    if (!triageState.samples.length) {
      triageState.grid.appendChild(h("div", "placeholder", "no samples match"));
      return;
    }
    triageState.samples.forEach((s, i) => {
      const thumb = h("div", "thumb");
      if (s.has_image && s.thumb_url) {
        const img = document.createElement("img");
        img.loading = "lazy";
        img.src = s.thumb_url;
        img.alt = s.id;
        thumb.appendChild(img);
      } else {
        thumb.appendChild(h("div", "noimg", "no image"));
      }
      if (s.verified) {
        const b = h("div", "badge-verify", "✓");
        b.title = "verified";
        thumb.appendChild(b);
      }
      const meta = h("div", "meta");
      meta.appendChild(document.createTextNode(
        `${s.camera || "—"} · ${s.label || "—"} · ${num(s.score, 2)} · `));
      meta.appendChild(document.createTextNode(fmtTs(s.timestamp)));
      meta.appendChild(document.createTextNode(" "));
      meta.appendChild(pillNode(s.quality || "unset", s.quality,
        s.quality ? QUALITY_DESC[s.quality] : "no verdict yet"));
      thumb.appendChild(meta);
      thumb.addEventListener("click", () => openLightbox(triageState.samples, i));
      triageState.grid.appendChild(thumb);
    });
  } catch (e) {
    triageState.grid.innerHTML = "";
    triageState.grid.appendChild(h("div", "placeholder", e.message));
  }
}

async function openLightbox(list, idx) {
  triageState.list = list;
  triageState.idx = Math.max(0, Math.min(idx, list.length - 1));
  $("#lightbox").hidden = false;
  await renderLightbox();
}

async function renderLightbox() {
  const lb = $("#lightbox");
  const item = triageState.list[triageState.idx];
  if (!item) {
    closeLightbox();
    return;
  }
  lb.innerHTML = "";
  const panel = h("div", "lightbox-panel");
  const stage = h("div", "lightbox-stage");
  const img = document.createElement("img");
  img.alt = item.id;
  img.addEventListener("error", () => {
    stage.appendChild(h("div", "placeholder lb-note", "image not found"));
  });
  img.src = item.image_url || `/images/${item.id}`;
  stage.appendChild(img);
  const overlay = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  overlay.setAttribute("viewBox", "0 0 1 1");
  overlay.setAttribute("preserveAspectRatio", "none");
  overlay.setAttribute("class", "overlay");
  stage.appendChild(overlay);

  const meta = h("div", "lightbox-meta");
  meta.appendChild(h("span", "mono", item.id.slice(0, 12)));
  meta.appendChild(h("span", null, item.camera || "—"));
  meta.appendChild(h("span", null, item.label || "—"));
  meta.appendChild(h("span", null, fmtTs(item.timestamp)));
  meta.appendChild(pillNode(item.quality || "unset", item.quality,
    item.quality ? QUALITY_DESC[item.quality] : "no verdict yet"));

  const qbar = h("div", "lightbox-qual");
  for (const q of QUALITIES) {
    const btn = h("button", null, QUALITY_LABELS[q] || q);
    btn.title = QUALITY_DESC[q] || q;
    btn.dataset.quality = q;
    btn.dataset.action = "quality";
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      try {
        await api(`/api/samples/${encodeURIComponent(item.id)}/quality`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ quality: q }),
        });
        views.quality.stale = true;
        await openLightbox(triageState.list, triageState.idx);
        loadTriageGrid();
      } catch (e) {
        btn.disabled = false;
        qbar.appendChild(h("span", "status-line", e.message));
      }
    });
    qbar.appendChild(btn);
  }

  const nav = h("div", "lightbox-qual");
  const prev = h("button", null, "← prev");
  prev.disabled = triageState.idx <= 0;
  prev.addEventListener("click", () => {
    if (triageState.idx > 0) { triageState.idx -= 1; renderLightbox(); }
  });
  const next = h("button", null, "next →");
  next.disabled = triageState.idx >= triageState.list.length - 1;
  next.addEventListener("click", () => {
    if (triageState.idx < triageState.list.length - 1) { triageState.idx += 1; renderLightbox(); }
  });
  const close = h("button", null, "close");
  close.addEventListener("click", closeLightbox);
  nav.appendChild(prev);
  nav.appendChild(next);
  nav.appendChild(close);

  panel.appendChild(stage);
  panel.appendChild(meta);
  panel.appendChild(qbar);
  panel.appendChild(nav);

  try {
    const detail = await api("/api/samples/" + encodeURIComponent(item.id));
    drawBoxes(overlay, detail);
    const first = panel.querySelector(".lightbox-meta .pill");
    if (first) first.remove();
    panel.querySelector(".lightbox-meta").appendChild(
      pillNode(detail.sample.quality || "unset", detail.sample.quality,
        detail.sample.quality ? QUALITY_DESC[detail.sample.quality] : "no verdict yet"));
  } catch (e) {
    stage.appendChild(h("div", "placeholder lb-note", e.message));
  }
  lb.appendChild(panel);
}

function drawBoxes(overlay, detail) {
  const ns = "http://www.w3.org/2000/svg";
  function boxOf(b) {
    return {
      x: Math.min(b[0], b[2]),
      y: Math.min(b[1], b[3]),
      w: Math.max(0, Math.abs(b[2] - b[0])),
      ht: Math.max(0, Math.abs(b[3] - b[1])),
    };
  }
  function rect(b, color) {
    const r = document.createElementNS(ns, "rect");
    r.setAttribute("x", b.x);
    r.setAttribute("y", b.y);
    r.setAttribute("width", b.w);
    r.setAttribute("height", b.ht);
    r.setAttribute("fill", "none");
    r.setAttribute("stroke", color);
    r.setAttribute("stroke-width", "0.004");
    r.setAttribute("vector-effect", "non-scaling-stroke");
    overlay.appendChild(r);
  }
  function label(b, text, color) {
    if (!text) return;
    const t = document.createElementNS(ns, "text");
    t.setAttribute("x", b.x);
    t.setAttribute("y", Math.max(0.03, b.y - 0.01));
    t.setAttribute("font-size", "0.035");
    t.setAttribute("fill", color);
    t.setAttribute("style",
      "paint-order:stroke; stroke:#000; stroke-width:0.006; stroke-linejoin:round");
    t.textContent = text;
    overlay.appendChild(t);
  }
  for (const a of detail.annotations || []) {
    if (!Array.isArray(a.box) || a.box.length < 4) continue;
    const b = boxOf(a.box);
    if (!b.w && !b.ht) continue;
    rect(b, "#56c7ff");
    const conf = num(a.confidence, 2);
    label(b, `${a.label}${conf ? " " + conf : ""}`, "#56c7ff");
  }
  const f = detail.frigate;
  if (f && Array.isArray(f.box) && f.box.length >= 4) {
    const b = boxOf(f.box);
    if (!b.w && !b.ht) return;
    rect(b, "#ff7b3d");
    const conf = num(f.score, 2);
    label(b, `${f.label}${conf ? " " + conf : ""}`, "#ff7b3d");
  }
}

function closeLightbox() {
  const lb = $("#lightbox");
  lb.hidden = true;
  lb.innerHTML = "";
}

document.addEventListener("keydown", (e) => {
  const lb = $("#lightbox");
  const dr = $("#drawer");
  if (e.key === "Escape" && !dr.hidden) {
    closeDrawer();
    return;
  }
  if (lb.hidden) return;
  if (e.key === "Escape") {
    closeLightbox();
  } else if (e.key === "ArrowLeft" && triageState.idx > 0) {
    triageState.idx -= 1;
    renderLightbox();
  } else if (e.key === "ArrowRight" && triageState.idx < triageState.list.length - 1) {
    triageState.idx += 1;
    renderLightbox();
  }
});

let hasRunning = false;
let jobPolling = false;

function renderBanner(runningJob) {
  const banner = $("#job-banner");
  banner.onclick = null;
  document.body.classList.toggle("running", !!runningJob);
  if (!runningJob) {
    banner.hidden = true;
    banner.innerHTML = "";
    return;
  }
  banner.hidden = false;
  banner.innerHTML = "";
  const body = h("span", "banner-body");
  body.appendChild(h("span", "spinner"));
  body.appendChild(h("span", null,
    `job ${runningJob.id.slice(0, 8)} · ${jobKindLabel(runningJob.type).toLowerCase()} running — click for live log`));
  banner.appendChild(body);
  banner.onclick = () => openJobDrawer(runningJob.id);
  document.documentElement.style.setProperty("--banner-h", banner.offsetHeight + "px");
}

async function pollJobs() {
  if (jobPolling) return;
  jobPolling = true;
  try {
    const jobs = await api("/api/jobs");
    const runningJob = jobs.find((j) => j.status === "running") || null;
    const running = !!runningJob;
    const becameIdle = hasRunning && !running;
    hasRunning = running;
    renderBanner(runningJob);
    if (running && currentView === "overview" && views.overview.loaded) {
      render("overview");
    }
    if (currentView === "audit" && views.audit.loaded) {
      try {
        const probe = await api("/api/audit?limit=1");
        if (probe.updated_at !== auditState.updatedAt) render("audit");
      } catch (e) {}
    }
    if (becameIdle) {
      views.benchmark.stale = true;
      views.quality.stale = true;
      views.audit.stale = true;
      views.datasets.stale = true;
      if (drawerJobId) renderJobDrawer(drawerJobId);
      if (views[currentView] && views[currentView].loaded) render(currentView);
    }
  } catch (e) {}
  finally {
    jobPolling = false;
  }
}

(async () => {
  await pollJobs();
  render("overview");
  setInterval(pollJobs, POLL_MS);
})();