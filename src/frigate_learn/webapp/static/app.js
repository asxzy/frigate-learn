const POLL_MS = 5000;
const QUALITIES = ["useful", "bad", "duplicate", "ignore"];
const MAP50_KEY = "metrics/mAP50(B)";
const RECALL_KEY = "metrics/recall(B)";

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
  if (typeof v === "number" && Number.isFinite(v)) {
    return new Date(v * 1000).toLocaleString();
  }
  return String(v).slice(0, 19).replace("T", " ");
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

function pillNode(text, quality) {
  const span = document.createElement("span");
  span.className = "pill " + (quality || "");
  span.textContent = text;
  return span;
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
  panel.appendChild(h("div", "status-line",
    "runs the selected stages as one job in this server (one job at a time)"));
  const steps = data.steps || [];
  const enabled = new Set(data.auto_enable || []);
  const selected = new Set(steps.length ? (enabled.size ? enabled : steps) : []);
  const grid = h("div", "step-grid");
  const checks = new Map();
  for (const step of steps) {
    const lab = h("label", null);
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = selected.has(step);
    cb.value = step;
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(step));
    grid.appendChild(lab);
    checks.set(step, cb);
  }
  panel.appendChild(grid);
  const controls = h("div", "buttons-row");
  const dryLab = h("label", null);
  const dry = document.createElement("input");
  dry.type = "checkbox";
  dryLab.appendChild(dry);
  dryLab.appendChild(document.createTextNode(" dry run"));
  dryLab.title = "validate and report without writing data";
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
    runBtn.disabled = true;
    status.textContent = "starting job…";
    try {
      const res = await api("/api/jobs/run", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ steps: chosen, dry_run: dry.checked }),
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
  controls.appendChild(runBtn);
  controls.appendChild(dryLab);
  controls.appendChild(status);
  panel.appendChild(controls);
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
      ["type", job.type],
      ["started", fmtTs(job.started_at)],
      ["finished", job.finished_at ? fmtTs(job.finished_at) : "running"],
      ["duration", fmtDur(job.started_at, job.finished_at)],
    ];
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

let charts = [];
function clearCharts() {
  for (const c of charts) { try { c.destroy(); } catch (e) {} }
  charts = [];
}

function plotBox(container) {
  const box = h("div", "plot");
  container.appendChild(box);
  return box;
}

function renderScatter(container, points) {
  if (typeof uPlot === "undefined") {
    container.innerHTML = "";
    container.appendChild(h("div", "plot-missing", "uPlot vendor script missing"));
    return;
  }
  if (!points.length) {
    container.innerHTML = "";
    container.appendChild(h("div", "plot-missing", "no valid data points"));
    return;
  }
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const width = Math.max(320, (container.clientWidth || 600) - 20);
  const opts = {
    width, height: 340,
    scales: { x: { time: false }, y: {} },
    axes: [ { label: "latency ms" }, { label: "mAP50" } ],
    series: [
      {},
      {
        label: "mAP50",
        paths: () => null,
        points: { show: true, size: 8, fill: "#56c7ff", stroke: "#0d1117" },
      },
    ],
  };
  charts.push(new uPlot(opts, [xs, ys], container));
}

function renderLineChart(container, xs, series) {
  if (typeof uPlot === "undefined") {
    container.innerHTML = "";
    container.appendChild(h("div", "plot-missing", "uPlot vendor script missing"));
    return;
  }
  if (!xs.length) {
    container.innerHTML = "";
    container.appendChild(h("div", "plot-missing", "no numeric epoch data"));
    return;
  }
  const width = Math.max(320, (container.clientWidth || 600) - 20);
  const opts = {
    width, height: 340,
    legend: { show: true },
    scales: { x: { time: false }, y: { auto: false, range: [0, 1] } },
    axes: [ { label: "epoch" }, {} ],
    series: [
      {},
      ...series.map((s) => ({
        label: s.label,
        stroke: s.color,
        width: s.width === undefined ? 2 : s.width,
        dash: s.dash || [],
        spanGaps: true,
        points: { show: false },
      })),
    ],
  };
  charts.push(new uPlot(opts, [xs, ...series.map((s) => s.data)], container));
}

const views = {
  overview: { fn: renderOverview, loaded: false, stale: false },
  benchmark: { fn: renderBenchmark, loaded: false, stale: false },
  quality: { fn: renderQuality, loaded: false, stale: false },
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

async function renderOverview(sec) {
  const data = await api("/api/overview");
  refreshStageStrip(data);
  sec.innerHTML = "";
  const cards = h("div", "cards");
  cards.appendChild(countCard("Samples", data.samples.total,
    `verified ${data.samples.verified} / unverified ${data.samples.unverified}`));
  cards.appendChild(countCard("Cameras", data.cameras.length, data.cameras.join(", ") || "none"));
  cards.appendChild(countCard("Disk free", fmtBytes(data.disk_free_bytes), "dataset root"));
  cards.appendChild(countCard("VLM", data.vlm_enabled ? "enabled" : "disabled", "verification engine"));
  const quals = Object.keys(data.samples)
    .filter((k) => k.startsWith("quality_"))
    .filter((k) => data.samples[k] > 0)
    .map((k) => `${k.slice(8)}=${data.samples[k]}`)
    .join(", ");
  cards.appendChild(countCard("Quality", quals || "none", "verdict counts"));
  cards.appendChild(countCard("Auto-enable", data.auto_enable.length,
    data.auto_enable.join(", ") || "none"));
  cards.appendChild(countCard("Next build", data.next_build_version || "—", "dataset version"));
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
      <thead><tr><th>id</th><th>type</th><th>status</th><th>started</th>
      <th>duration</th><th>error</th><th></th></tr></thead>
      <tbody></tbody>`;
    const tbody = table.querySelector("tbody");
    for (const j of jobs) {
      const tr = h("tr");
      tr.innerHTML = `
        <td class="mono">${esc(j.id.slice(0, 8))}</td>
        <td>${esc(j.type)}</td>
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
    const chip = h("span", cls, step);
    if (rep && rep.message) chip.title = rep.message;
    strip.appendChild(chip);
  }
}

async function renderBenchmark(sec) {
  clearCharts();
  const data = await api("/api/benchmark");
  const deps = await api("/api/deployments");
  sec.innerHTML = "";
  const latestByModel = {};
  for (const d of deps.deployments || []) {
    if (!(d.model_name in latestByModel)) latestByModel[d.model_name] = d;
  }

  sec.appendChild(h("div", "status-line",
    `golden ${data.golden} · baseline ${data.baseline} · ` +
    `max latency ${num(data.max_latency_ms, 1)}ms · results updated ${fmtTs(data.results_updated_at) || "—"}`));

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
  cards.appendChild(countCard("Samples", data.total, `one-class boxes ${data.one_class_boxes}`));
  cards.appendChild(countCard("Verified", data.verified, `problematic ${data.problematic}`));
  cards.appendChild(countCard("Unverified", data.unverified, "awaiting review"));
  cards.appendChild(countCard("Frigate-reviewed", data.reviewed, "human-confirmed in Frigate Review UI"));
  cards.appendChild(countCard("Frigate-unreviewed", data.unreviewed, "not yet confirmed by a human"));
  for (const q of QUALITIES) {
    cards.appendChild(countCard(q, data.by_quality[q] ?? 0, "quality verdict"));
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

  sec.appendChild(tableCaption("Review backlog"));
  const samples = backlog.samples || [];
  if (!samples.length) {
    sec.appendChild(h("div", "placeholder", "no unverified samples to review"));
  } else {
    const table = h("table");
    table.innerHTML = `
      <thead><tr><th>id</th><th>camera</th><th>label</th><th>timestamp</th><th></th></tr></thead>
      <tbody></tbody>`;
    const tbody = table.querySelector("tbody");
    samples.forEach((s, i) => {
      const tr = h("tr");
      const link = h("a", null, "review");
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
  clearCharts();
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
    { label: "mAP50", color: "#56c7ff", data: map },
    { label: "recall", color: "#56d364", data: rec },
  ];
  if (beforeMap50 !== null) series.push({ label: "golden before", color: "#8b949e", width: 1, dash: [6, 4], data: xs.map(() => beforeMap50) });
  if (afterMap50 !== null) series.push({ label: "golden after", color: "#d29922", width: 1, dash: [6, 4], data: xs.map(() => afterMap50) });
  renderLineChart(plotBox(sec), xs, series);
  return data;
}

const triageState = {
  quality: "", status: "", camera: "", verified: false, reviewed: "",
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
  bar.appendChild(filterSelect("quality", QUALITIES, triageState.quality,
    (v) => { triageState.quality = v; loadTriageGrid(); }));
  bar.appendChild(filterSelect("status", [...triageState.statuses].sort(), triageState.status,
    (v) => { triageState.status = v; loadTriageGrid(); }));
  bar.appendChild(filterSelect("camera", triageState.cameras.slice().sort(), triageState.camera,
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
  bar.appendChild(filterSelect("reviewed", ["reviewed", "unreviewed"], triageState.reviewed,
    (v) => { triageState.reviewed = v; loadTriageGrid(); }));
  sec.appendChild(bar);

  triageState.grid = h("div", "thumb-grid");
  sec.appendChild(triageState.grid);
  await loadTriageGrid();
}

function filterSelect(name, options, current, onChange) {
  const label = h("span", null, name);
  const select = h("select");
  const empty = h("option", null, "all " + name + "s");
  empty.value = "";
  select.appendChild(empty);
  for (const o of options) {
    const opt = h("option", null, o);
    opt.value = o;
    select.appendChild(opt);
  }
  select.value = current || "";
  select.addEventListener("change", () => onChange(select.value));
  const wrap = h("div", "filter-bar");
  wrap.style.margin = "0";
  wrap.appendChild(label);
  wrap.appendChild(select);
  return wrap;
}

function samplesQuery() {
  const p = new URLSearchParams();
  if (triageState.quality) p.set("quality", triageState.quality);
  if (triageState.status) p.set("status", triageState.status);
  if (triageState.camera) p.set("camera", triageState.camera);
  if (triageState.verified) p.set("verified", 1);
  if (triageState.reviewed === "reviewed") p.set("reviewed", 1);
  if (triageState.reviewed === "unreviewed") p.set("reviewed", 0);
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
      `${data.total} sample(s), showing ${triageState.samples.length}`));
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
      if (s.verified) thumb.appendChild(h("div", "badge-verify", "✓"));
      if (s.reviewed === 1) thumb.appendChild(h("div", "badge-review", "R"));
      const meta = h("div", "meta");
      meta.appendChild(document.createTextNode(
        `${s.camera || "—"} · ${s.label || "—"} · ${num(s.score, 2)} · `));
      meta.appendChild(document.createTextNode(fmtTs(s.timestamp)));
      meta.appendChild(document.createTextNode(" "));
      meta.appendChild(pillNode(s.quality || "unset", s.quality));
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
  meta.appendChild(pillNode(item.quality || "unset", item.quality));
  const revTxt = item.reviewed === 1 ? "reviewed" : item.reviewed === 0 ? "unreviewed" : "";
  if (revTxt) meta.appendChild(pillNode(revTxt, item.reviewed === 1 ? "rev" : "unrev"));

  const qbar = h("div", "lightbox-qual");
  for (const q of QUALITIES) {
    const btn = h("button", null, q);
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
      pillNode(detail.sample.quality || "unset", detail.sample.quality));
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
    `job ${runningJob.id.slice(0, 8)} · pipeline running — click for live log`));
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
    if (becameIdle) {
      views.benchmark.stale = true;
      views.quality.stale = true;
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