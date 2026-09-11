# Training View Before/After Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show golden-set **before**/**after** (pretrained vs fine-tuned) alongside a pinned, non-overflowing epoch curve in the Training run detail.

**Architecture:** The Training run detail endpoint (`webapp/queries.py::training_run`) adds a `before_after` field parsed from the run dir's `before_after.json`. The SPA's `renderTrainingDetail` shows two extra cards, renders the before/after values as dashed constant reference series on the epoch curve, and `renderLineChart` gets a y-scale pinned to `[0, 1]` so metric lines cannot leave the plot box.

**Tech Stack:** Python 3.13 / SQLAlchemy (read-side only), vanilla JS + uPlot v1 (vendored, no build step), pytest, external jsdom harness for the JS view, ruff.

## Global Constraints

- **Zero comments** in `src/frigate_learn/webapp/static/app.js` (and anywhere in `webapp/` / `static/`). Vendored uPlot license header excepted. No `//` or `/* */` or `#` comments added by this plan.
- Full pytest suite must stay green (`VENV/bin/python -m pytest -p no:warnings 2>&1 | tail -1` → `N passed`). Everything under `webapp/` is FastAPI-optional: `tests` guard via `pytest.importorskip("fastapi")` at module top; `queries.py` must NOT import optional-extras modules.
- The `ml` extra (torch/ultralytics) is NOT installed in the dev venv; nothing in this plan may import it.
- `ruff` clean on changed files. Binary at `/opt/homebrew/bin/ruff`; run with `--select E,F` over the changed files only (34 pre-existing violations remain untouched).
- `webapp/static/app.js` is a single-file ES script served verbatim (no build step). Keep existing helpers: `h()`, `num()`, `countCard(title, number, sub)`, `tableCaption(title)`, `plotBox(container)`, `clearCharts()`, `charts` array global.
- Default shell is zsh; venv binaries at `.venv/bin/...`. Config fixture helper in tests is `build_config({"data": {"root": "data"}}, tmp_path)` → run dirs live at `tmp_path/data/training/<run>/`.
- Reference-line colors: before `#8b949e`, after `#d29922`, dash `[6, 4]`, width 1.
- Specs: `docs/superpowers/specs/2026-09-10-training-view-before-after-design.md`. Out of scope (do NOT touch): `renderScatter` y-axis, `cli.py`, `training/trainer.py`, pipeline stages, re-training v001/v003.

---

### Task 1: `before_after` in the run detail backend

**Files:**
- Modify: `src/frigate_learn/webapp/queries.py` (add helper near `_read_csv`, wire into `training_run`)
- Test: `tests/test_webapp.py`

**Interfaces:**
- Consumes: `config.resolve(data.root, "training", <run>, "results.csv")` returns a `Path`; `training_run(config, run)` returns a dict (`"run"`, `"columns"`, `"rows"`, `"best_map50"`, `"latest_map50"`, `"epochs"`, `"weights_exists"`, `"dataset"`).
- Produces: `queries.training_run()` now also returns `"before_after": dict | None` with shape
  `{"before": {"map50": float|None, "recall": float|None, "latency_ms": float|None},
    "after": {"map50": float|None, "recall": float|None, "latency_ms": float|None}}`,
  or `None` when the file is absent/unreadable/invalid or a whole segment is missing.
- `_read_before_after(run_dir: Path) -> dict | None` — private helper, only consumer is `training_run`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_webapp.py` (after `test_training_run_nan_empty_inf_become_none`, ~line 455):

```python
def test_training_run_before_after(tmp_path):
    cfg = build_config({"data": {"root": "data"}}, tmp_path)
    run_dir = tmp_path / "data" / "training" / "yolov8n-vba"
    run_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text(
        "epoch,metrics/mAP50(B)\n1,0.5\n", encoding="utf-8"
    )
    (run_dir / "before_after.json").write_text(
        json.dumps(
            {
                "before": {"map50": 0.333, "recall": 0.417, "latency_ms": 15.3},
                "after": {"map50": 0.333, "recall": 0.083, "latency_ms": 14.5},
            }
        ),
        encoding="utf-8",
    )
    result = queries.training_run(cfg, "yolov8n-vba")
    assert result["before_after"] == {
        "before": {"map50": 0.333, "recall": 0.417, "latency_ms": 15.3},
        "after": {"map50": 0.333, "recall": 0.083, "latency_ms": 14.5},
    }


def test_training_run_before_after_missing(seeded):
    cfg, _ = seeded
    result = queries.training_run(cfg, "yolov8n-v001")
    assert result["before_after"] is None


def test_training_run_before_after_corrupt(tmp_path):
    cfg = build_config({"data": {"root": "data"}}, tmp_path)
    run_dir = tmp_path / "data" / "training" / "yolov8n-vbad"
    run_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text(
        "epoch,metrics/mAP50(B)\n1,0.5\n", encoding="utf-8"
    )
    (run_dir / "before_after.json").write_text("{not json", encoding="utf-8")
    result = queries.training_run(cfg, "yolov8n-vbad")
    assert result["before_after"] is None


def test_training_run_before_after_nonfinite_dropped(tmp_path):
    cfg = build_config({"data": {"root": "data"}}, tmp_path)
    run_dir = tmp_path / "data" / "training" / "yolov8n-vnf"
    run_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text(
        "epoch,metrics/mAP50(B)\n1,0.5\n", encoding="utf-8"
    )
    (run_dir / "before_after.json").write_text(
        json.dumps(
            {
                "before": {"map50": float("nan"), "recall": 0.4},
                "after": {"map50": 0.3, "recall": float("inf")},
            }
        ),
        encoding="utf-8",
    )
    result = queries.training_run(cfg, "yolov8n-vnf")
    assert result["before_after"] == {
        "before": {"map50": None, "recall": 0.4, "latency_ms": None},
        "after": {"map50": 0.3, "recall": None, "latency_ms": None},
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -p no:warnings tests/test_webapp.py::test_training_run_before_after tests/test_webapp.py::test_training_run_before_after_missing tests/test_webapp.py::test_training_run_before_after_corrupt tests/test_webapp.py::test_training_run_before_after_nonfinite_dropped -v`
Expected: 3 FAIL (`KeyError: 'before_after'`), 1 FAIL or the missing test passes accidentally if `before_after` were already present — currently `training_run` has no such key, so all asserting `before_after` fail.

- [ ] **Step 3: Implement the helper and wire it in**

In `src/frigate_learn/webapp/queries.py`, add right before `def _read_csv` (line ~385):

```python
def _read_before_after(run_dir: Path) -> dict | None:
    path = run_dir / "before_after.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    def _segment(key: str) -> dict | None:
        seg = payload.get(key)
        if not isinstance(seg, dict):
            return None
        out: dict[str, float | None] = {}
        for metric in ("map50", "recall", "latency_ms"):
            value = seg.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(value):
                out[metric] = float(value)
            else:
                out[metric] = None
        return out

    before = _segment("before")
    after = _segment("after")
    if before is None or after is None:
        return None
    return {"before": before, "after": after}
```

In `training_run` (line ~207), replace the return block with:

```python
    _, tag = _split_run(run)
    run_dir = results_csv.parent
    return {
        "run": run,
        "columns": columns,
        "rows": rows,
        "best_map50": max(map50s) if map50s else None,
        "latest_map50": map50s[-1] if map50s else None,
        "epochs": len(rows),
        "weights_exists": (run_dir / "weights" / "best.pt").is_file(),
        "before_after": _read_before_after(run_dir),
        "dataset": tag,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -p no:warnings tests/test_webapp.py::test_training_run_before_after tests/test_webapp.py::test_training_run_before_after_missing tests/test_webapp.py::test_training_run_before_after_corrupt tests/test_webapp.py::test_training_run_before_after_nonfinite_dropped -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/frigate_learn/webapp/queries.py tests/test_webapp.py
git commit -m "webapp: expose golden before/after metrics in training run detail"
```

---

### Task 2: Frontend — cards, pinned scale, reference series

**Files:**
- Modify: `src/frigate_learn/webapp/static/app.js` (`renderLineChart` ~line 130, `renderTrainingDetail` ~lines 557-589)
- Verify: external jsdom harness (created in this task under a scratch dir, kept outside the repo)

**Interfaces:**
- Consumes: `data.before_after` (shape from Task 1) on the `/api/training/<run>` payload.
- Produces: `renderLineChart(container, xs, series)` now accepts any number of `series` entries, each `{ label, color, width?, dash?, data }`, and pins the y-scale; `renderTrainingDetail` renders 2-3 extra cards and passes 2 dashed constant series when `before_after` is present.

- [ ] **Step 1: Prepare the harness fixtures and write an external failing check**

Create a scratch dir and pull real fixtures from a running server (uses an unused port; do not touch a user dashboard already on 8088):

```bash
mkdir -p /tmp/tv-harness/fixtures
nohup .venv/bin/frigate-learn web --host 127.0.0.1 --port 8093 > /tmp/tv-harness/web.log 2>&1 &
sleep 7
curl -s -m 10 http://127.0.0.1:8093/api/training/yolov8n-v003 > /tmp/tv-harness/fixtures/api_training_yolov8n-v003.json
curl -s -m 10 http://127.0.0.1:8093/api/training/yolov8n-v004 > /tmp/tv-harness/fixtures/api_training_yolov8n-v004.json
curl -s -m 10 http://127.0.0.1:8093/api/training/yolov8n-v001 > /tmp/tv-harness/fixtures/api_training_yolov8n-v001.json
curl -s -m 10 http://127.0.0.1:8093/api/training/list > /tmp/tv-harness/fixtures/api_training_list.json
kill -9 %1 2>/dev/null || true
```

Note: if the repo checkout lacks `v003`/`v004`/`v001` runs, run `dataset build` + a dry-run train first, or substitute any three run names consistently (the harness only needs real `results.csv` content). The `v001` fixture is used as the "no `before_after`" case.

Create `/tmp/tv-harness/run.cjs` with node's `jsdom` resolvable (e.g. `npm i jsdom` in that dir, or reuse an existing harness `node_modules`):

```js
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");
const FIX = path.join(__dirname, "fixtures");
const INDEX = fs.readFileSync("/Users/asxzy/src/frigate-learn/src/frigate_learn/webapp/static/index.html", "utf8");
const UPLOT = fs.readFileSync("/Users/asxzy/src/frigate-learn/src/frigate_learn/webapp/static/vendor/uPlot.iife.min.js", "utf8");
const APPJS = fs.readFileSync("/Users/asxzy/src/frigate-learn/src/frigate_learn/webapp/static/app.js", "utf8");
const dom = new JSDOM(INDEX, { url: "http://localhost:8093/", runScripts: "outside-only", pretendToBeVisual: true,
  beforeParse(window) {
    window.matchMedia = window.matchMedia || ((q) => ({ matches:false, onchange:null, addListener(){}, removeListener(){}, addEventListener(){}, removeEventListener(){}, dispatchEvent:()=>false }));
    window.ResizeObserver = window.ResizeObserver || class { observe(){} unobserve(){} disconnect(){} };
    window.Path2D = window.Path2D || class Path2D { constructor(){} moveTo(){} lineTo(){} bezierCurveTo(){} quadraticCurveTo(){} arc(){} arcTo(){} closePath(){} rect(){} addPath(){} fractalize(){} };
    let ctxStub = null;
    const makeCtx = () => ctxStub || (ctxStub = new Proxy({}, { get(t,k){ if(k==="canvas") return {width:0,height:0}; if(k==="measureText") return ()=>({width:10}); if(k==="getImageData") return ()=>({data:[]}); return ()=>0; }, set(){ return true; } }));
    const proto = window.HTMLCanvasElement && window.HTMLCanvasElement.prototype;
    if (proto) proto.getContext = (kind) => (String(kind).startsWith("2d") ? makeCtx() : null);
    window.fetch = async (p, opts) => {
      let key = String(p).trim().replace(/^http:\/\/[^/]+/, "").replace(/^\//, "").replace(/\//g, "_");
      key = key.replace(/[^a-zA-Z0-9_-]/g, "_");
      const f = path.join(FIX, key + ".json");
      let body = null, status = 200;
      if (fs.existsSync(f)) body = fs.readFileSync(f, "utf8");
      else { status = 404; body = '{"detail":"Not Found"}'; }
      return { ok: status>=200 && status<300, status, headers: { get: (n) => (n.toLowerCase()==="content-type" ? "application/json" : "") }, text: async () => body, json: async () => JSON.parse(body) };
    };
  } });
const window = dom.window, document = window.document;
window.__errors = [];
window.console.error = (...a) => window.__errors.push("console.error: " + a.join(" "));
window.console.warn = (...a) => window.__errors.push("console.warn: " + a.join(" "));
window.addEventListener("error", (e) => window.__errors.push("window.error: " + (e.error && e.error.stack || e.message)));
window.addEventListener("unhandledrejection", (e) => window.__errors.push("unhandledrejection: " + (e.reason && (e.reason.stack || e.reason.message) || e.reason)));
window.eval(UPLOT);
window.eval(APPJS + `
renderTrainingDetail(document.getElementById("view-training"), "yolov8n-v004");
setTimeout(() => renderTrainingDetail(document.getElementById("view-training"), "yolov8n-v001"), 800);
setTimeout(() => {
  window.__probe = {
    root: document.getElementById("view-training"),
    seriesCountForV004: charts[0] ? charts[0].series.length : -1,
    seriesCountForV001: charts[1] ? charts[1].series.length : -1,
    yMaxForV001: charts[1] ? charts[1].scales.y.max : -1,
  };
}, 2400);
`);
setTimeout(() => {
  const probe = window.__probe;
  const html = probe ? probe.root.innerHTML : "";
  const checks = {
    errors: window.__errors.slice(),
    v004htmlHasBefore: html.includes("Golden before"),
    v004htmlHasAfter: html.includes("Golden after"),
    yScaleMaxIs1: probe ? Math.abs(probe.yMaxForV001 - 1) < 1e-9 : false,
    seriesCountForV004: probe ? probe.seriesCountForV004 : -1,
    seriesCountForV001: probe ? probe.seriesCountForV001 : -1,
  };
  console.log(JSON.stringify(checks, null, 1));
  process.exit(0);
}, 3400);
```

- [ ] **Step 2: Run the harness to verify it fails**

Run: `node /tmp/tv-harness/run.cjs`
Expected (current code): `v004htmlHasBefore: false`, `v004htmlHasAfter: false`, `yScaleMaxIs1: false` (auto scale → ~1.1), `seriesCountForV004: 3`, `seriesCountForV001: 3`. `errors` may hold harmless uPlot warnings. This is the RED state.

- [ ] **Step 3: Rewrite `renderLineChart` with N series and a pinned y-scale**

Replace the `renderLineChart` body in `app.js` (lines 130-154) with:

```js
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
```

- [ ] **Step 4: Update `renderTrainingDetail` — cards, caption, reference series**

Replace the body of `renderTrainingDetail` after the `data` fetch in `app.js` (lines 565-589: the `cards` block, the `tableCaption`, and the chart build) with:

```js
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
```

No comments added. Existing `num()` renders 3 decimals so `0.333` shows for v004's equal before/after.

- [ ] **Step 5: Re-run the harness to verify it passes**

Run: `node /tmp/tv-harness/run.cjs`
Expected: `errors: []`, `v004htmlHasBefore: true`, `v004htmlHasAfter: true`, `yScaleMaxIs1: true`, `seriesCountForV004: 5`, `seriesCountForV001: 3`.
The three decisive assertions are the two v004 HTML flags and `yScaleMaxIs1` (pinned to 1 even on a run without `before_after`). `seriesCountForV001` stays 3 because v001 has no reference lines — that is the desired no-before/after degradation.

- [ ] **Step 6: Commit**

```bash
git add src/frigate_learn/webapp/static/app.js
git commit -m "webapp: pinned epoch curve with golden before/after reference lines"
```

---

### Task 3: Full verification pass

**Files:** none (read-only checks).

- [ ] **Step 1: Run the full suite**

Run: `.venv/bin/python -m pytest -p no:warnings 2>&1 | tail -1`
Expected: `N passed` with N ≥ previous count (was 304 before this plan; +4 new webapp tests).

- [ ] **Step 2: Ruff on changed files**

Run:
```bash
/opt/homebrew/bin/ruff check --select E,F src/frigate_learn/webapp/queries.py tests/test_webapp.py 2>&1 | tail -3
```
Expected: no new violations (a pre-existing list may print; it must not include `queries.py`/`test_webapp.py` lines).

- [ ] **Step 3: Live API smoke**

Run:
```bash
nohup .venv/bin/frigate-learn web --host 127.0.0.1 --port 8094 > /tmp/tv-harness/web2.log 2>&1 &
sleep 7
curl -s -m 10 http://127.0.0.1:8094/api/training/yolov8n-v004 | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d['before_after'], sort_keys=True))"
kill -9 %1 2>/dev/null || true
```
Expected: a JSON object with `before`/`after` containing `latency_ms`, `map50`, `recall` keys (finite numbers) — not `null`. For a run without the file (e.g. `yolov8n-v001`) the field prints `null`.

- [ ] **Step 4: Git status sanity**

Run: `git status --short`
Expected: only the two intended files (`webapp/queries.py`, `webapp/static/app.js`, `tests/test_webapp.py`) from this plan's tasks, plus pre-existing unrelated changes. No new `node_modules`, fixture, or scratch dirs inside the repo.

- [ ] **Step 5: Report**

Summarize: what changed, harness/assertion output, suite + ruff results, and note that the reference lines are golden-set vs the curve being val-set per the caption.