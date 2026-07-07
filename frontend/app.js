"use strict";

// ---- map ---------------------------------------------------------------
const esriImg = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  { attribution: "Imagery © Esri", maxZoom: 19 });
const esriTopo = L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
  { attribution: "© Esri", maxZoom: 19 });
const osm = L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
  { attribution: "© OSM © CARTO", subdomains: "abcd", maxZoom: 19 });

const map = L.map("map", { zoomControl: true, layers: [esriTopo] }).setView([39, -98], 5);
// dedicated pane so 2-D streamflow draws above tiles but below pins
map.createPane("q2d");
map.getPane("q2d").style.zIndex = 450;
const q2dGroup = L.layerGroup().addTo(map);      // toggleable in the layers control
L.control.layers({ "Topographic": esriTopo, "Satellite": esriImg, "Dark": osm },
  { "2-D streamflow": q2dGroup }, { position: "topright" }).addTo(map);

// ---- state -------------------------------------------------------------
const gaugeMarkers = {};          // id -> marker
const gaugeData = {};             // id -> {id,name,lat,lon,area_km2}
const selected = new Set();
let eventLayer = L.layerGroup().addTo(map);
let gaugeLayer = L.layerGroup().addTo(map);
let MAX_SIMS = 10;

let queryCtx = null;              // {t_start, t_end, bbox, label}  (AI-defined window)
let chatTime = null;              // {start, end|null} — dates the user typed in chat
let manualTime = null;            // {start|null, end|null} — Model options “Set” override
let lastSim = null;               // {tStart, tEnd, hours, expectedSteps}
let awaitingTime = false;         // waiting for the user to give a date range/link
let pendingQuery = null;          // original query text while awaiting time
let simRunning = false;           // a job is in flight
let selKeyAtRun = "";             // selection snapshot when Simulate was clicked
let currentES = null;             // the open SSE connection (only ever one)
const simHydro = {};              // gid -> accumulated rows
const gaugeResult = {};           // gid -> {meta, metrics, report}
const gaugeState = {};            // gid -> "running" | "done"
const overlays = {};              // gid -> L.imageOverlay (inside q2dGroup)
let panelGauge = null;            // gauge focused in the right panel
let currentSim = null;
let zoomedToOverlay = false;

// streamflow animation
const gaugeFrames = {};           // gid -> {n, bounds}
let animTimes = [];
let animMax = 0;
let animIdx = 0;
let animTimer = null;

// AI info mode (default ON, persisted)
let aiInfo = localStorage.getItem("aiInfo") !== "off";

// ---- drawing (rectangle select) ----------------------------------------
const drawn = new L.FeatureGroup().addTo(map);
const drawControl = new L.Control.Draw({
  draw: { rectangle: { shapeOptions: { color: "#f4a259", weight: 2, fillOpacity: 0.05 } },
          polygon: false, polyline: false, circle: false, marker: false, circlemarker: false },
  edit: false,
});
map.addControl(drawControl);
map.on(L.Draw.Event.CREATED, (e) => {
  drawn.clearLayers(); drawn.addLayer(e.layer);
  const b = e.layer.getBounds();
  Object.values(gaugeData).forEach((g) => {
    if (b.contains([g.lat, g.lon])) selected.add(g.id);
  });
  refreshSelection();
});

// ---- chat --------------------------------------------------------------
const log = document.getElementById("chat-log");
function addMsg(html, cls = "bot") {
  const d = document.createElement("div");
  d.className = "msg " + cls; d.innerHTML = html; log.appendChild(d);
  log.scrollTop = log.scrollHeight; return d;
}
function statusMsg(gid, text) {           // raw log line — only when AI info is OFF
  if (!aiInfo) addMsg(`<b>${gid}</b> · ${text}`, "status");
}

// ---- map-first gauge pins (no AI needed) --------------------------------
let vpTimer = null;
async function loadViewportGauges() {
  const b = map.getBounds();
  try {
    const r = await fetch(`/api/gauges?w=${b.getWest()}&s=${b.getSouth()}&e=${b.getEast()}&n=${b.getNorth()}`);
    if (!r.ok) return;
    const d = await r.json();
    MAX_SIMS = d.max_sims || MAX_SIMS;
    addGaugePins(d.gauge_pins || []);
  } catch (_) { /* offline / transient */ }
}
map.on("moveend", () => { clearTimeout(vpTimer); vpTimer = setTimeout(loadViewportGauges, 400); });

function addGaugePins(pins) {
  pins.forEach((g) => {
    if (gaugeMarkers[g.id]) return;                   // already on the map
    gaugeData[g.id] = g;
    const m = L.circleMarker([g.lat, g.lon], gaugeStyle(g.id))
      .bindTooltip(`${g.id} · ${g.name}<br>${Math.round(g.area_km2).toLocaleString()} km²`, { direction: "top" })
      .on("click", () => toggleGauge(g.id));
    m.addTo(gaugeLayer); gaugeMarkers[g.id] = m;
  });
}

async function runQuery(text, userDates, echo = true) {
  if (echo) addMsg(escapeHtml(text), "user");
  const s = addMsg("🧭 Analyzing…", "status");
  try {
    const r = await fetch("/api/query", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const d = await r.json();
    MAX_SIMS = d.max_sims || 10;
    renderResult(d);
    s.textContent = `📍 ${d.label} — ${d.n_gauges} USGS gauges nearby. ` +
      `Pick a few gauges closest to the event — click the pins or draw a small box ` +
      `(up to ${MAX_SIMS}; fewer runs faster).`;
    if (!d.time_known && !userDates) {
      awaitingTime = true; pendingQuery = text;
      addMsg("🗓️ I found the <b>place</b> but couldn't pin down <b>when</b> this happened. " +
        "Reply with a date range like <code>2025-07-03 to 2025-07-06</code>, a single start date, " +
        "or paste a news link about the event — or set dates in ⚙️ Model options.", "bot");
    } else if (userDates) {
      confirmDates(userDates, d);
    }
    if (aiInfo) fetchEventInfo();      // event background BEFORE simulating
  } catch (err) {
    if (/could not parse a location/i.test(err.message)) {
      s.remove();
      addMsg("🧭 I couldn't place that yet — let's narrow it down together. " +
        "Which <b>region</b> are you interested in (a state, city or river)? " +
        "And roughly <b>when</b> — a recent event or a historic one? For example: " +
        "<i>“flood in Kerrville, Texas, July 2025”</i>, <i>“Allagash River spring flood”</i>, " +
        "or just click any blue gauge pin on the map.", "bot");
    } else {
      s.textContent = "⚠️ " + err.message;
    }
  }
}

// the user typed explicit dates AND the AI proposed a window — if they differ,
// ask which one to use (chat wins only after the user confirms)
function confirmDates(ud, d) {
  const aiS = d.time_known && d.t_start ? d.t_start.slice(0, 10) : null;
  const aiE = d.time_known && d.t_end ? d.t_end.slice(0, 10) : null;
  const same = aiS === ud.start && (!ud.end || aiE === ud.end);
  if (!aiS || same) {                     // no AI window, or they agree -> use typed dates
    setChatTime(ud);
    return;
  }
  const card = addMsg(
    `🗓️ You wrote <b>${ud.start}${ud.end ? " → " + ud.end : ""}</b>, but the AI identified this ` +
    `event as <b>${aiS}${aiE ? " → " + aiE : ""}</b>. Which window should I simulate?` +
    `<div class="btn-row"><button class="primary" data-act="mine">Use my dates</button>` +
    `<button data-act="ai">Use the AI's dates</button></div>`, "bot");
  card.querySelector('[data-act="mine"]').onclick = () => {
    card.querySelector(".btn-row").remove();
    setChatTime(ud, "Using <b>your</b> dates.");
  };
  card.querySelector('[data-act="ai"]').onclick = () => {
    card.querySelector(".btn-row").remove();
    chatTime = null;                      // AI window (queryCtx) applies
    addMsg(`🗓️ Using the AI-identified window <b>${aiS}${aiE ? " → " + aiE : ""}</b>.`, "bot");
    allowResim();
  };
}

function renderResult(d) {
  eventLayer.clearLayers();
  queryCtx = { t_start: d.t_start, t_end: d.t_end, bbox: d.bbox, label: d.label };
  (d.event_pins || []).forEach((e) => {
    L.circleMarker([e.lat, e.lon], { radius: 9, color: "#fff", weight: 2,
      fillColor: "#e74c3c", fillOpacity: 0.95 })
      .bindTooltip(`🌊 ${e.label}`, { direction: "top" }).addTo(eventLayer);
  });
  addGaugePins(d.gauge_pins || []);
  if (d.bbox) map.fitBounds([[d.bbox[1], d.bbox[0]], [d.bbox[3], d.bbox[2]]], { padding: [40, 40] });
  refreshSelection();
}

function gaugeStyle(id) {
  const on = selected.has(id);
  return { radius: on ? 8 : 5, color: on ? "#fff" : "#0b0f14", weight: on ? 2 : 1,
    fillColor: on ? "#ffd479" : "#3aa3ff", fillOpacity: 0.95 };
}
function toggleGauge(id) {
  selected.has(id) ? selected.delete(id) : selected.add(id);
  refreshSelection();
  if (simHydro[id]) focusGauge(id);        // has results -> show them
}
function selKey() { return [...selected].sort().join(","); }

function refreshSelection() {
  Object.entries(gaugeMarkers).forEach(([id, m]) => m.setStyle(gaugeStyle(id)));
  const n = selected.size;
  document.getElementById("selinfo").textContent = n ? `${n} gauge${n > 1 ? "s" : ""} selected` : "";
  const b = document.getElementById("btn-sim");
  // greyed out for the selection that was already simulated (or is running) —
  // changing gauges, the time window, or any model option re-enables it
  const held = selKeyAtRun !== "" && selKey() === selKeyAtRun;
  b.textContent = (held && simRunning) ? "⏳ Simulating…" : `▶ Simulate (${n})`;
  b.disabled = n === 0 || held;
  if (n > MAX_SIMS) document.getElementById("selinfo").textContent += `  ⚠ max ${MAX_SIMS}`;
}

// something that changes what a new run would compute (window/params) happened —
// let the user hit Simulate again even with the same gauges
function allowResim() {
  selKeyAtRun = "";
  refreshSelection();
}

function readOptions() {
  const ov = advancedOverrides();
  return {
    hours: parseInt(document.getElementById("k-hours").value) || 48,
    model: document.getElementById("k-model").value,
    snow: document.getElementById("k-snow").value,
    timestep: document.getElementById("k-step").value,
    warmup_days: (() => { const v = parseInt(document.getElementById("k-warmup").value, 10);
                          return Number.isFinite(v) ? v : 90; })(),
    overrides: Object.keys(ov).length ? ov : null,
  };
}

// ---- time-window resolution -------------------------------------------
// precedence per field: Model-options “Set” override  >  chat-typed dates  >
// AI-identified event window. A field left blank at a higher level falls
// through to the next. No end anywhere -> start + Duration knob.
function resolveWindow(hours) {
  const pick = (k) =>
    manualTime && manualTime[k] ? [manualTime[k], "manual"] :
    chatTime && chatTime[k] ? [chatTime[k], "chat"] :
    queryCtx && queryCtx["t_" + k] ? [queryCtx["t_" + k].slice(0, 10), "AI"] : [null, null];
  const [s, srcS] = pick("start");
  if (!s) return null;
  const tStart = s.length > 10 ? s : `${s}T00:00:00`;
  let [e, srcE] = pick("end");
  let tEnd = e ? (e.length > 10 ? e : `${e}T00:00:00`) : null;
  if (tEnd && tEnd <= tStart) { tEnd = null; srcE = null; }    // guard nonsense
  if (!tEnd) {
    tEnd = new Date(new Date(tStart + "Z").getTime() + hours * 3600e3)
      .toISOString().slice(0, 19);
    srcE = "duration knob";
  }
  return { tStart, tEnd, srcS, srcE };
}

function windowHours(t0, t1) {
  return Math.max(1, Math.round((new Date(t1 + "Z") - new Date(t0 + "Z")) / 3600e3));
}

async function simulate() {
  if (simRunning && selKey() === selKeyAtRun) return;   // double-click guard
  const ids = [...selected];
  const opt = readOptions();
  const win = resolveWindow(opt.hours);
  if (!win) {                             // map-first flow with no time context
    addMsg("🗓️ I need a time period first — tell me the event (e.g. <i>“flood in Kerrville, July 2025”</i>), " +
      "reply with a date range like <code>2025-07-03 to 2025-07-06</code>, or set dates in ⚙️ Model options.", "bot");
    awaitingTime = true;
    document.getElementById("left-panel").classList.remove("collapsed");
    return;
  }
  const H = windowHours(win.tStart, win.tEnd);
  addMsg(`▶ Simulating ${ids.length} gauge(s) — <b>${win.tStart.slice(0, 16).replace("T", " ")}</b> → ` +
    `<b>${win.tEnd.slice(0, 16).replace("T", " ")}</b> (${H} h · start from ${win.srcS}, end from ${win.srcE})`, "status");
  simRunning = true; selKeyAtRun = selKey(); refreshSelection();
  let d;
  try {
    const r = await fetch("/api/simulate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gauge_ids: ids, t_start: win.tStart, t_end: win.tEnd, ...opt }),
    });
    d = await r.json();
    if (!r.ok || !d.sim_id) throw new Error(d.detail || "simulation could not start");
  } catch (err) {
    addMsg("⚠️ " + err.message, "status");
    simRunning = false; refreshSelection();
    return;
  }
  if (d.warning) addMsg("⚠️ " + d.warning, "status");
  localStorage.setItem("lastSimId", d.sim_id);   // reattach after closing the app
  resetAnim();
  zoomedToOverlay = false;
  const tS = d.t_start || win.tStart, tE = d.t_end || win.tEnd;   // server may clamp
  const HH = windowHours(tS, tE);
  lastSim = { tStart: tS, tEnd: tE, hours: HH, expectedSteps: HH + 1 };
  ids.forEach((id) => { simHydro[id] = []; gaugeState[id] = "running"; delete gaugeResult[id]; });
  renderTabs();
  if (ids.length && !panelGauge) focusGauge(ids[0]);
  if (aiInfo) {
    initProgress(ids);
    fetchEventInfo();
  }
  openStream(d.sim_id);
}

function openStream(simId) {
  if (currentES) { currentES.close(); currentES = null; }   // never two streams
  currentSim = simId;
  const es = currentES = new EventSource(`/api/stream/${simId}`);
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    handleSimEvent(simId, ev);
    if (ev.kind === "all_done") es.close();
  };
  es.onerror = () => {
    es.close();
    if (currentES === es) { simRunning = false; refreshSelection(); }
  };
}

function handleSimEvent(simId, ev) {
  if (ev.kind === "status") {
    statusMsg(ev.gauge_id, ev.msg);
    if (aiInfo) progressFromStatus(ev.gauge_id, ev.msg);
  } else if (ev.kind === "hydro") {
    (simHydro[ev.gauge_id] = simHydro[ev.gauge_id] || []).push(...ev.rows);
    if (aiInfo) progressFromRows(ev.gauge_id);
    if (ev.gauge_id === panelGauge) renderHydro(ev.gauge_id);
  } else if (ev.kind === "q2d") {
    updateOverlay(simId, ev.gauge_id, ev.bounds, ev.frame);
  } else if (ev.kind === "gauge_done") {
    gaugeState[ev.gauge_id] = "done";
    renderTabs();
    if (aiInfo) setProgress(ev.gauge_id, 100, "complete ✓");
    else addMsg(`✅ <b>${ev.gauge_id}</b> complete (${ev.n} steps)`, "status");
    if (ev.gauge_id === panelGauge) renderHydro(ev.gauge_id);
  } else if (ev.kind === "result") {
    gaugeResult[ev.gauge_id] = { meta: ev.meta, metrics: ev.metrics, report: ev.report };
    renderTabs();
    if (ev.gauge_id === panelGauge) { renderStats(ev.gauge_id); renderReport(ev.gauge_id); }
    maybeOfferCalibration(ev.gauge_id, ev.metrics);
  } else if (ev.kind === "timeline") {
    gaugeFrames[ev.gauge_id] = { n: ev.n, bounds: ev.bounds };
    if (ev.n - 1 > animMax) { animMax = ev.n - 1; animTimes = ev.times || animTimes; }
    if (ev.vmax) document.getElementById("q-max").textContent = Math.round(ev.vmax).toLocaleString();
    showAnim();
  } else if (ev.kind === "all_done") {
    addMsg("✅ All simulations complete — use the time bar to replay the flood, and the tabs " +
      "in the results panel to switch between gauges.", "status");
    setFrame(animMax);
    simRunning = false;                    // same selection stays greyed out until it changes
    refreshSelection();
  }
}

// ---- AI info: per-gauge progress bars + event background ----------------
let progressBox = null;
const progressEls = {};           // gid -> {fill, stage, pct, value}

function initProgress(ids) {
  progressBox = addMsg("", "progress");
  ids.forEach((gid) => {
    const row = document.createElement("div"); row.className = "pg-row";
    row.innerHTML = `<span class="pg-name">${gid}</span>
      <span class="pg-track"><span class="pg-fill"></span></span>
      <span class="pg-pct">0%</span><span class="pg-stage">queued…</span>`;
    progressBox.appendChild(row);
    progressEls[gid] = { fill: row.querySelector(".pg-fill"), stage: row.querySelector(".pg-stage"),
                         pct: row.querySelector(".pg-pct"), value: 0 };
  });
  log.scrollTop = log.scrollHeight;
}

function setProgress(gid, pct, stage) {
  const p = progressEls[gid];
  if (!p) return;
  p.value = Math.max(p.value, Math.min(100, pct));   // monotonic
  p.fill.style.width = p.value + "%";
  p.pct.textContent = Math.round(p.value) + "%";
  if (stage) p.stage.textContent = stage;
}

// map raw pipeline statuses to friendly stages + progress. The LABEL always
// describes the step that is happening NOW (statuses are emitted just before
// each long operation), so during e.g. the warm-up the bar says “warm-up”.
const STAGES = [
  [/another simulation of this gauge/i, 3, "queued — waiting for another run of this gauge to finish…"],
  [/re-running the window to render/i, 6, "re-running to render the 2-D streamflow maps…"],
  [/reused .* cached/i,  8,  "reusing cached results for the overlap"],
  [/simulation window/i, 10, null],
  [/clip DEM/i,          12, "preparing terrain (DEM, flow direction, accumulation)…"],
  [/clip @gauge/i,       14, null],
  [/derived from the DEM/i, 15, "flow network rebuilt from the DEM (pysheds)"],
  [/SNOW17 enabled/i,    17, "snow module ON (cold basin detected)"],
  [/no snow/i,           17, "snow module off (warm basin)"],
  [/stored best parameters/i, 18, "loading this basin's best-known parameters"],
  [/-day warm-up from/i, 20, null],       // plan only — the run itself comes later
  [/short warm-up/i,     20, null],
  [/warm-up disabled/i,  20, "cold start (no warm-up)"],
  [/warm start from exact/i, 22, "warm-starting from a saved model state"],
  [/USGS observed/i,     24, "downloading observed discharge (USGS)…"],
  [/downloading rainfall/i, 28, "downloading rainfall forcing (MRMS)…"],
  [/downloading PET/i,   36, "downloading PET forcing…"],
  [/downloading temperature/i, 38, "downloading temperature forcing (snow)…"],
  [/running warm-up/i,   42, "running the warm-up simulation (builds the initial soil state)…"],
  [/warm-up done/i,      58, "warm-up finished — initial state saved"],
  [/running CREST/i,     60, "CREST is running — hydrograph streaming live…"],
  [/served entirely from cache/i, 95, "results served from cache"],
];

function progressFromStatus(gid, msg) {
  for (const [re, pct, label] of STAGES) {
    if (re.test(msg)) { setProgress(gid, pct, label || undefined); return; }
  }
  if (/⚠️/.test(msg)) setProgress(gid, undefined === undefined ? (progressEls[gid]?.value || 0) : 0,
                                  msg.replace(/⚠️\s*/, "⚠ ").slice(0, 90));
}

function progressFromRows(gid) {
  const n = (simHydro[gid] || []).length;
  const exp = lastSim?.expectedSteps || 49;
  setProgress(gid, 60 + 38 * Math.min(1, n / exp),
              `simulating — ${n}/${exp} timesteps`);
}

let lastEventInfoLabel = null;      // one news card per event, not per click
async function fetchEventInfo() {
  if (!queryCtx?.label || queryCtx.label === lastEventInfoLabel) return;
  lastEventInfoLabel = queryCtx.label;
  const card = addMsg(`<div class="news-h">📰 About this event — ${escapeHtml(queryCtx.label)}</div>` +
                      `<i>looking up impacts, damage and news…</i>`, "news");
  try {
    const r = await fetch("/api/eventinfo", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: queryCtx.label, t_start: queryCtx.t_start, t_end: queryCtx.t_end }),
    });
    const d = await r.json();
    card.innerHTML = `<div class="news-h">📰 About this event — ${escapeHtml(queryCtx.label)}</div>` +
                     mdLite(d.text || "(no information found)");
  } catch (_) {
    card.innerHTML += "<br><i>(event lookup unavailable)</i>";
  }
}

// minimal markdown: links + bold + newlines (LLM output is short + trusted-ish)
function mdLite(t) {
  return escapeHtml(t)
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/\n/g, "<br>");
}
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ---- auto-calibration offer (NSE < 0.3) ---------------------------------
const calOffered = new Set();
function maybeOfferCalibration(gid, metrics) {
  const nse = metrics && metrics.nsce;
  if (nse == null || nse >= 0.3 || calOffered.has(gid)) return;
  calOffered.add(gid);
  const card = addMsg(
    `📉 <b>${gid}</b> finished with NSE = <b>${nse}</b> (below 0.3 — a weak fit). ` +
    `Want help calibrating the parameters?` +
    `<div class="btn-row"><button class="primary" data-act="ai">🤖 AI calibration</button>` +
    `<button data-act="manual">🛠 I'll adjust manually</button></div>`, "bot");
  card.querySelector('[data-act="ai"]').onclick = () => { card.querySelector(".btn-row").remove(); startCalibration(gid); };
  card.querySelector('[data-act="manual"]').onclick = () => {
    card.querySelector(".btn-row").remove();
    const lp = document.getElementById("left-panel");
    lp.classList.remove("collapsed");
    document.getElementById("adv-body").classList.remove("hidden");
    document.getElementById("adv-arrow").textContent = "▾";
    addMsg("🛠 Opened <b>Model options → Advanced parameters</b>. Adjust values and hit Simulate again — " +
      "if your run beats the stored NSE, the parameters are saved for this basin automatically.", "bot");
  };
}

async function startCalibration(gid) {
  if (!lastSim) return;
  const opt = readOptions();
  addMsg(`🤖 Starting AI calibration for <b>${gid}</b> — the assistant proposes parameter changes ` +
    `(within hydrologic bounds), tests each with a real 1-D model run, and keeps the best. ` +
    `The winning parameters are saved for this basin.`, "bot");
  const box = addMsg("", "progress");
  const row = document.createElement("div"); row.className = "pg-row";
  row.innerHTML = `<span class="pg-name">${gid} 🎯</span>
    <span class="pg-track"><span class="pg-fill cal"></span></span>
    <span class="pg-pct">0%</span><span class="pg-stage">baseline run…</span>`;
  box.appendChild(row);
  const fill = row.querySelector(".pg-fill"), stage = row.querySelector(".pg-stage"),
        pct = row.querySelector(".pg-pct");

  const r = await fetch("/api/calibrate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gauge_id: gid, t_start: lastSim.tStart,
                           t_end: lastSim.tEnd || new Date(new Date(lastSim.tStart).getTime() + lastSim.hours * 3600e3).toISOString().slice(0, 19),
                           model: opt.model, snow: opt.snow }),
  });
  const d = await r.json();
  const es = new EventSource(`/api/calstream/${d.cal_id}`);
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.kind === "cal_status") {
      stage.textContent = ev.msg.slice(0, 90);
    } else if (ev.kind === "cal_round") {
      const last = (ev.tried || [])[ev.tried.length - 1];
      const p = last && last.progress ? last.progress * 100 : undefined;
      if (p) { fill.style.width = p + "%"; pct.textContent = Math.round(p) + "%"; }
      stage.textContent = `round ${ev.round}: best NSE so far ${ev.best_nse}`;
    } else if (ev.kind === "cal_hydro") {
      simHydro[gid] = ev.rows;                     // preview the improving fit
      if (gid === panelGauge) renderHydro(gid);
    } else if (ev.kind === "cal_done") {
      es.close();
      fill.style.width = "100%"; pct.textContent = "100%";
      if (ev.error) { stage.textContent = "failed: " + ev.error; return; }
      stage.textContent = `done — NSE ${ev.baseline_nse} → ${ev.best_nse}`;
      addMsg(`🎯 Calibration finished for <b>${gid}</b>: NSE <b>${ev.baseline_nse}</b> → <b>${ev.best_nse}</b>` +
        (ev.saved ? " — saved as this basin's best parameter set (it will be used automatically from now on)."
                  : " — did not beat the stored parameters, keeping the previous set.") +
        ` Hit <b>Simulate</b> again to re-run with the ${ev.saved ? "new" : "existing"} parameters (with the 2-D map).`, "bot");
    }
  };
  es.onerror = () => es.close();
}

// ---- streamflow time animation -----------------------------------------
function resetAnim() {
  stopPlay();
  Object.keys(gaugeFrames).forEach((k) => delete gaugeFrames[k]);
  animTimes = []; animMax = 0; animIdx = 0;
  document.getElementById("anim").classList.add("hidden");
  Object.values(overlays).forEach((o) => q2dGroup.removeLayer(o));
  Object.keys(overlays).forEach((k) => delete overlays[k]);
  document.getElementById("q-legend").classList.add("hidden");
}

function showAnim() {
  const bar = document.getElementById("anim");
  bar.classList.remove("hidden");
  document.getElementById("anim-slider").max = String(animMax);
}

function setFrame(idx) {
  animIdx = Math.max(0, Math.min(animMax, idx | 0));
  document.getElementById("anim-slider").value = String(animIdx);
  document.getElementById("anim-time").textContent = animTimes[animIdx] || `#${animIdx + 1}`;
  Object.entries(gaugeFrames).forEach(([gid, info]) => {
    const i = Math.min(animIdx, info.n - 1);
    const url = `/api/frame/${currentSim}/${gid}/${i}.png`;
    if (overlays[gid]) { overlays[gid].setUrl(url); if (info.bounds) overlays[gid].setBounds(info.bounds); }
    else if (info.bounds) addOverlay(gid, url, info.bounds);
  });
}

function stepFrame() { setFrame(animIdx >= animMax ? 0 : animIdx + 1); }

function togglePlay() {
  if (animTimer) { stopPlay(); return; }
  if (animIdx >= animMax) setFrame(0);
  document.getElementById("anim-play").textContent = "⏸";
  animTimer = setInterval(stepFrame, 350);
}
function stopPlay() {
  if (animTimer) { clearInterval(animTimer); animTimer = null; }
  document.getElementById("anim-play").textContent = "▶";
}

function addOverlay(gid, url, bounds) {
  overlays[gid] = L.imageOverlay(url, bounds,
    { opacity: 0.9, interactive: false, pane: "q2d" }).addTo(q2dGroup);
  document.getElementById("q-legend").classList.remove("hidden");
  if (!zoomedToOverlay) {                 // make the 2-D layer impossible to miss
    zoomedToOverlay = true;
    try { map.fitBounds(bounds, { padding: [60, 60] }); } catch (_) {}
  }
}

function updateOverlay(simId, gid, bounds, frame) {
  const url = `/api/overlay/${simId}/${gid}.png?f=${frame}`;
  if (overlays[gid]) { overlays[gid].setBounds(bounds); overlays[gid].setUrl(url); }
  else addOverlay(gid, url, bounds);
}

// ---- right panel: gauge tabs + live hydrograph ---------------------------
function renderTabs() {
  const bar = document.getElementById("rp-tabs");
  const ids = Object.keys(simHydro);
  bar.innerHTML = "";
  ids.forEach((id) => {
    const t = document.createElement("button");
    t.className = "rp-tab" + (id === panelGauge ? " active" : "") +
                  (gaugeState[id] === "done" ? " done" : gaugeState[id] === "running" ? " running" : "");
    t.innerHTML = `<span class="dot"></span>${id}`;
    t.onclick = () => focusGauge(id);
    bar.appendChild(t);
  });
}

function focusGauge(id) {
  panelGauge = id;
  const g = gaugeData[id];
  document.getElementById("right-panel").classList.remove("hidden");
  document.getElementById("rp-title").textContent = `${id} · ${g ? g.name : ""}`;
  renderTabs();
  renderStats(id);
  renderHydro(id);
  renderReport(id);
}

function statCard(k, v) {
  return `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`;
}

function renderStats(id) {
  const el = document.getElementById("rp-stats");
  const g = gaugeData[id], res = gaugeResult[id];
  const cards = [];
  if (g) cards.push(statCard("Drainage", Math.round(g.area_km2).toLocaleString() + " km²"));
  const m = res && res.metrics;
  if (m) {
    if (m.peak_sim != null) cards.push(statCard("Peak Q (sim)", m.peak_sim + " m³/s"));
    if (m.nsce != null) cards.push(statCard("NSCE", m.nsce));
    if (m.cc != null) cards.push(statCard("Corr (CC)", m.cc));
    if (m.bias_pct != null) cards.push(statCard("% bias", m.bias_pct));
    if (m.rmse != null) cards.push(statCard("RMSE", m.rmse));
  }
  el.innerHTML = cards.join("");
}

function renderReport(id) {
  const el = document.getElementById("rp-report");
  el.innerHTML = "";
  const res = gaugeResult[id];
  if (res && res.report) {
    const h = document.createElement("div"); h.className = "rp-h"; h.textContent = "📄 Report";
    const p = document.createElement("div"); p.textContent = res.report;
    el.append(h, p);
  }
}

function renderHydro(id) {
  const rows = simHydro[id] || [];
  const el = document.getElementById("rp-hydro");
  if (!rows.length) { el.innerHTML = '<div class="muted">Select a gauge and run a simulation to see its hydrograph.</div>'; return; }
  if (el.querySelector(".muted")) el.innerHTML = "";     // drop placeholder before plotting
  const x = rows.map((r) => r.time), sim = rows.map((r) => r.sim_q),
    obs = rows.map((r) => r.obs_q), pr = rows.map((r) => r.precip || 0);
  const maxp = Math.max(0.1, ...pr);
  Plotly.react(el, [
    { x, y: pr, name: "Precip", type: "bar", marker: { color: "#5b9bd5" }, yaxis: "y2", opacity: 0.7 },
    { x, y: obs, name: "Obs Q", mode: "lines",
      line: { color: "#f4f4f4", width: 1.3, shape: "spline", smoothing: 0.8 } },
    { x, y: sim, name: "Sim Q", mode: "lines",
      line: { color: "#4cc9a0", width: 1.8, shape: "spline", smoothing: 0.8 } },
  ], {
    height: 250, margin: { l: 46, r: 46, t: 12, b: 30 }, bargap: 0,
    showlegend: true, legend: { orientation: "h", y: 1.18, font: { size: 9 } },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#cdd9e2", size: 10 },
    xaxis: { gridcolor: "rgba(255,255,255,.06)" },
    yaxis: { title: "Q m³/s", rangemode: "tozero", gridcolor: "rgba(255,255,255,.06)" },
    yaxis2: { overlaying: "y", side: "right", range: [maxp * 3.4, 0], showgrid: false },
  }, { displayModeBar: false, responsive: true });
}

// ---- chat routing + flexible date parsing ---------------------------------
// tokens: 2025-07-03 · 7/3/2025 · 07/03/25 · 3 July 2025 · July 3, 2025
const DATE_TOKEN = /\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[\/.]\d{1,2}[\/.]\d{2,4}|(?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}?,?\s*\d{4}|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s*\d{4}/gi;
const MONTHS = { jan: 1, feb: 2, mar: 3, apr: 4, may: 5, jun: 6, jul: 7, aug: 8, sep: 9, oct: 10, nov: 11, dec: 12 };

function normDate(s) {
  s = s.trim();
  let m = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/);            // ISO
  if (m) return isoDate(+m[1], +m[2], +m[3]);
  m = s.match(/^(\d{1,2})[\/.](\d{1,2})[\/.](\d{2,4})$/);      // US M/D/Y (D/M if M>12)
  if (m) {
    let a = +m[1], b = +m[2], y = +m[3];
    if (y < 100) y += 2000;
    let mo = a, d = b;
    if (a > 12 && b <= 12) { mo = b; d = a; }
    return isoDate(y, mo, d);
  }
  m = s.toLowerCase().match(/^(?:(\d{1,2})\s+)?([a-z]{3})[a-z]*\.?\s*(\d{1,2})?,?\s*(\d{4})$/);
  if (m && MONTHS[m[2]]) {                                     // "July 3, 2025" / "3 July 2025"
    const d = +(m[1] || m[3] || 1);
    return isoDate(+m[4], MONTHS[m[2]], d);
  }
  return null;
}
function isoDate(y, mo, d) {
  if (mo < 1 || mo > 12 || d < 1 || d > 31) return null;
  return `${y}-${String(mo).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
}
function parseUserDates(text) {
  const toks = (text.match(DATE_TOKEN) || []).map(normDate).filter(Boolean);
  if (!toks.length) return null;
  let start = toks[0], end = toks[1] || null;
  if (end && end < start) [start, end] = [end, start];         // tolerate swapped order
  return { start, end };
}

function setChatTime(ud, note) {
  chatTime = { start: ud.start, end: ud.end };
  addMsg(`🗓️ Got it — simulation window starts <b>${ud.start}</b>` +
    (ud.end ? ` and ends <b>${ud.end}</b>` : ` (end = start + the Duration knob)`) +
    (note ? `. ${note}` : ". Select gauges and hit Simulate."), "bot");
  allowResim();
}

let chatHistory = [];             // rolling conversation for the agent

function chatContext() {
  return {
    event: queryCtx ? { label: queryCtx.label, t_start: queryCtx.t_start, t_end: queryCtx.t_end } : null,
    selected: [...selected],
    sim_running: simRunning,
    last_window: lastSim ? { start: lastSim.tStart, end: lastSim.tEnd } : null,
    results: Object.entries(gaugeResult).map(([id, r]) => ({
      gauge: id, name: gaugeData[id] ? gaugeData[id].name : null,
      nse: r.metrics ? r.metrics.nsce : null,
      peak_sim: r.metrics ? r.metrics.peak_sim : null,
    })),
    manual_time_override: manualTime, chat_time: chatTime,
  };
}

function remember(role, content) {
  chatHistory.push({ role, content });
  if (chatHistory.length > 24) chatHistory = chatHistory.slice(-24);
}

async function handleChat(text) {
  const t = text.trim();
  const tl = t.toLowerCase();
  if (/simulate all|all gauges|run all/.test(tl)) {
    Object.keys(gaugeData).forEach((id) => selected.add(id));
    refreshSelection();
    addMsg(`⚠️ ${selected.size} gauges selected — the demo runs at most ${MAX_SIMS} at once; ` +
      `picking just the few nearest the event is faster and clearer.`, "status");
    return;
  }
  const ud = parseUserDates(t);
  // dates-only message, or an answer to "when did this happen?" -> chat-defined window
  const leftover = t.replace(DATE_TOKEN, " ")
    .replace(/\b(from|to|through|until|between|and|simulate|run|please)\b/gi, " ")
    .replace(/[-–—,.\s]+/g, " ").trim();
  if (ud && (awaitingTime || leftover.length < 8)) {
    awaitingTime = false;
    addMsg(escapeHtml(t), "user");
    remember("user", t);
    setChatTime(ud);
    return;
  }
  // a pasted link while waiting for time -> re-parse with the original context
  if (awaitingTime && /https?:\/\//.test(t)) {
    awaitingTime = false;
    runQuery(`${pendingQuery || ""} ${t}`.trim());
    return;
  }
  // everything else -> the conversational agent decides (guide / answer / locate)
  addMsg(escapeHtml(t), "user");
  remember("user", t);
  const thinking = addMsg("💬 thinking…", "status");
  let d = null;
  try {
    const r = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: t, history: chatHistory.slice(0, -1), context: chatContext() }),
    });
    d = await r.json();
  } catch (_) { /* offline -> fallback below */ }
  thinking.remove();
  if (!d || d.action === "fallback" || !d.reply) {
    runQuery(t, ud, false);              // rule-based fallback (no LLM configured)
    return;
  }
  addMsg(mdLite(d.reply), "bot");
  remember("assistant", d.reply);
  if (d.start) {                          // agent extracted/confirmed a window
    chatTime = { start: d.start, end: d.end || null };
    awaitingTime = false;
    allowResim();
  }
  if (d.action === "locate" && d.location_query) {
    awaitingTime = false;
    runQuery(d.location_query, ud || (d.start ? { start: d.start, end: d.end || null } : null), false);
  } else if (d.event_info) {
    lastEventInfoLabel = null;            // user explicitly asked -> refresh the card
    fetchEventInfo();
  }
}

// ---- auth: HF OAuth + profile modal ---------------------------------------
async function initAuth() {
  try {
    const r = await fetch("/api/me");
    const d = await r.json();
    const el = document.getElementById("auth");
    if (!d.user) { el.innerHTML = `<a class="tb-btn" href="/login">Sign in</a>`; return; }
    const pic = d.user.picture ? `<img src="${d.user.picture}" alt="">` : "👤";
    el.innerHTML = `<button class="tb-btn" id="auth-btn">${pic} ${escapeHtml(d.user.name || d.user.username)}</button>`;
    document.getElementById("auth-btn").onclick = () => openProfile(d);
  } catch (_) {}
}

function openProfile(d) {
  const m = document.getElementById("profile-modal");
  m.classList.remove("hidden");
  document.getElementById("pm-name").textContent = d.user.name || d.user.username;
  document.getElementById("pm-username").textContent = "@" + d.user.username + (d.user.dev ? " (dev mode)" : "");
  const av = document.getElementById("pm-avatar");
  if (d.user.picture) { av.src = d.user.picture; av.style.display = ""; } else av.style.display = "none";
  const p = d.profile || {};
  ["display_name", "affiliation", "email", "bio"].forEach((k) => {
    document.getElementById("pf-" + k).value = p[k] || "";
  });
}

document.getElementById("pm-close").onclick = () =>
  document.getElementById("profile-modal").classList.add("hidden");
document.getElementById("profile-modal").addEventListener("click", (e) => {
  if (e.target.id === "profile-modal") e.target.classList.add("hidden");
});
document.getElementById("pf-save").onclick = async () => {
  const body = {};
  ["display_name", "affiliation", "email", "bio"].forEach((k) => {
    body[k] = document.getElementById("pf-" + k).value;
  });
  const r = await fetch("/api/profile", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  document.getElementById("pf-saved").textContent = r.ok ? "✓ saved" : "⚠ failed";
  setTimeout(() => { document.getElementById("pf-saved").textContent = ""; }, 2500);
};

// ---- AI info toggle --------------------------------------------------------
const aiBtn = document.getElementById("ai-info-btn");
function renderAiBtn() {
  aiBtn.textContent = `🤖 AI info: ${aiInfo ? "ON" : "OFF"}`;
  aiBtn.classList.toggle("on", aiInfo);
}
aiBtn.onclick = () => {
  aiInfo = !aiInfo;
  localStorage.setItem("aiInfo", aiInfo ? "on" : "off");
  renderAiBtn();
  addMsg(aiInfo
    ? "🤖 AI info ON — you'll see progress bars, plain-word stages and event background instead of raw logs."
    : "🤖 AI info OFF — showing the raw model log.", "status");
};
renderAiBtn();

// ---- wire UI -----------------------------------------------------------
document.getElementById("chat-send").onclick = () => {
  const el = document.getElementById("chat-text");
  if (el.value.trim()) { handleChat(el.value.trim()); el.value = ""; }
};
document.getElementById("chat-text").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("chat-send").click();
});
document.getElementById("btn-sim").onclick = simulate;
document.querySelectorAll(".panel-head .toggle, .panel-head .close").forEach((btn) => {
  btn.onclick = () => {
    const p = document.getElementById(btn.dataset.target);
    if (btn.classList.contains("close")) p.classList.add("hidden");
    else p.classList.toggle("collapsed");
  };
});
// ---- advanced parameters (all model/routing/snow params) ---------------
const PARAM_GROUPS = [
  { title: "Water balance — CREST / CRESTPHYS",
    keys: ["wm", "b", "im", "ke", "fc", "iwu", "igw", "hmaxaq", "gwc", "gwe"] },
  { title: "Routing — kinematic wave",
    keys: ["under", "leaki", "th", "isu", "alpha", "beta", "alpha0"] },
  { title: "Snow — SNOW17",
    keys: ["uadj", "mbase", "mfmax", "mfmin", "tipm", "nmf", "plwhc", "scf"] },
  { title: "HP — hydrophobic",
    keys: ["precip", "split"] },
];

function buildAdvanced() {
  const body = document.getElementById("adv-body");
  PARAM_GROUPS.forEach((g) => {
    const h = document.createElement("div");
    h.className = "adv-group"; h.textContent = g.title; body.appendChild(h);
    const grid = document.createElement("div");
    grid.className = "adv-grid";
    g.keys.forEach((k) => {
      const l = document.createElement("label");
      l.textContent = k;
      const inp = document.createElement("input");
      inp.type = "number"; inp.step = "any"; inp.placeholder = "auto";
      inp.id = "adv-" + k; inp.dataset.param = k;
      l.appendChild(inp); grid.appendChild(l);
    });
    body.appendChild(grid);
  });
}
buildAdvanced();

// collect only the overridden (non-empty) advanced params
function advancedOverrides() {
  const out = {};
  document.querySelectorAll("#adv-body input[data-param]").forEach((i) => {
    if (i.value.trim() !== "") out[i.dataset.param] = parseFloat(i.value);
  });
  return out;
}

document.getElementById("adv-toggle").onclick = () => {
  const b = document.getElementById("adv-body");
  const open = b.classList.toggle("hidden");
  document.getElementById("adv-arrow").textContent = open ? "▸" : "▾";
};

document.getElementById("anim-play").onclick = togglePlay;
document.getElementById("anim-slider").oninput = (e) => { stopPlay(); setFrame(parseInt(e.target.value)); };

// ---- manual time override (Set / Clear) ----------------------------------
document.getElementById("k-time-set").onclick = () => {
  const s = document.getElementById("k-start").value || null;
  const e = document.getElementById("k-end").value || null;
  if (!s && !e) { addMsg("⚠ Enter a start and/or end date before hitting Set.", "status"); return; }
  if (s && e && e <= s) { addMsg("⚠ The end date must be after the start date.", "status"); return; }
  manualTime = { start: s, end: e };
  document.getElementById("k-time-state").textContent = "🔒 override active";
  addMsg(`🔒 Manual time override <b>active</b>: <b>${s || "(AI/chat start)"}</b> → <b>${e || "(AI/chat end)"}</b>. ` +
    `It overrides the AI- and chat-defined windows until you hit <b>Clear</b>. ` +
    `Fields left blank fall back to the AI/chat value.`, "bot");
  allowResim();
};
document.getElementById("k-time-clear").onclick = () => {
  manualTime = null;
  document.getElementById("k-start").value = "";
  document.getElementById("k-end").value = "";
  document.getElementById("k-time-state").textContent = "";
  addMsg("🔓 Manual time override cleared — the AI/chat-defined window is used again.", "bot");
  allowResim();
};

// changing any model option means a new run would differ — re-enable Simulate
document.querySelectorAll("#left-panel input, #left-panel select").forEach((el) => {
  el.addEventListener("change", allowResim);
});

// ---- reattach: a run keeps going in the backend even if the app is closed --
async function reattach() {
  const simId = localStorage.getItem("lastSimId");
  if (!simId) return;
  try {
    const r = await fetch(`/api/job/${simId}`);
    if (!r.ok) { localStorage.removeItem("lastSimId"); return; }
    const j = await r.json();
    if (j.done && j.age_s > 24 * 3600) { localStorage.removeItem("lastSimId"); return; }
    const tS = j.t_start.slice(0, 19), tE = j.t_end.slice(0, 19);
    const H = windowHours(tS, tE);
    lastSim = { tStart: tS, tEnd: tE, hours: H, expectedSteps: H + 1 };
    j.gauge_ids.forEach((id) => {
      simHydro[id] = [];
      gaugeState[id] = j.done ? "done" : "running";
    });
    renderTabs();
    addMsg(j.done
      ? "🔁 Restored your previous simulation — the run finished in the background while you were away."
      : "🔁 Reattached to your running simulation — it kept going in the background.", "status");
    if (!j.done) {
      simRunning = true;
      selKeyAtRun = j.gauge_ids.slice().sort().join(",");
      if (aiInfo) initProgress(j.gauge_ids);
    }
    refreshSelection();
    openStream(simId);              // cursor 0 -> full event replay rebuilds the UI
    if (j.gauge_ids.length) focusGauge(j.gauge_ids[0]);
  } catch (_) { /* server restarted; job registry is gone */ }
}

// ---- boot ----------------------------------------------------------------
initAuth();
loadViewportGauges();               // gauge pins visible with zero AI interaction
reattach();                         // pick up a run started before the app was closed
