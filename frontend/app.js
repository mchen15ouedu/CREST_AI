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

let queryCtx = null;              // {t_start, t_end, bbox, label}
let lastSim = null;               // {tStart, tEnd, hours, expectedSteps}
let awaitingTime = false;         // waiting for the user to give a date range/link
let pendingQuery = null;          // original query text while awaiting time
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

async function runQuery(text) {
  addMsg(escapeHtml(text), "user");
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
    if (!d.time_known) {
      awaitingTime = true; pendingQuery = text;
      addMsg("🗓️ I found the <b>place</b> but couldn't pin down <b>when</b> this happened. " +
        "Reply with a date range like <code>2025-07-03 to 2025-07-06</code>, a single start date, " +
        "or paste a news link about the event — or set the start date in ⚙️ Model options.", "bot");
    }
  } catch (err) {
    s.textContent = "⚠️ " + err.message;
  }
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
function refreshSelection() {
  Object.entries(gaugeMarkers).forEach(([id, m]) => m.setStyle(gaugeStyle(id)));
  const n = selected.size;
  document.getElementById("selinfo").textContent = n ? `${n} gauge${n > 1 ? "s" : ""} selected` : "";
  const b = document.getElementById("btn-sim");
  b.textContent = `▶ Simulate (${n})`; b.disabled = n === 0;
  if (n > MAX_SIMS) document.getElementById("selinfo").textContent += `  ⚠ max ${MAX_SIMS}`;
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

async function simulate() {
  const ids = [...selected];
  const opt = readOptions();
  // the start-date knob overrides the query's event window (end = start + hours)
  const startOv = document.getElementById("k-start").value;
  const tStart = startOv ? `${startOv}T00:00:00` : queryCtx?.t_start;
  const tEnd = startOv ? null : queryCtx?.t_end;
  if (!tStart) {                          // map-first flow with no time context
    addMsg("🗓️ I need a time period first — tell me the event (e.g. <i>“flood in Kerrville, July 2025”</i>), " +
      "reply with a date range like <code>2025-07-03 to 2025-07-06</code>, or set the start date in ⚙️ Model options.", "bot");
    awaitingTime = true;
    document.getElementById("left-panel").classList.remove("collapsed");
    return;
  }
  addMsg(`▶ Simulating ${ids.length} gauge(s)…`, "status");
  const r = await fetch("/api/simulate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gauge_ids: ids, t_start: tStart, t_end: tEnd, ...opt }),
  });
  const d = await r.json();
  if (d.warning) addMsg("⚠️ " + d.warning, "status");
  resetAnim();
  zoomedToOverlay = false;
  const hours = opt.hours;
  lastSim = { tStart, tEnd, hours, expectedSteps: hours + 1 };
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
  currentSim = simId;
  const es = new EventSource(`/api/stream/${simId}`);
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    handleSimEvent(simId, ev);
    if (ev.kind === "all_done") es.close();
  };
  es.onerror = () => es.close();
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

// map raw pipeline statuses to friendly stages + progress
const STAGES = [
  [/clip DEM/i,          8,  "preparing terrain (DEM, flow direction, accumulation)…"],
  [/clip @gauge/i,       10, null],
  [/derived from the DEM/i, 12, "flow network rebuilt from the DEM (pysheds)"],
  [/SNOW17 enabled/i,    14, "snow module ON (cold basin detected)"],
  [/no snow/i,           14, "snow module off (warm basin)"],
  [/stored best parameters/i, 16, "loading this basin's best-known parameters"],
  [/reused .* cached/i,  18, "reusing cached results for the overlap"],
  [/USGS observed/i,     20, "fetched observed discharge from USGS"],
  [/warm start from exact/i, 45, "warm-starting from a saved model state"],
  [/short warm-up/i,     30, "short warm-up bridging to the saved state…"],
  [/-day warm-up from/i, 25, "downloading forcing + warming up the soil state…"],
  [/warm-up disabled/i,  25, "cold start (no warm-up)"],
  [/running warm-up/i,   35, "running the warm-up simulation…"],
  [/warm-up done/i,      55, "warm-up finished — initial state saved"],
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

async function fetchEventInfo() {
  if (!queryCtx?.label) return;
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

// ---- chat routing ---------------------------------------------------------
const DATE_RANGE = /(\d{4}-\d{2}-\d{2})(?:\s*(?:to|through|–|—|-)\s*(\d{4}-\d{2}-\d{2}))?/;

function handleChat(text) {
  const t = text.trim();
  const tl = t.toLowerCase();
  if (/simulate all|all gauges|run all/.test(tl)) {
    Object.keys(gaugeData).forEach((id) => selected.add(id));
    refreshSelection();
    addMsg(`⚠️ ${selected.size} gauges selected — the demo runs at most ${MAX_SIMS} at once; ` +
      `picking just the few nearest the event is faster and clearer.`, "status");
    return;
  }
  // reply to the "when did this happen?" prompt: a date range or single date
  const dm = t.match(DATE_RANGE);
  if (awaitingTime && dm) {
    awaitingTime = false;
    const start = dm[1], end = dm[2];
    queryCtx = queryCtx || {};
    queryCtx.t_start = `${start}T00:00:00`;
    queryCtx.t_end = end ? `${end}T00:00:00` : null;
    document.getElementById("k-start").value = start;
    addMsg(escapeHtml(t), "user");
    addMsg(`🗓️ Got it — simulation window starts <b>${start}</b>${end ? ` and ends <b>${end}</b>` : ""}. ` +
      `Select gauges and hit Simulate.`, "bot");
    return;
  }
  // a pasted link (or anything else) while waiting for time -> re-parse with context
  if (awaitingTime && /https?:\/\//.test(t)) {
    awaitingTime = false;
    runQuery(`${pendingQuery || ""} ${t}`.trim());
    return;
  }
  runQuery(t);
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

// ---- boot ----------------------------------------------------------------
initAuth();
loadViewportGauges();               // gauge pins visible with zero AI interaction
