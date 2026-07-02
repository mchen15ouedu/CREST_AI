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

const map = L.map("map", { zoomControl: true, layers: [esriTopo] }).setView([39, -98], 4);
L.control.layers({ "Topographic": esriTopo, "Satellite": esriImg, "Dark": osm }, {},
  { position: "topright" }).addTo(map);

// ---- state -------------------------------------------------------------
const gaugeMarkers = {};          // id -> marker
const gaugeData = {};             // id -> {id,name,lat,lon,area_km2}
const selected = new Set();
let eventLayer = L.layerGroup().addTo(map);
let gaugeLayer = L.layerGroup().addTo(map);
let MAX_SIMS = 10;

let queryCtx = null;              // {t_start, t_end, bbox}
const simHydro = {};              // gid -> accumulated rows
const gaugeResult = {};           // gid -> {meta, metrics, report}
const overlays = {};              // gid -> L.imageOverlay
let panelGauge = null;            // gauge focused in the right panel
let currentSim = null;

// streamflow animation
const gaugeFrames = {};           // gid -> {n, bounds}
let animTimes = [];               // shared time axis (labels)
let animMax = 0;                  // max frame index
let animIdx = 0;
let animTimer = null;

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
function addMsg(text, cls = "bot") {
  const d = document.createElement("div");
  d.className = "msg " + cls; d.innerHTML = text; log.appendChild(d);
  log.scrollTop = log.scrollHeight; return d;
}

async function runQuery(text) {
  addMsg(text, "user");
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
  } catch (err) {
    s.textContent = "⚠️ " + err.message;
  }
}

function renderResult(d) {
  eventLayer.clearLayers(); gaugeLayer.clearLayers();
  Object.keys(gaugeMarkers).forEach((k) => delete gaugeMarkers[k]);
  Object.keys(gaugeData).forEach((k) => delete gaugeData[k]);
  selected.clear();
  queryCtx = { t_start: d.t_start, t_end: d.t_end, bbox: d.bbox };
  Object.values(overlays).forEach((o) => map.removeLayer(o));
  Object.keys(overlays).forEach((k) => delete overlays[k]);
  Object.keys(simHydro).forEach((k) => delete simHydro[k]);
  Object.keys(gaugeResult).forEach((k) => delete gaugeResult[k]);
  resetAnim();

  (d.event_pins || []).forEach((e) => {
    L.circleMarker([e.lat, e.lon], { radius: 9, color: "#fff", weight: 2,
      fillColor: "#e74c3c", fillOpacity: 0.95 })
      .bindTooltip(`🌊 ${e.label}`, { direction: "top" }).addTo(eventLayer);
  });
  (d.gauge_pins || []).forEach((g) => {
    gaugeData[g.id] = g;
    const m = L.circleMarker([g.lat, g.lon], gaugeStyle(g.id))
      .bindTooltip(`${g.id} · ${g.name}<br>${Math.round(g.area_km2).toLocaleString()} km²`, { direction: "top" })
      .on("click", () => toggleGauge(g.id));
    m.addTo(gaugeLayer); gaugeMarkers[g.id] = m;
  });
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
  focusGauge(id);                 // open right panel on the clicked gauge
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
  addMsg(`▶ Simulating ${ids.length} gauge(s)…`, "status");
  const r = await fetch("/api/simulate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      gauge_ids: ids, t_start: tStart, t_end: tEnd, ...opt,
    }),
  });
  const d = await r.json();
  if (d.warning) addMsg("⚠️ " + d.warning, "status");
  resetAnim();
  ids.forEach((id) => { simHydro[id] = []; });
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
    addMsg(`<b>${ev.gauge_id}</b> · ${ev.msg}`, "status");
  } else if (ev.kind === "hydro") {
    (simHydro[ev.gauge_id] = simHydro[ev.gauge_id] || []).push(...ev.rows);
    if (ev.gauge_id === panelGauge) renderHydro(ev.gauge_id);
  } else if (ev.kind === "q2d") {
    updateOverlay(simId, ev.gauge_id, ev.bounds, ev.frame);
  } else if (ev.kind === "gauge_done") {
    addMsg(`✅ <b>${ev.gauge_id}</b> complete (${ev.n} steps)`, "status");
    if (ev.gauge_id === panelGauge) renderHydro(ev.gauge_id);
  } else if (ev.kind === "result") {
    gaugeResult[ev.gauge_id] = { meta: ev.meta, metrics: ev.metrics, report: ev.report };
    if (ev.gauge_id === panelGauge) { renderStats(ev.gauge_id); renderReport(ev.gauge_id); }
  } else if (ev.kind === "timeline") {
    gaugeFrames[ev.gauge_id] = { n: ev.n, bounds: ev.bounds };
    if (ev.n - 1 > animMax) { animMax = ev.n - 1; animTimes = ev.times || animTimes; }
    showAnim();
  } else if (ev.kind === "all_done") {
    addMsg("✅ All simulations complete — use the time bar to replay the flood.", "status");
    setFrame(animMax);            // rest on the final timestep
  }
}

// ---- streamflow time animation -----------------------------------------
function resetAnim() {
  stopPlay();
  Object.keys(gaugeFrames).forEach((k) => delete gaugeFrames[k]);
  animTimes = []; animMax = 0; animIdx = 0;
  document.getElementById("anim").classList.add("hidden");
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
    else if (info.bounds) overlays[gid] = L.imageOverlay(url, info.bounds, { opacity: 0.9, interactive: false }).addTo(map);
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

function updateOverlay(simId, gid, bounds, frame) {
  const url = `/api/overlay/${simId}/${gid}.png?f=${frame}`;
  if (overlays[gid]) { overlays[gid].setBounds(bounds); overlays[gid].setUrl(url); }
  else overlays[gid] = L.imageOverlay(url, bounds, { opacity: 0.85, interactive: false }).addTo(map);
}

// ---- right panel: focus a gauge + live hydrograph ----------------------
function focusGauge(id) {
  panelGauge = id;
  const g = gaugeData[id];
  document.getElementById("right-panel").classList.remove("hidden");
  document.getElementById("rp-title").textContent = `${id} · ${g ? g.name : ""}`;
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
    { x, y: obs, name: "Obs Q", mode: "lines", line: { color: "#f4f4f4", width: 1.3 } },
    { x, y: sim, name: "Sim Q", mode: "lines", line: { color: "#4cc9a0", width: 1.6 } },
  ], {
    height: 250, margin: { l: 46, r: 46, t: 12, b: 30 }, bargap: 0,
    showlegend: true, legend: { orientation: "h", y: 1.18, font: { size: 9 } },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#cdd9e2", size: 10 },
    xaxis: { gridcolor: "rgba(255,255,255,.06)" },
    yaxis: { title: "Q m³/s", rangemode: "tozero", gridcolor: "rgba(255,255,255,.06)" },
    yaxis2: { overlaying: "y", side: "right", range: [maxp * 3.4, 0], showgrid: false },
  }, { displayModeBar: false, responsive: true });
}

// ---- chat select shortcuts ("simulate all") ----------------------------
function handleChat(text) {
  const t = text.trim().toLowerCase();
  if (/simulate all|all gauges|run all/.test(t)) {
    Object.keys(gaugeData).forEach((id) => selected.add(id));
    refreshSelection();
    if (selected.size > MAX_SIMS) addMsg(`⚠️ ${selected.size} selected — the demo runs at most ${MAX_SIMS} at once.`, "status");
    return;
  }
  runQuery(text);
}

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
