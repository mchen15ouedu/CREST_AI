"use strict";

// ---- error watchdog beacon: browser errors -> server crashlog -----------
let errBeaconCount = 0;
function reportClientError(message, source, line, stack) {
  if (errBeaconCount >= 10) return;             // per-page-load cap
  errBeaconCount++;
  try {
    fetch("/api/clienterror", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: String(message || "").slice(0, 400),
        source: String(source || "").slice(0, 200),
        line: Number(line) || 0,
        stack: String(stack || "").slice(0, 1200),
      }),
    }).catch(() => {});
  } catch (e) { /* the beacon itself must never throw */ }
}
window.addEventListener("error", (e) => {
  if (e.message) reportClientError(e.message, e.filename, e.lineno,
                                   e.error && e.error.stack);
});
window.addEventListener("unhandledrejection", (e) =>
  reportClientError("unhandledrejection: " +
      ((e.reason && (e.reason.message || e.reason)) || "?"),
    "", 0, e.reason && e.reason.stack));

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
// bottom-right keeps it clear of the right results panel (which owns the top-right
// edge and used to cover this control — test-user feedback 2026-07-17)
L.control.layers({ "Topographic": esriTopo, "Satellite": esriImg, "Dark": osm },
  { "2-D streamflow": q2dGroup }, { position: "bottomright" }).addTo(map);

// ---- state -------------------------------------------------------------
const gaugeMarkers = {};          // id -> marker
const gaugeData = {};             // id -> {id,name,lat,lon,area_km2}
const selected = new Set();
let eventLayer = L.layerGroup().addTo(map);
let gaugeLayer = L.layerGroup().addTo(map);
let MAX_SIMS = 10;

let queryCtx = null;              // {t_start, t_end, bbox, label}  (AI-defined window)
let chatTime = null;              // {start, end|null} — dates the user typed in chat
let manualOpts = null;            // FULL Model-options snapshot taken on “Set” —
                                  // nothing in the panel applies until Set is hit
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
  if (nowcastMode) scheduleNowcast();
});

// ---- chat --------------------------------------------------------------
const log = document.getElementById("chat-log");
function addMsg(html, cls = "bot") {
  const d = document.createElement("div");
  d.className = "msg " + cls; d.innerHTML = html; log.appendChild(d);
  log.scrollTop = log.scrollHeight; return d;
}
function statusMsg(gid, text) {           // raw log line — only when AI info is OFF
  // exception: upstream-gauge discovery + speed-run decisions are worth a chat
  // card either way — they change what the run actually computes
  if (aiInfo && /upstream gauge\(s\) inside the domain|speed run/i.test(text)) {
    addMsg(`<b>${gid}</b> · ${text}`, "status");
    return;
  }
  if (!aiInfo) addMsg(`<b>${gid}</b> · ${text}`, "status");
}

// ---- HUC8 basin guide layer ---------------------------------------------
// Zoomed out: HUC8 polygons show WHERE the gauged basins are (fast, clean).
// Zoomed in (>= PIN_ZOOM) inside a basin: that basin's USGS gauge pins appear.
const PIN_ZOOM = 8;
let huc8Layer = null;

async function loadHuc8() {
  try {
    const r = await fetch("/static/data/huc8.geojson");
    if (!r.ok) return;
    huc8Layer = L.geoJSON(await r.json(), {
      renderer: L.canvas({ padding: 0.3 }),
      style: { color: "#3aa3ff", weight: 1, opacity: 0.5, fillColor: "#3aa3ff", fillOpacity: 0.07 },
      onEachFeature: (f, layer) => {
        layer.bindTooltip(
          `🗺 ${f.properties.name} · HUC ${f.properties.huc} · ${(f.properties.gauge_ids || []).length} gauges` +
          `<br><i>click to zoom in</i>`, { sticky: true });
        layer.on("click", () => map.fitBounds(layer.getBounds(), { padding: [30, 30] }));
        layer.on("mouseover", () => layer.setStyle({ weight: 2.5, fillOpacity: 0.18 }));
        layer.on("mouseout", () => huc8Layer.resetStyle(layer));
      },
    });
    updateMapMode();
  } catch (_) { /* guide layer is optional */ }
}

// polygons visible in the current viewport
function visibleHucLayers() {
  const out = [];
  if (!huc8Layer) return out;
  const b = map.getBounds();
  huc8Layer.eachLayer((l) => { if (b.intersects(l.getBounds())) out.push(l); });
  return out;
}

// ray-cast point-in-polygon on GeoJSON coords (outer rings; holes ignored)
function pipGeom(geom, lat, lon) {
  const polys = geom.type === "Polygon" ? [geom.coordinates] :
    geom.type === "MultiPolygon" ? geom.coordinates : [];
  for (const poly of polys) {
    const ring = poly[0];
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1];
      if (((yi > lat) !== (yj > lat)) && (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi)) inside = !inside;
    }
    if (inside) return true;
  }
  return false;
}

function updateMapMode() {
  const zoomedIn = map.getZoom() >= PIN_ZOOM;
  if (huc8Layer) {
    if (zoomedIn) { if (map.hasLayer(huc8Layer)) map.removeLayer(huc8Layer); }
    else if (!map.hasLayer(huc8Layer)) huc8Layer.addTo(map);
  }
  // pins: zoomed-in views, or an AI-located event (its pins stay visible)
  const showPins = zoomedIn || !!queryCtx;
  if (showPins) { if (!map.hasLayer(gaugeLayer)) gaugeLayer.addTo(map); }
  else if (map.hasLayer(gaugeLayer)) map.removeLayer(gaugeLayer);
}

// ---- map-first gauge pins (no AI needed) --------------------------------
let vpTimer = null;
async function loadViewportGauges() {
  if (map.getZoom() < PIN_ZOOM) return;      // zoomed out -> HUC8 guide instead
  const b = map.getBounds();
  try {
    const r = await fetch(`/api/gauges?w=${b.getWest()}&s=${b.getSouth()}&e=${b.getEast()}&n=${b.getNorth()}`);
    if (!r.ok) return;
    const d = await r.json();
    MAX_SIMS = d.max_sims || MAX_SIMS;
    let pins = d.gauge_pins || [];
    if (huc8Layer) {                          // only gauges inside visible HUC8s
      const vis = visibleHucLayers();
      pins = pins.filter((g) => vis.some((l) =>
        l.getBounds().contains([g.lat, g.lon]) && pipGeom(l.feature.geometry, g.lat, g.lon)));
    }
    addGaugePins(pins);
  } catch (_) { /* offline / transient */ }
}
map.on("moveend zoomend", () => {
  updateMapMode();
  clearTimeout(vpTimer); vpTimer = setTimeout(loadViewportGauges, 400);
  if (nowcastMode) scheduleAutoView();
});

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
  // a new location: drop the previous location's gauge selection so it can't
  // pile up past the simulate cap (feedback 2026-07-17)
  selected.clear();
  selKeyAtRun = "";                 // and release any "already simulated" hold
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
  if (nowcastMode) { scheduleNowcast(); return; }   // instant precomputed nowcasts
  if (simHydro[id]) focusGauge(id);        // has results -> show them
}
function selKey() { return [...selected].sort().join(","); }

function refreshSelection() {
  Object.entries(gaugeMarkers).forEach(([id, m]) => m.setStyle(gaugeStyle(id)));
  const n = selected.size;
  document.getElementById("selinfo").textContent = n ? `${n} gauge${n > 1 ? "s" : ""} selected` : "";
  const b = document.getElementById("btn-sim");
  if (nowcastMode) {                       // precomputed — no run to hold/grey
    b.textContent = `⚡ Nowcast (${n})`;
    b.disabled = n === 0 || n > 25;
    if (n > 25) document.getElementById("selinfo").textContent += "  ⚠ max 25 in nowcast";
    return;
  }
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

// timestep = integer + unit (u=minutes, h=hours, d=days — EF5 TIME_STEP syntax)
function readTimestep() {
  const n = Math.max(1, Math.min(60, parseInt(document.getElementById("k-step-n").value, 10) || 1));
  const u = document.getElementById("k-step-u").value;
  return `${n}${u}`;
}
function timestepHours(ts) {
  const m = String(ts).match(/^(\d+)([uhd])$/);
  if (!m) return 1;
  const n = parseInt(m[1], 10);
  return m[2] === "d" ? n * 24 : m[2] === "h" ? n : n / 60;
}

// snapshot of EVERYTHING in the Model options panel (dates + knobs + advanced)
function readPanel() {
  const ov = advancedOverrides();
  return {
    start: document.getElementById("k-start").value || null,
    end: document.getElementById("k-end").value || null,
    hours: parseInt(document.getElementById("k-hours").value) || 48,
    model: document.getElementById("k-model").value,
    snow: document.getElementById("k-snow").value,
    timestep: readTimestep(),
    warmup_days: (() => { const v = parseInt(document.getElementById("k-warmup").value, 10);
                          return Number.isFinite(v) ? v : 90; })(),
    overrides: Object.keys(ov).length ? ov : null,
  };
}

const DEFAULT_OPTS = { start: null, end: null, hours: 48, model: "auto", snow: "auto",
                       timestep: "1h", warmup_days: 90, overrides: null };

function panelDirty() {
  return JSON.stringify(readPanel()) !== JSON.stringify(DEFAULT_OPTS);
}

// options used for a run: the Set snapshot if one is active, else defaults.
// Panel edits WITHOUT hitting Set are intentionally ignored.
function readOptions() {
  const src = manualOpts || DEFAULT_OPTS;
  return { hours: src.hours, model: src.model, snow: src.snow, timestep: src.timestep,
           warmup_days: src.warmup_days, overrides: src.overrides };
}

// ---- time-window resolution -------------------------------------------
// precedence per field: Model-options “Set” override  >  chat-typed dates  >
// AI-identified event window. A field left blank at a higher level falls
// through to the next. No end anywhere -> start + Duration knob.
function resolveWindow(hours) {
  const pick = (k) =>
    manualOpts && manualOpts[k] ? [manualOpts[k], "manual"] :
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
  if (nowcastMode) { showNowcastsFor([...selected]); return; }
  if (simRunning && selKey() === selKeyAtRun) return;   // double-click guard
  const ids = [...selected];
  if (!manualOpts && panelDirty()) {
    addMsg("ℹ️ You changed Model options but didn't hit <b>Set</b> — running with the " +
      "defaults and the AI/chat window. Hit <b>Set</b> in ⚙️ Model options to apply your values.", "status");
  }
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
      body: JSON.stringify({ gauge_ids: ids, t_start: win.tStart, t_end: win.tEnd,
                             label: queryCtx ? queryCtx.label : null,
                             scheme: document.getElementById("run-scheme").value,
                             // a still-running previous job would hold the per-gauge
                             // lock — the server stops it so this run starts now
                             prev_sim_id: localStorage.getItem("lastSimId") || null,
                             ...opt }),
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
  const stepH = timestepHours(opt.timestep);
  lastSim = { tStart: tS, tEnd: tE, hours: HH,
              expectedSteps: Math.max(1, Math.round(HH / stepH)) + 1 };
  ids.forEach((id) => { simHydro[id] = []; gaugeState[id] = "running"; delete gaugeResult[id]; });
  renderTabs();
  if (ids.length && !panelGauge) focusGauge(ids[0]);
  else if (ids.includes(panelGauge)) renderHydro(panelGauge);  // honest "run in
  // progress" placeholder instead of silently keeping the previous run's plot
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
    if (ev.returncode === -9) {                       // user stop / superseded — not an error
      if (aiInfo) setProgress(ev.gauge_id, 100, "stopped ⏹");
      allowResim();
    } else if (ev.returncode != null && ev.returncode !== 0) {
      if (aiInfo) setProgress(ev.gauge_id, 100, "failed ✗");
      explainError("simulation", `EF5 run failed (rc=${ev.returncode}), ` +
        `${ev.n || 0} output rows produced`, ev.gauge_id);
    } else if (aiInfo) setProgress(ev.gauge_id, 100, "complete ✓");
    else addMsg(`✅ <b>${ev.gauge_id}</b> complete (${ev.n} steps)`, "status");
    if (ev.gauge_id === panelGauge) renderHydro(ev.gauge_id);
    if (ev.returncode === 0) fetchNowcast(ev.gauge_id);
  } else if (ev.kind === "params") {
    // effective run parameters — used to pre-fill Model options for manual tuning
    (gaugeResult[ev.gauge_id] = gaugeResult[ev.gauge_id] || {}).runParams = ev;
  } else if (ev.kind === "result") {
    gaugeResult[ev.gauge_id] = { ...(gaugeResult[ev.gauge_id] || {}),
      meta: ev.meta, metrics: ev.metrics, report: ev.report };
    renderTabs();
    if (ev.gauge_id === panelGauge) { renderStats(ev.gauge_id); renderReport(ev.gauge_id); }
    maybeOfferCalibration(ev.gauge_id, ev.metrics);
  } else if (ev.kind === "timeline") {
    gaugeFrames[ev.gauge_id] = { n: ev.n, bounds: ev.bounds };
    if (ev.n - 1 > animMax) { animMax = ev.n - 1; animTimes = ev.times || animTimes; }
    if (ev.vmax) document.getElementById("q-max").textContent =
      `peak ${Math.round(ev.vmax).toLocaleString()} m³/s`;
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
  const stopRow = document.createElement("div"); stopRow.className = "pg-stop";
  stopRow.innerHTML = `<button class="pg-stop-btn" title="Stop this simulation">⏹ Stop</button>`;
  const btn = stopRow.querySelector("button");
  btn.onclick = async () => {
    btn.disabled = true; btn.textContent = "⏹ stopping…";
    // immediate, honest feedback on every unfinished gauge — the workers stop
    // at the next checkpoint (each one flips to "stopped ⏹" as it exits)
    Object.entries(progressEls).forEach(([g2, p]) => {
      if (gaugeState[g2] === "running") p.stage.textContent = "⏹ stopping — finishing the current step…";
    });
    addMsg("⏹ Stop requested — each gauge halts at its next checkpoint (a few seconds; " +
           "up to ~a minute if one is mid-download). You can start a new simulation " +
           "right away — it takes over automatically.", "status");
    try {
      await fetch(`/api/cancel/${currentSim}`, { method: "POST" });
    } catch (_) {
      btn.disabled = false; btn.textContent = "⏹ Stop";   // cancel didn't reach the server
      addMsg("⚠️ Stop request failed to reach the server — try again.", "status");
    }
  };
  progressBox.appendChild(stopRow);
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
  [/speed run: domain cut/i, 16, "⚡ speed-run domain built — observed inflow at the cut gauges"],
  [/speed run (requested|not possible)/i, 16, null],
  [/obs coverage .* stays simulated/i, 15, null],
  [/assimilating observed flow/i, 26, null],
  [/derived from the DEM/i, 15, "flow network rebuilt from the DEM (pysheds)"],
  [/SNOW17 enabled/i,    17, "snow module ON (cold basin detected)"],
  [/no snow/i,           17, "snow module off (warm basin)"],
  [/stored best parameters/i, 18, "loading this basin's best-known parameters"],
  [/-day warm-up from/i, 20, null],       // plan only — the run itself comes later
  [/short warm-up/i,     20, null],
  [/warm-up disabled/i,  20, "cold start (no warm-up)"],
  [/warm start from exact/i, 22, "warm-starting from a saved model state"],
  [/USGS observed/i,     24, "fetching observed discharge (USGS)…"],
  [/preparing rainfall/i, 28, "preparing rainfall forcing (MRMS) from the archive…"],
  [/forcing store: reused/i, 34, null],
  [/preparing PET/i,     36, "preparing PET forcing…"],
  [/preparing temperature/i, 38, "preparing temperature forcing (snow)…"],
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

// ---- AI error interpreter: backend error -> plain-words chat message -------
const explained = new Set();          // one explanation per (source, gauge)
async function explainError(source, rawError, gid) {
  const key = `${source}:${gid}`;
  if (explained.has(key)) return;
  explained.add(key);
  const card = addMsg(`🧯 <b>${gid}</b> — the ${source} hit an error. <i>Interpreting…</i>`, "bot");
  let text = null;
  try {
    const r = await fetch("/api/explain", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ error: String(rawError), where: source,
                             context: { gauge: gid, window: lastSim } }),
    });
    text = (await r.json()).text;
  } catch (_) {}
  card.innerHTML = `🧯 <b>${gid}</b> — ` + (text ? mdLite(text)
    : `the ${source} failed with: <code>${escapeHtml(String(rawError).slice(0, 200))}</code>. ` +
      `This is often a temporary data-read hiccup — trying again usually works. ` +
      `If it keeps failing, report it with 💡 Feedback.`);
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
    sessionStorage.setItem("expertOk", "1");   // user chose manual tuning
    const lp = document.getElementById("left-panel");
    lp.classList.remove("collapsed");
    document.getElementById("adv-body").classList.remove("hidden");
    document.getElementById("adv-arrow").textContent = "▾";
    const pf = prefillModelOptions(gid);
    addMsg("🛠 Opened <b>Model options</b> pre-filled with this run's setup — " +
      `window <b>${document.getElementById("k-start").value || "?"} → ` +
      `${document.getElementById("k-end").value || "?"}</b>` +
      (pf.model ? `, model <b>${pf.model.toUpperCase()}</b>` : "") +
      (pf.n ? `, <b>${pf.n}</b> parameter values (${pf.src || "a-priori"})` : "") +
      ". That's your starting point — adjust, hit <b>Set</b>, then Simulate again; " +
      "if your run beats the stored NSE, the parameters are saved for this basin automatically.", "bot");
  };
}

// pre-fill the Model options panel from the last run's effective setup, so
// manual tuning starts from what the AI actually used (window, model, params)
function prefillModelOptions(gid) {
  const rp = (gaugeResult[gid] || {}).runParams || {};
  if (lastSim && lastSim.tStart) {
    document.getElementById("k-start").value = lastSim.tStart.slice(0, 10);
    if (lastSim.tEnd) document.getElementById("k-end").value = lastSim.tEnd.slice(0, 10);
    if (lastSim.hours) document.getElementById("k-hours").value = lastSim.hours;
  }
  const meta = (gaugeResult[gid] || {}).meta || {};
  const model = rp.model || meta.model;
  if (["crestphys", "crest", "hp"].includes(model))
    document.getElementById("k-model").value = model;   // enables the param inputs
  updateParamAvailability();
  let n = 0;
  const vals = { ...(rp.wb || {}), ...(rp.kw || {}) };
  Object.entries(vals).forEach(([k, v]) => {
    const inp = document.getElementById("adv-" + String(k).toLowerCase());
    if (inp && typeof v === "number" && isFinite(v)) {
      inp.value = +v.toPrecision(4);
      n++;
    }
  });
  return { n, model, src: rp.source };
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
      if (ev.msg.startsWith("📉"))       // extended-search kick-in: full note in chat
        addMsg(`<b>${gid}</b> · ${escapeHtml(ev.msg)}`, "status");
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
      if (ev.error) {
        stage.textContent = "failed ✗";
        explainError("calibration", ev.error, gid);
        return;
      }
      stage.textContent = `done — NSE ${ev.baseline_nse} → ${ev.best_nse}`;
      addMsg(`🎯 Calibration finished for <b>${gid}</b>: NSE <b>${ev.baseline_nse}</b> → <b>${ev.best_nse}</b>` +
        (ev.saved ? " — saved as this basin's best parameter set (it will be used automatically from now on)."
                  : " — did not beat the stored parameters, keeping the previous set.") +
        ` Re-running the simulation now with the ${ev.saved ? "new" : "stored"} parameters ` +
        `(2-D map included)…`, "bot");
      allowResim();                   // release the same-selection hold, then re-run
      simulate();
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
  clearTimestep();                  // stale readout/marker from the previous run
}

function showAnim() {
  const bar = document.getElementById("anim");
  bar.classList.remove("hidden");
  document.getElementById("anim-slider").max = String(animMax);
}

// scrubber label: the frame's real datetime; falls back to a time computed
// from the run window, and only then to a bare step number
function simTimeAt(i) {
  if (!lastSim || !lastSim.tStart) return null;
  const steps = Math.max(1, (lastSim.expectedSteps || 1) - 1);
  const ms = new Date(lastSim.tStart + "Z").getTime() + i * (lastSim.hours * 3600e3) / steps;
  return new Date(ms).toISOString().slice(0, 16).replace("T", " ");
}
function frameLabel(i) {
  return animTimes[i] || simTimeAt(i) || `#${i + 1}`;
}

function setFrame(idx, opts) {
  animIdx = Math.max(0, Math.min(animMax, idx | 0));
  document.getElementById("anim-slider").value = String(animIdx);
  const label = frameLabel(animIdx);
  document.getElementById("anim-time").textContent = label;
  Object.entries(gaugeFrames).forEach(([gid, info]) => {
    const i = Math.min(animIdx, info.n - 1);
    const url = `/api/frame/${currentSim}/${gid}/${i}.png`;
    if (overlays[gid]) { overlays[gid].setUrl(url); if (info.bounds) overlays[gid].setBounds(info.bounds); }
    else if (info.bounds) addOverlay(gid, url, info.bounds);
  });
  // the hydrograph mirrors the map: scrubbing moves the marker + readout too
  if (!(opts && opts.keepMarker) && label[0] !== "#" && panelGauge) {
    hydroSelTime = label;
    updateHydroMarker(label);
    renderReadout(panelGauge, label);
  }
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
    t.innerHTML = `<span class="dot"></span>${id}<span class="x" title="Close & unselect this gauge">✕</span>`;
    t.onclick = () => focusGauge(id);
    t.querySelector(".x").onclick = (e) => { e.stopPropagation(); closeSimTab(id); };
    bar.appendChild(t);
  });
}

function closeSimTab(id) {
  selected.delete(id);
  delete simHydro[id]; delete gaugeState[id]; delete gaugeResult[id];
  if (overlays[id]) { try { q2dGroup.removeLayer(overlays[id]); } catch (_) {} delete overlays[id]; }
  delete gaugeFrames[id];
  refreshSelection();
  const rest = Object.keys(simHydro);
  if (panelGauge === id) {
    if (rest.length) { focusGauge(rest[0]); return; }
    panelGauge = null;
    document.getElementById("right-panel").classList.add("hidden");
    document.getElementById("rp-reopen").classList.add("hidden");
    return;
  }
  renderTabs();
}

function focusGauge(id) {
  if (panelGauge !== id) clearTimestep();      // readout/marker belong to the old gauge
  panelGauge = id;
  nowcastPanelActive = false;
  const g = gaugeData[id];
  document.getElementById("right-panel").classList.remove("hidden");
  document.getElementById("rp-reopen").classList.add("hidden");
  document.getElementById("rp-title").textContent = `${id} · ${g ? g.name : ""}`;
  renderTabs();
  renderFavBtn();
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
    const h = document.createElement("div"); h.className = "rp-h";
    h.textContent = "📄 Report";
    // full report: the AQUAH report-writer agent builds a publication-style
    // PDF (figures + interpretation) on the server; first click takes ~1 min
    const dl = document.createElement("button");
    dl.className = "rp-dl";
    dl.textContent = "⬇ PDF report";
    dl.title = "Generate + download the full report (AI-written, with figures). " +
               "The first request takes about a minute.";
    dl.onclick = () => downloadReport(id, dl);
    h.appendChild(dl);
    const p = document.createElement("div"); p.textContent = res.report;
    el.append(h, p);
  }
}

async function downloadReport(gid, btn) {
  if (!currentSim) return;
  const orig = btn.textContent;
  btn.disabled = true;
  btn.textContent = "⏳ writing report…";
  try {
    const r = await fetch(`/api/report/${currentSim}/${gid}`);
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.error || r.statusText);
    }
    const blob = await r.blob();
    const cd = r.headers.get("Content-Disposition") || "";
    const m = /filename="?([^";]+)"?/.exec(cd);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = m ? m[1] : `CREST_report_${gid}.pdf`;
    a.click();
    URL.revokeObjectURL(a.href);
    if (r.headers.get("X-Report-Saved") === "1") {
      addMsg(`📄 Report for <b>${gid}</b> saved to your account — find it any time ` +
             `under 👤 → <b>My reports</b>.`, "status");
    } else if (!userSignedIn) {
      addMsg(`📄 Report downloaded. <a href='/login'>Sign in</a> to keep reports in ` +
             `your account — as a guest, generated reports are discarded when you ` +
             `close the app.`, "status");
    }
    btn.textContent = "✓ downloaded";
    setTimeout(() => { btn.textContent = orig; btn.disabled = false; }, 4000);
  } catch (e) {
    btn.textContent = orig;
    btn.disabled = false;
    addMsg(`⚠️ Report for <b>${gid}</b>: ${escapeHtml(e.message)}`, "status");
  }
}

let hydroSelTime = null;            // clicked timestep (marker + readout + 2-D sync)

function _hydroMarker(t) {
  return { type: "line", x0: t, x1: t, y0: 0, y1: 1, yref: "paper",
           line: { color: "#ffd23f", width: 1.4, dash: "dot" } };
}

function _hydroFig(rows, big, nc) {
  const x = rows.map((r) => r.time), sim = rows.map((r) => r.sim_q),
    obs = rows.map((r) => r.obs_q), pr = rows.map((r) => r.precip || 0);
  const maxp = Math.max(0.1, ...pr);
  const traces = [
    { x, y: pr, name: "Precip", type: "bar", marker: { color: "#5b9bd5" }, yaxis: "y2", opacity: 0.7 },
    { x, y: obs, name: "Obs Q", mode: "lines",
      line: { color: "#f4f4f4", width: big ? 1.6 : 1.3, shape: "spline", smoothing: 0.8 } },
    { x, y: sim, name: "Sim Q", mode: "lines",
      line: { color: "#4cc9a0", width: big ? 2.2 : 1.8, shape: "spline", smoothing: 0.8 } },
  ];
  if (nc && nc.ok && nc.times && nc.times.length) {
    const last = rows[rows.length - 1];             // anchor for a continuous tail
    traces.push({ x: [last.time, ...nc.times], y: [last.sim_q, ...nc.q],
      name: "🔮 AI nowcast", mode: "lines",
      line: { color: "#ff9f43", width: big ? 2.2 : 1.8, dash: "dot" } });
  }
  const layout = {
    margin: big ? { l: 56, r: 56, t: 18, b: 40 } : { l: 46, r: 46, t: 12, b: 30 },
    bargap: 0, showlegend: true,
    legend: { orientation: "h", y: big ? 1.08 : 1.18, font: { size: big ? 11 : 9 } },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#cdd9e2", size: big ? 12 : 10 },
    xaxis: { gridcolor: "rgba(255,255,255,.06)" },
    yaxis: { title: "Q m³/s", rangemode: "tozero", gridcolor: "rgba(255,255,255,.06)" },
    yaxis2: { overlaying: "y", side: "right", range: [maxp * 3.4, 0], showgrid: false },
    shapes: hydroSelTime ? [_hydroMarker(hydroSelTime)] : [],
    hovermode: "x",
  };
  if (!big) layout.height = 250;
  return { traces, layout };
}

function _bindHydroClick(el, id) {
  if (!el.on) return;                            // Plotly attaches .on after first plot
  if (el.removeAllListeners) el.removeAllListeners("plotly_click");
  el.on("plotly_click", (ev) => {
    const p = ev.points && ev.points[0];
    if (p) selectTimestep(id, String(p.x));
  });
}

// ---- AI nowcast tail (CREST_nowcast Space) --------------------------------
async function fetchNowcast(gid) {
  if (!currentSim) return;
  try {
    const d = await (await fetch(`/api/nowcast/${currentSim}/${gid}`)).json();
    if (!d.ok) return;                       // not trained / Space asleep — no tail
    (gaugeResult[gid] = gaugeResult[gid] || {}).nowcast = d;
    if (gid === panelGauge) renderHydro(gid);
    const m = d.model || {};
    addMsg(`🔮 <b>${gid}</b>: AI nowcast added to the hydrograph — next ` +
      `${d.q.length} h from the DI-LSTM (experimental${m.val_nse != null
        ? `, val NSE ${m.val_nse}` : ""}). It fuses the latest USGS ` +
      `observation with MRMS rainfall.`, "status");
  } catch (_) { /* no tail — never break the results panel */ }
}

function renderHydro(id) {
  const rows = simHydro[id] || [];
  const el = document.getElementById("rp-hydro");
  const xp = document.getElementById("rp-expand");
  if (!rows.length) {
    if (el.data) { try { Plotly.purge(el); } catch (_) {} }   // don't gut a live plot
    el.innerHTML = gaugeState[id] === "running"
      ? '<div class="muted">⏳ new run in progress — the hydrograph streams in live once ' +
        'the model starts producing output (the warm-up runs first).</div>'
      : '<div class="muted">Select a gauge and run a simulation to see its hydrograph.</div>';
    xp.classList.add("hidden");
    return;
  }
  if (el.querySelector(".muted")) el.innerHTML = "";     // drop placeholder before plotting
  const { traces, layout } = _hydroFig(rows, false,
    gaugeResult[id] && gaugeResult[id].nowcast);
  Plotly.react(el, traces, layout, { displayModeBar: false, responsive: true });
  _bindHydroClick(el, id);
  xp.classList.remove("hidden");
  if (hydroModalOpen) renderHydroBig();                  // keep the big view live
}

// ---- clicked-timestep readout + hydrograph <-> 2-D map sync ---------------
const READOUT_FIELDS = [        // row key -> label, unit, decimals
  ["sim_q", "Sim Q", "m³/s", 1], ["obs_q", "Obs Q", "m³/s", 1],
  ["precip", "Precip", "mm/h", 2], ["pet", "PET", "mm", 2],
  ["sm", "Soil moisture", "%", 1], ["gw", "Groundwater", "mm", 1],
  ["temp", "Air temp", "°C", 1], ["swe", "SWE", "mm", 1],
  ["fast", "Fast flow", "", 3], ["slow", "Slow flow", "", 3], ["base", "Base flow", "", 3],
];

function _rowAt(id, t) {
  const rows = simHydro[id] || [];
  if (!rows.length) return null;
  const tgt = Date.parse(String(t).replace(" ", "T") + "Z");
  let best = null, bd = Infinity;
  for (const r of rows) {
    const d = Math.abs(Date.parse(String(r.time).replace(" ", "T") + "Z") - tgt);
    if (d < bd) { bd = d; best = r; }
  }
  return best;
}

function renderReadout(id, t) {
  const row = _rowAt(id, t);
  ["rp-readout", "hm-readout"].forEach((eid) => {
    const el = document.getElementById(eid);
    if (!row) { el.classList.add("hidden"); return; }
    const cells = [`<span class="ro t">at<b>${row.time}</b></span>`];
    READOUT_FIELDS.forEach(([k, label, unit, dec]) => {
      const v = row[k];
      if (v == null || !isFinite(v)) return;
      cells.push(`<span class="ro">${label}<b>${(+v).toFixed(dec)}${unit ? " " + unit : ""}</b></span>`);
    });
    el.innerHTML = cells.join("");
    el.classList.remove("hidden");
  });
}

function updateHydroMarker(t) {
  const shapes = t ? [_hydroMarker(t)] : [];
  ["rp-hydro", "hm-plot"].forEach((eid) => {
    const el = document.getElementById(eid);
    if (el && el.data) { try { Plotly.relayout(el, { shapes }); } catch (_) {} }
  });
}

function selectTimestep(id, t) {
  hydroSelTime = t;
  renderReadout(id, t);
  updateHydroMarker(t);
  jumpAnimTo(t);                    // 2-D map + scrubber follow the click
}

function jumpAnimTo(t) {
  if (animMax <= 0) return;
  const tgt = Date.parse(String(t).replace(" ", "T") + "Z");
  if (!isFinite(tgt)) return;
  let best = -1, bd = Infinity;
  for (let i = 0; i <= animMax; i++) {
    const lbl = frameLabel(i);
    if (!lbl || lbl[0] === "#") continue;
    const d = Math.abs(Date.parse(lbl.replace(" ", "T") + "Z") - tgt);
    if (d < bd) { bd = d; best = i; }
  }
  if (best >= 0) { stopPlay(); setFrame(best, { keepMarker: true }); }
}

function clearTimestep() {
  hydroSelTime = null;
  ["rp-readout", "hm-readout"].forEach((eid) => document.getElementById(eid).classList.add("hidden"));
  updateHydroMarker(null);
}

// ---- enlarged hydrograph modal --------------------------------------------
let hydroModalOpen = false;
let hmZoom = 1;                     // canvas magnification (1x..5x)
let hmBaseW = 0;                    // scroll-container width at 1x (scrollbar-free)
const HM_ZMIN = 1, HM_ZMAX = 5, HM_ZSTEP = 0.5;

function _hmBaseH() { return Math.min(Math.round(window.innerHeight * 0.56), 520); }

function applyHmZoom() {
  const el = document.getElementById("hm-plot");
  if (!el.data) return;
  el.style.width = Math.round(hmBaseW * hmZoom) + "px";
  el.style.height = Math.round(_hmBaseH() * Math.max(1, (hmZoom + 1) / 2)) + "px";
  try { Plotly.Plots.resize(el); } catch (_) {}   // grow the canvas -> #hm-scroll
  document.getElementById("hm-zlvl").textContent = Math.round(hmZoom * 100) + "%";
  document.getElementById("hm-zout").disabled = hmZoom <= HM_ZMIN;
  document.getElementById("hm-zin").disabled = hmZoom >= HM_ZMAX;
}

function hmZoomBy(delta) {
  hmZoom = Math.min(HM_ZMAX, Math.max(HM_ZMIN, Math.round((hmZoom + delta) * 2) / 2));
  applyHmZoom();
}

function renderHydroBig() {
  const nc = nowcastPanelActive && nowcastRes && nowcastRes.gauges[panelGauge];
  const rows = simHydro[panelGauge] || [];
  if (!nc && !rows.length) return;
  const el = document.getElementById("hm-plot");
  if (!hmBaseW) hmBaseW = document.getElementById("hm-scroll").clientWidth - 2;
  el.style.width = Math.round(hmBaseW * hmZoom) + "px";
  el.style.height = Math.round(_hmBaseH() * Math.max(1, (hmZoom + 1) / 2)) + "px";
  const { traces, layout } = nc ? _nowcastFig(panelGauge, true)
    : _hydroFig(rows, true, gaugeResult[panelGauge] && gaugeResult[panelGauge].nowcast);
  Plotly.react(el, traces, layout, { displayModeBar: false, responsive: true,
                                     scrollZoom: true, doubleClick: "reset" });
  if (!nc) _bindHydroClick(el, panelGauge);
  applyHmZoom();
  if (!nc && hydroSelTime) renderReadout(panelGauge, hydroSelTime);
}

function openHydroModal() {
  const nc = nowcastPanelActive && nowcastRes && nowcastRes.gauges[panelGauge];
  if (!panelGauge || (!nc && !(simHydro[panelGauge] || []).length)) return;
  hydroModalOpen = true;
  hmZoom = 1;                                   // start at 100% each time
  hmBaseW = 0;                                  // re-measure (window may have resized)
  const g = gaugeData[panelGauge];
  document.getElementById("hm-title").textContent =
    `${nc ? "⚡" : "📈"} ${panelGauge}${g ? " · " + g.name : ""}`;
  document.getElementById("hydro-modal").classList.remove("hidden");
  renderHydroBig();
}

function closeHydroModal() {
  hydroModalOpen = false;
  document.getElementById("hydro-modal").classList.add("hidden");
}

document.getElementById("rp-expand").onclick = openHydroModal;
document.getElementById("hm-zin").onclick = () => hmZoomBy(+HM_ZSTEP);
document.getElementById("hm-zout").onclick = () => hmZoomBy(-HM_ZSTEP);
document.getElementById("hm-reset").onclick = () => {
  hmZoom = 1;                 // undo canvas magnification AND any axis drag/scroll zoom
  renderHydroBig();           // fresh layout -> original full time range
};
document.getElementById("hm-close").onclick = closeHydroModal;
document.getElementById("hydro-modal").addEventListener("click", (e) => {
  if (e.target.id === "hydro-modal") closeHydroModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && hydroModalOpen) closeHydroModal();
});

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
    // map-first signals: a zoomed-in map with pins IS a location choice — the
    // agent must not keep asking "where" (signed-in users open pre-zoomed)
    gauges_on_map: Object.keys(gaugeMarkers).length,
    map_zoomed_in: map.getZoom() >= PIN_ZOOM,
    signed_in: userSignedIn,
    sim_running: simRunning,
    last_window: lastSim ? { start: lastSim.tStart, end: lastSim.tEnd } : null,
    results: Object.entries(gaugeResult).map(([id, r]) => ({
      gauge: id, name: gaugeData[id] ? gaugeData[id].name : null,
      nse: r.metrics ? r.metrics.nsce : null,
      peak_sim: r.metrics ? r.metrics.peak_sim : null,
    })),
    manual_options_override: manualOpts, chat_time: chatTime,
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
    userSignedIn = true;
    loadFavorites();                  // registered-user benefit: focused basins
    zoomToUserLocation();             // registered-user benefit: open at home
  } catch (_) {}
}

// ---- registered-user benefit: focused basins (up to 5 favorite gauges) -----
// Favorites live in the account profile and show up as gold ★ pins at every
// zoom level — one click zooms in and selects the gauge, no HUC8 digging.
let userSignedIn = false;
let userFavs = [];                 // [{id,name,lat,lon,area_km2}]
let favMax = 5;
let favLayer = null;

async function loadFavorites() {
  try {
    const r = await fetch("/api/favorites");
    if (!r.ok) return;
    const d = await r.json();
    userFavs = d.favorites || [];
    favMax = d.max || 5;
    renderFavorites();
  } catch (_) {}
}

async function setFavorite(gid, add) {
  const r = await fetch("/api/favorites", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gauge_id: gid, action: add ? "add" : "remove" }) });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.error || r.statusText);
  userFavs = d.favorites || [];
  renderFavorites();
}

function isFav(id) { return userFavs.some((g) => g.id === id); }

function renderFavorites() {          // pins + panel star + profile list together
  if (!favLayer) favLayer = L.layerGroup().addTo(map);
  favLayer.clearLayers();
  userFavs.forEach((g) => {
    gaugeData[g.id] = gaugeData[g.id] || g;   // selectable before viewport pins load
    L.marker([g.lat, g.lon], {
      icon: L.divIcon({ className: "fav-pin", html: "★", iconSize: [22, 22], iconAnchor: [11, 11] }),
    }).bindTooltip(`⭐ ${g.id} · ${g.name}<br>${Math.round(g.area_km2).toLocaleString()} km²`,
                   { direction: "top" })
      .on("click", () => {
        if (map.getZoom() < PIN_ZOOM)         // zoom in enough for the pin layer
          map.setView([g.lat, g.lon], PIN_ZOOM + 1, { animate: false });
        toggleGauge(g.id);
      })
      .addTo(favLayer);
  });
  renderFavBtn();
  renderFavList();
}

function renderFavBtn() {
  const b = document.getElementById("rp-fav");
  if (!panelGauge) { b.classList.add("hidden"); return; }
  b.classList.remove("hidden");
  const on = isFav(panelGauge);
  b.textContent = on ? "★" : "☆";
  b.classList.toggle("on", on);
  b.title = on ? "Remove from your focused basins"
    : userSignedIn ? `Add to your focused basins (${userFavs.length}/${favMax})`
    : "Sign in to save favorite gauges";
}

document.getElementById("rp-fav").onclick = async () => {
  if (!panelGauge) return;
  if (!userSignedIn) {
    addMsg("⭐ <a href='/login'>Sign in</a> to save favorite gauges — your focused " +
           "basins appear as gold stars on the map every time you open the app.", "status");
    return;
  }
  const add = !isFav(panelGauge);
  try {
    await setFavorite(panelGauge, add);
    addMsg(add
      ? `⭐ Added <b>${panelGauge}</b> to your focused basins (${userFavs.length}/${favMax}) — ` +
        `it'll be starred on your map whenever you sign in.`
      : `☆ Removed <b>${panelGauge}</b> from your focused basins.`, "status");
  } catch (e) { addMsg("⚠️ " + e.message, "status"); }
};

function renderFavList() {            // profile modal: list + remove + show-on-map
  const el = document.getElementById("pf-favs");
  if (!userFavs.length) {
    el.innerHTML = '<i class="pm-sub">none yet — run a simulation and hit ☆, or add one below</i>';
    return;
  }
  el.innerHTML = "";
  userFavs.forEach((g) => {
    const row = document.createElement("div");
    row.className = "hist-row";
    row.innerHTML =
      `<div class="hist-info"><b>★ ${g.id}</b> · ${escapeHtml(g.name)}<br>` +
      `<span class="pm-sub">${Math.round(g.area_km2).toLocaleString()} km²</span></div>` +
      `<button class="hist-load fav-go">Show</button>` +
      `<button class="hist-load fav-del" title="Remove from focused basins">✕</button>`;
    row.querySelector(".fav-go").onclick = () => {
      document.getElementById("profile-modal").classList.add("hidden");
      map.setView([g.lat, g.lon], PIN_ZOOM + 1, { animate: false });
    };
    row.querySelector(".fav-del").onclick = async () => {
      try { await setFavorite(g.id, false); } catch (_) {}
    };
    el.appendChild(row);
  });
}

document.getElementById("pf-fav-add").onclick = async () => {
  const inp = document.getElementById("pf-fav-input");
  const st = document.getElementById("pf-fav-status");
  const v = inp.value.trim();
  if (!v) return;
  st.textContent = "…";
  try {
    await setFavorite(v, true);
    inp.value = "";
    st.textContent = "✓ added";
  } catch (e) { st.textContent = "⚠ " + e.message; }
  setTimeout(() => { st.textContent = ""; }, 5000);
};
document.getElementById("pf-fav-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("pf-fav-add").click();
});

// ---- registered-user benefit: auto-zoom to the user's location ------------
// On open, signed-in users land on their own area with the nearby USGS gauge
// pins already visible — no HUC8 clicking needed. Falls back silently to the
// HUC8 guide flow when the browser denies/lacks geolocation (also the case
// inside HF's Space iframe unless it grants the geolocation permission —
// the direct *.hf.space URL always prompts normally).
const HOME_VIEW = { lat: 39, lon: -98, zoom: 5 };   // the initial setView above
function mapUntouched() {
  const c = map.getCenter();
  return map.getZoom() === HOME_VIEW.zoom &&
         Math.abs(c.lat - HOME_VIEW.lat) < 0.01 && Math.abs(c.lng - HOME_VIEW.lon) < 0.01;
}
function zoomToUserLocation() {
  if (!("geolocation" in navigator)) return;
  navigator.geolocation.getCurrentPosition((pos) => {
    const lat = pos.coords.latitude, lon = pos.coords.longitude;
    // don't fight a restored/running simulation or a map the user already moved
    if (currentSim || simRunning || !mapUntouched()) return;
    if (lat < 24 || lat > 50 || lon < -125 || lon > -66) return;  // outside CONUS
    // no animation: at app open a direct jump beats a cross-country fly-in
    // (and animated zooms can stall in throttled/background tabs)
    map.setView([lat, lon], PIN_ZOOM + 1, { animate: false });  // >= PIN_ZOOM -> pins load
    L.circleMarker([lat, lon], { radius: 7, color: "#ffd23f", weight: 2,
                                 fillColor: "#ffd23f", fillOpacity: 0.45, pane: "markerPane" })
      .bindTooltip("📍 you are here").addTo(map);
    addMsg("📍 Welcome back — zoomed to your area, with the USGS gauges around you " +
           "on the map. Click a pin to select it, or just describe a flood event.", "status");
  }, () => { /* denied / unavailable -> normal HUC8 guide flow */ },
  { maximumAge: 600000, timeout: 8000 });
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
  renderFavList();
  loadMyReports();
  loadHistory();
}

// ---- registered-user benefit: per-account simulation history --------------
async function loadHistory() {
  const el = document.getElementById("pf-history");
  try {
    const r = await fetch("/api/history");
    if (!r.ok) { el.innerHTML = '<i class="pm-sub">sign in to keep a history</i>'; return; }
    const hist = (await r.json()).history || [];
    if (!hist.length) { el.innerHTML = '<i class="pm-sub">no simulations yet</i>'; return; }
    el.innerHTML = "";
    hist.forEach((h) => {
      const row = document.createElement("div");
      row.className = "hist-row";
      const badge = h.status === "running" ? "🟢 running" : h.status === "done" ? "✓ done" : "♻ cached";
      row.innerHTML =
        `<div class="hist-info"><b>${escapeHtml(h.label || h.gauge_ids.join(", "))}</b><br>` +
        `<span class="pm-sub">${h.gauge_ids.length} gauge(s) · ${h.t_start.slice(0, 10)} → ` +
        `${h.t_end.slice(0, 10)} · ${h.when || ""} · ${badge}</span></div>` +
        `<button class="hist-load">Open</button>`;
      row.querySelector(".hist-load").onclick = () => restoreFromHistory(h);
      el.appendChild(row);
    });
  } catch (_) {
    el.innerHTML = '<i class="pm-sub">history unavailable</i>';
  }
}

function restoreFromHistory(h) {
  document.getElementById("profile-modal").classList.add("hidden");
  chatTime = { start: h.t_start.slice(0, 19), end: h.t_end.slice(0, 19) };
  selected.clear();
  h.gauge_ids.forEach((id) => selected.add(id));
  refreshSelection();
  if (h.status === "expired") {
    // job no longer in RAM (e.g. server restarted) — re-run; the result cache
    // + frame cache serve it back almost instantly
    addMsg(`🔁 Restoring <b>${escapeHtml(h.label || h.gauge_ids.join(", "))}</b> — ` +
      `re-running from the result cache…`, "status");
    allowResim();
    simulate();
  } else {
    addMsg(`🔁 Reopening <b>${escapeHtml(h.label || h.gauge_ids.join(", "))}</b>…`, "status");
    localStorage.setItem("lastSimId", h.sim_id);
    reattach(h.sim_id);
  }
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

// ---- registered-user benefit: persistent report library --------------------
async function loadMyReports() {
  const el = document.getElementById("pf-reports");
  try {
    const r = await fetch("/api/myreports");
    if (!r.ok) { el.innerHTML = '<i class="pm-sub">sign in to keep reports</i>'; return; }
    const reps = (await r.json()).reports || [];
    if (!reps.length) {
      el.innerHTML = '<i class="pm-sub">none yet — run a simulation and hit ⬇ PDF report</i>';
      return;
    }
    el.innerHTML = "";
    reps.forEach((rep) => {
      const row = document.createElement("div");
      row.className = "hist-row";
      row.innerHTML =
        `<div class="hist-info"><b>📄 ${escapeHtml(rep.name)}</b><br>` +
        `<span class="pm-sub">${rep.when} · ${rep.kb} KB</span></div>` +
        `<a class="hist-load" href="/api/myreports/${encodeURIComponent(rep.name)}" ` +
        `download>Download</a>` +
        `<button class="hist-load fav-del" title="Delete this report">✕</button>`;
      row.querySelector(".fav-del").onclick = async () => {
        await fetch(`/api/myreports/${encodeURIComponent(rep.name)}`, { method: "DELETE" });
        loadMyReports();
      };
      el.appendChild(row);
    });
  } catch (_) {
    el.innerHTML = '<i class="pm-sub">reports unavailable</i>';
  }
}

// guests: generated reports are ephemeral — tell the server to drop them when
// the app closes (registered users' saved copies are kept server-side)
window.addEventListener("pagehide", () => {
  if (!userSignedIn && currentSim && navigator.sendBeacon)
    navigator.sendBeacon(`/api/report_discard/${currentSim}`);
});

// ---- display font size (everything except the top bar) ---------------------
const FONT_MODES = [["", "🔠 A", "normal"], ["font-lg", "🔠 A+", "bigger"],
                    ["font-xl", "🔠 A++", "much bigger"]];
let fontIdx = Math.max(0, FONT_MODES.findIndex(([c]) => c === (localStorage.getItem("fontSize") || "")));
function applyFont() {
  document.body.classList.remove("font-lg", "font-xl");
  const [cls, label] = FONT_MODES[fontIdx];
  if (cls) document.body.classList.add(cls);
  document.getElementById("font-btn").textContent = label;
  localStorage.setItem("fontSize", cls);
}
document.getElementById("font-btn").onclick = () => {
  fontIdx = (fontIdx + 1) % FONT_MODES.length;
  applyFont();
};
applyFont();

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

// run scheme (🏞 full / ⚡ speed) — persisted; changing it makes a new run differ
const schemeSel = document.getElementById("run-scheme");
schemeSel.value = localStorage.getItem("runScheme") || "full";
schemeSel.onchange = () => {
  localStorage.setItem("runScheme", schemeSel.value);
  addMsg(schemeSel.value === "speed"
    ? "⚡ <b>Speed run</b> — the basin is cut at upstream USGS gauges and their observed " +
      "flow is injected as boundary conditions. Much faster on big rivers; needs " +
      "near-complete observations (otherwise it falls back to the full basin)."
    : "🏞 <b>Full run</b> — the whole basin is simulated (required for forecasts, " +
      "where no future observations exist).", "status");
  allowResim();
};

// Model options is an expert panel — confirm once per session before opening
function expertGateOk() { return sessionStorage.getItem("expertOk") === "1"; }
function openExpertGate(onYes) {
  const m = document.getElementById("expert-modal");
  m.classList.remove("hidden");
  document.getElementById("xp-yes").onclick = () => {
    sessionStorage.setItem("expertOk", "1");
    m.classList.add("hidden");
    onYes();
  };
  const deny = () => {
    m.classList.add("hidden");
    addMsg("👍 No problem — just tell me about the flood you're interested in " +
      "(a place, a date, or a news link) and I'll set everything up for you.", "bot");
  };
  document.getElementById("xp-no").onclick = deny;
  document.getElementById("xp-close").onclick = deny;
}

document.querySelectorAll(".panel-head .toggle, .panel-head .close").forEach((btn) => {
  btn.onclick = () => {
    const p = document.getElementById(btn.dataset.target);
    if (btn.classList.contains("close")) {
      p.classList.add("hidden");
      // the right panel has no header toggle to bring it back — offer a reopen chip
      if (p.id === "right-panel" && panelGauge)
        document.getElementById("rp-reopen").classList.remove("hidden");
      return;
    }
    const opening = p.classList.contains("collapsed");
    if (p.id === "left-panel" && opening && !expertGateOk()) {
      openExpertGate(() => p.classList.remove("collapsed"));
      return;
    }
    p.classList.toggle("collapsed");
  };
});
document.getElementById("rp-reopen").onclick = () => {
  document.getElementById("rp-reopen").classList.add("hidden");
  if (nowcastMode) {                 // nowcast results are precomputed — always restorable
    const have = nowcastRes ? Object.keys(nowcastRes.gauges) : [];
    if (nowcastRes && nowcastRes.gauges[panelGauge]) focusNowcastGauge(panelGauge);
    else if (have.length) focusNowcastGauge(have[0]);
    else {
      document.getElementById("right-panel").classList.remove("hidden");
      if (selected.size) scheduleNowcast(); else scheduleAutoView();
    }
    return;
  }
  if (panelGauge) focusGauge(panelGauge);
  else document.getElementById("right-panel").classList.remove("hidden");
};
// ---- advanced parameters (all model/routing/snow params) ---------------
const PARAM_GROUPS = [
  { title: "Water balance — CREST / CRESTPHYS",
    keys: ["wm", "b", "im", "ke", "fc", "iwu", "igw", "hmaxaq", "gwc", "gwe"] },
  { title: "Routing — kinematic wave",
    keys: ["under", "leaki", "th", "isu", "alpha", "beta", "alpha0"] },
  { title: "Snow — SNOW17",
    keys: ["uadj", "mbase", "mfmax", "mfmin", "tipm", "nmf", "plwhc", "scf", "pxtemp"] },
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
      inp.type = "number"; inp.step = "0.01"; inp.placeholder = "auto";
      inp.id = "adv-" + k; inp.dataset.param = k;
      l.appendChild(inp); grid.appendChild(l);
    });
    body.appendChild(grid);
  });
}
buildAdvanced();

// collect only the overridden (non-empty, ENABLED) advanced params
function advancedOverrides() {
  const out = {};
  document.querySelectorAll("#adv-body input[data-param]").forEach((i) => {
    if (!i.disabled && i.value.trim() !== "") out[i.dataset.param] = parseFloat(i.value);
  });
  return out;
}

// ---- grey out parameters that don't belong to the chosen model/snow -------
const CRESTPHYS_ONLY = ["igw", "hmaxaq", "gwc", "gwe"];   // groundwater terms
const WB_KEYS = ["wm", "b", "im", "ke", "fc", "iwu", ...CRESTPHYS_ONLY];
const HP_KEYS = ["precip", "split"];
const SNOW_KEYS = ["uadj", "mbase", "mfmax", "mfmin", "tipm", "nmf", "plwhc", "scf", "pxtemp"];

function paramEnabled(k) {
  const m = document.getElementById("k-model").value;     // auto|crestphys|crest|hp
  const s = document.getElementById("k-snow").value;      // auto|on|off
  if (m === "auto") return false;   // Auto = the agent's calibrated parameter
                                    // set from the database — nothing editable
  if (SNOW_KEYS.includes(k)) return s !== "off";
  if (HP_KEYS.includes(k)) return m === "hp";
  if (CRESTPHYS_ONLY.includes(k)) return m === "crestphys";
  if (WB_KEYS.includes(k)) return m !== "hp";
  return true;                                            // KW routing
}

function updateParamAvailability() {
  const auto = document.getElementById("k-model").value === "auto";
  const note = document.getElementById("adv-auto-note");
  if (note) note.classList.toggle("hidden", !auto);
  document.querySelectorAll("#adv-body input[data-param]").forEach((i) => {
    const on = paramEnabled(i.dataset.param);
    i.disabled = !on;
    i.parentElement.classList.toggle("off", !on);
  });
  // grey a whole group header when none of its params apply
  document.querySelectorAll("#adv-body .adv-group").forEach((h) => {
    const grid = h.nextElementSibling;
    if (!grid) return;
    const inputs = [...grid.querySelectorAll("input[data-param]")];
    h.classList.toggle("off", inputs.length > 0 && inputs.every((i) => i.disabled));
  });
}
document.getElementById("k-model").addEventListener("change", updateParamAvailability);
document.getElementById("k-snow").addEventListener("change", updateParamAvailability);
updateParamAvailability();      // initial state (auto model, auto snow)

document.getElementById("adv-toggle").onclick = () => {
  const b = document.getElementById("adv-body");
  const open = b.classList.toggle("hidden");
  document.getElementById("adv-arrow").textContent = open ? "▸" : "▾";
};

document.getElementById("anim-play").onclick = togglePlay;
document.getElementById("anim-slider").oninput = (e) => { stopPlay(); setFrame(parseInt(e.target.value)); };

// ---- Model options Set / Clear (applies to the WHOLE panel) ---------------
function describeOpts(o) {
  const bits = [];
  if (o.start || o.end) bits.push(`window ${o.start || "(AI start)"} → ${o.end || "(AI end)"}`);
  if (o.hours !== DEFAULT_OPTS.hours) bits.push(`duration ${o.hours} h`);
  if (o.timestep !== DEFAULT_OPTS.timestep) bits.push(`step ${o.timestep}`);
  if (o.warmup_days !== DEFAULT_OPTS.warmup_days) bits.push(`warm-up ${o.warmup_days} d`);
  if (o.model !== DEFAULT_OPTS.model) bits.push(`model ${o.model.toUpperCase()}`);
  if (o.snow !== DEFAULT_OPTS.snow) bits.push(`snow ${o.snow}`);
  if (o.overrides) bits.push(`${Object.keys(o.overrides).length} advanced param(s)`);
  return bits.length ? bits.join(" · ") : "all defaults";
}

document.getElementById("k-time-set").onclick = () => {
  const o = readPanel();
  if (o.start && o.end && o.end <= o.start) {
    addMsg("⚠ The end date must be after the start date.", "status");
    return;
  }
  manualOpts = o;
  document.getElementById("k-time-state").textContent = "🔒 options set";
  addMsg(`🔒 Model options <b>set</b> — ${describeOpts(o)}. These override the AI/chat ` +
    `values for every simulation until you hit <b>Clear</b>. Fields left at their ` +
    `default (or blank dates) keep the AI/default value.`, "bot");
  allowResim();
};
document.getElementById("k-time-clear").onclick = () => {
  manualOpts = null;
  document.getElementById("k-start").value = "";
  document.getElementById("k-end").value = "";
  document.getElementById("k-hours").value = "48";
  document.getElementById("k-step-n").value = "1";
  document.getElementById("k-step-u").value = "h";
  document.getElementById("k-warmup").value = "90";
  document.getElementById("k-model").value = "auto";
  document.getElementById("k-snow").value = "auto";
  document.querySelectorAll("#adv-body input[data-param]").forEach((i) => { i.value = ""; });
  document.getElementById("k-time-state").textContent = "";
  updateParamAvailability();
  addMsg("🔓 Model options cleared — back to defaults, with the AI/chat-defined window.", "bot");
  allowResim();
};

// changing any model option means a new run would differ — re-enable Simulate
document.querySelectorAll("#left-panel input, #left-panel select").forEach((el) => {
  el.addEventListener("change", allowResim);
});

// ---- test-user feedback ----------------------------------------------------
document.getElementById("btn-feedback").onclick = () => {
  document.getElementById("feedback-modal").classList.remove("hidden");
  document.getElementById("fb-text").focus();
};
document.getElementById("fb-close").onclick = () =>
  document.getElementById("feedback-modal").classList.add("hidden");
document.getElementById("feedback-modal").addEventListener("click", (e) => {
  if (e.target.id === "feedback-modal") e.target.classList.add("hidden");
});
document.getElementById("fb-send").onclick = async () => {
  const text = document.getElementById("fb-text").value.trim();
  const status = document.getElementById("fb-status");
  if (!text) { status.textContent = "⚠ write a comment first"; return; }
  status.textContent = "sending…";
  try {
    const r = await fetch("/api/feedback", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, contact: document.getElementById("fb-contact").value.trim() || null }),
    });
    if (!r.ok) throw new Error("failed");
    status.textContent = "✓ thank you! recorded for the daily review";
    document.getElementById("fb-text").value = "";
    setTimeout(() => {
      document.getElementById("feedback-modal").classList.add("hidden");
      status.textContent = "";
    }, 1800);
  } catch (_) {
    status.textContent = "⚠ could not send — please try again";
  }
};

// ---- reattach: a run keeps going in the backend even if the app is closed --
async function reattach(explicitId) {
  const simId = explicitId || localStorage.getItem("lastSimId");
  if (!simId) return;
  try {
    const r = await fetch(`/api/job/${simId}`);
    if (!r.ok) { if (!explicitId) localStorage.removeItem("lastSimId"); return; }
    const j = await r.json();
    if (!explicitId && j.done && j.age_s > 24 * 3600) { localStorage.removeItem("lastSimId"); return; }
    zoomedToOverlay = false;               // zoom to the 2-D layer on first frame
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

// ---- Nowcast mode: precomputed DI-LSTM +6 h predictions ------------------
// The updater Space refreshes nowcast/latest.parquet hourly for every CONUS
// gauge, so this mode never runs a simulation — selection -> instant plot.
// Gauge-point predictions only: no 2-D streamflow in nowcast mode.
let nowcastMode = false;
let nowcastRes = null;              // {t0, times, generated, model, gauges: {id: g}}
let nowcastPanelActive = false;     // right panel is currently showing a nowcast
let ncTimer = null;

function setMode(nc) {
  nowcastMode = nc;
  document.getElementById("mode-hind").classList.toggle("on", !nc);
  document.getElementById("mode-now").classList.toggle("on", nc);
  refreshSelection();
  if (nc) {
    addMsg("⚡ <b>Nowcast mode</b> — no dates needed. Click gauges, draw a rectangle, " +
           "or just zoom until ≤25 gauges are in view: each shows observed flow plus " +
           "the AI's next-6-hour prediction, precomputed hourly for every CONUS gauge. " +
           "<b>Experimental</b>; gauge points only (2-D maps stay in Hindcast).", "status");
    scheduleAutoView();
  } else {
    addMsg("🕘 <b>Hindcast mode</b> — historical CREST simulations (pick gauges and a time window).", "status");
    if (panelGauge && simHydro[panelGauge]) focusGauge(panelGauge);
  }
}

function scheduleNowcast() {
  clearTimeout(ncTimer);
  ncTimer = setTimeout(() => { if (selected.size) showNowcastsFor([...selected]); }, 400);
}

function scheduleAutoView() {        // zoomed to a small area -> show everything in view
  clearTimeout(ncTimer);
  ncTimer = setTimeout(() => {
    if (!nowcastMode || selected.size) return;
    const b = map.getBounds();
    const vis = Object.values(gaugeData).filter((g) => b.contains([g.lat, g.lon]));
    if (vis.length && vis.length <= 25) showNowcastsFor(vis.map((g) => g.id));
  }, 900);
}

async function showNowcastsFor(ids) {
  ids = ids.slice(0, 25);
  if (!ids.length) return;
  try {
    const r = await fetch(`/api/nowcast_now?w=0&s=0&e=0&n=0&ids=${ids.join(",")}&obs_hours=48`);
    const d = await r.json();
    if (!d.ok) {
      addMsg(`⚠️ Nowcast unavailable: ${escapeHtml(d.reason || "no precomputed data yet")}`, "status");
      return;
    }
    nowcastRes = { t0: d.t0, times: d.times, generated: d.generated,
                   model: d.model, gauges: {} };
    (d.gauges || []).forEach((g) => { nowcastRes.gauges[g.id] = g; });
    const first = ids.find((i) => nowcastRes.gauges[i]);
    if (!first) {
      addMsg("⚠️ No precomputed nowcast for these gauges (outside CONUS radar coverage?).", "status");
      return;
    }
    focusNowcastGauge(nowcastRes.gauges[panelGauge] ? panelGauge : first);
  } catch (e) {
    addMsg(`⚠️ Nowcast fetch failed: ${escapeHtml(e.message)}`, "status");
  }
}

function focusNowcastGauge(id) {
  panelGauge = id;
  nowcastPanelActive = true;
  const nc = nowcastRes.gauges[id];
  const g = gaugeData[id];
  document.getElementById("right-panel").classList.remove("hidden");
  document.getElementById("rp-reopen").classList.add("hidden");
  document.getElementById("rp-title").textContent = `⚡ ${id} · ${g ? g.name : ""}`;
  const bar = document.getElementById("rp-tabs");
  bar.innerHTML = "";
  Object.keys(nowcastRes.gauges).forEach((gid2) => {
    const t = document.createElement("button");
    t.className = "rp-tab done" + (gid2 === panelGauge ? " active" : "");
    t.innerHTML = `<span class="dot"></span>${gid2}<span class="x" title="Close & unselect this gauge">✕</span>`;
    t.onclick = () => focusNowcastGauge(gid2);
    t.querySelector(".x").onclick = (e) => { e.stopPropagation(); closeNowcastTab(gid2); };
    bar.appendChild(t);
  });
  renderFavBtn();
  const peak = Math.max(...nc.q);
  const cards = [statCard("Drainage",
    Math.round((g ? g.area_km2 : nc.area_km2)).toLocaleString() + " km²")];
  if (nc.obs_last_q != null) cards.push(statCard("Latest obs", nc.obs_last_q + " m³/s"));
  if (nc.obs_age_h != null) cards.push(statCard("Obs age", nc.obs_age_h + " h"));
  cards.push(statCard("Peak +6 h (AI)", (Math.round(peak * 10) / 10) + " m³/s"));
  document.getElementById("rp-stats").innerHTML = cards.join("");
  renderNowcastHydro(id);
  document.getElementById("rp-report").innerHTML =
    `<div class="adv-note">🔮 AI nowcast issued <b>${nowcastRes.t0 || "?"}</b> (newest radar hour), ` +
    `refreshed hourly. <b>Experimental</b> — machine-learning prediction at the gauge point, ` +
    `not a CREST simulation. Switch to 🕘 Hindcast for physics runs and 2-D maps.</div>`;
}

function _nowcastFig(id, big) {
  const nc = nowcastRes.gauges[id];
  const obs = nc.obs || [];
  const traces = [];
  if (obs.length) {
    traces.push({ x: obs.map((r) => r[0]), y: obs.map((r) => r[1]), name: "Obs Q",
      mode: "lines", line: { color: "#f4f4f4", width: big ? 1.8 : 1.5,
                             shape: "spline", smoothing: 0.8 } });
  }
  traces.push({ x: nowcastRes.times, y: nc.q, name: "🔮 AI next 6 h",
    mode: "lines+markers", line: { color: "#ff9f43", width: big ? 2.4 : 2, dash: "dot" },
    marker: { size: big ? 7 : 5 } });
  const issue = (nowcastRes.t0 || "").slice(0, 16);
  // default view = 6 h history + 6 h prediction; the full 48 h of obs are in
  // the traces, so dragging (or scroll-zoom in the enlarged view) reveals them
  let range = null;
  if (issue) {
    const t0ms = Date.parse(issue.replace(" ", "T") + ":00Z");
    const fmt = (ms) => new Date(ms).toISOString().slice(0, 16).replace("T", " ");
    range = [fmt(t0ms - 6 * 3600e3), fmt(t0ms + 6.5 * 3600e3)];
  }
  const layout = {
    margin: big ? { l: 56, r: 24, t: 18, b: 40 } : { l: 46, r: 12, t: 12, b: 30 },
    showlegend: true,
    legend: { orientation: "h", y: big ? 1.08 : 1.18, font: { size: big ? 11 : 9 } },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#cdd9e2", size: big ? 12 : 10 },
    xaxis: { gridcolor: "rgba(255,255,255,.06)", range: range || undefined },
    yaxis: { title: "Q m³/s", rangemode: "tozero", gridcolor: "rgba(255,255,255,.06)" },
    shapes: issue ? [{ type: "line", x0: issue, x1: issue, y0: 0, y1: 1, yref: "paper",
                       line: { color: "#ffd23f", width: 1.2, dash: "dot" } }] : [],
    hovermode: "x",
  };
  return { traces, layout };
}

function renderNowcastHydro(id) {
  const el = document.getElementById("rp-hydro");
  document.getElementById("rp-expand").classList.remove("hidden");
  document.getElementById("rp-readout").classList.add("hidden");
  const { traces, layout } = _nowcastFig(id, false);
  if (el.querySelector(".muted")) el.innerHTML = "";
  Plotly.react(el, traces, layout, { displayModeBar: false, responsive: true });
  if (hydroModalOpen) renderHydroBig();          // keep the big view in sync
}

function closeNowcastTab(id) {
  selected.delete(id);
  if (nowcastRes) delete nowcastRes.gauges[id];
  refreshSelection();
  const rest = nowcastRes ? Object.keys(nowcastRes.gauges) : [];
  if (panelGauge === id) {
    if (rest.length) { focusNowcastGauge(rest[0]); return; }
    panelGauge = null;
    nowcastPanelActive = false;
    document.getElementById("right-panel").classList.add("hidden");
    document.getElementById("rp-reopen").classList.add("hidden");
    return;
  }
  focusNowcastGauge(panelGauge);                 // re-render tabs without `id`
}

document.getElementById("mode-hind").onclick = () => { if (nowcastMode) setMode(false); };
document.getElementById("mode-now").onclick = () => { if (!nowcastMode) setMode(true); };

// ---- boot ----------------------------------------------------------------
initAuth();
loadHuc8();                         // HUC8 basin guide (zoom in for gauge pins)
loadViewportGauges();               // gauge pins when already zoomed in
reattach();                         // pick up a run started before the app was closed
