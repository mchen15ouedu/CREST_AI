/* Risk view — the three-armed "risk lens" for 2-D inundation events.
 *
 * Visual idiom adapted from mchen15ouedu/map_visual (Demographic Profiles,
 * a VisQuill-design reimplementation): a central balloon under a fixed
 * lens, three radial arms with perpendicular value bars, and the map
 * panning underneath to select what is probed. Here the probed unit is a
 * single SOLVER PIXEL (lit up on the map as its exact footprint), and the
 * arms show that pixel's values:
 *
 *   inundation      depth (scrubbed frame or episode max), peak speed,
 *                   compass flow direction at the peak
 *   population      WorldPop 2026-2030 age-sex people IN THIS CELL
 *                   (resampled to the inundation grid, mass-conserving)
 *   infrastructure  OSM values at the pixel: buildings, road class,
 *                   critical facilities within a 250 m ring
 *
 * The balloon's color is the risk level: DEFRA hazard rating
 * HR = depth x (speed + 0.5) — depth-only classes when the event predates
 * the flux rasters. Plain script, no modules: exposes window.RiskView.
 */
(function () {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var HAZ = {
    dry:        { color: "#6b7f94", label: "dry" },
    low:        { color: "#4cc9a0", label: "low" },
    moderate:   { color: "#ffd23f", label: "moderate" },
    significant:{ color: "#ff9f1c", label: "significant" },
    extreme:    { color: "#ef476f", label: "extreme" },
  };
  var POP_BUCKETS = ["0-4", "5-14", "15-24", "25-44", "45-64", "65-74", "75+"];
  var G = {                       // design-box geometry (viewBox units)
    w: 900, h: 760, r: 64,        // balloon radius
    armInner: 96,                 // first slot radius
    slotStep: 34,                 // radial distance between slots
    barOff: 10,                   // gap axis -> bar start
    barLen: 130,                  // full-value bar length
    capR: 78,                     // curved caption radius
  };
  var ARMS = {
    inund: { angle: -Math.PI / 3, caption: "Inundation" },
    pop:   { angle:  Math.PI / 3, caption: "Population" },
    infra: { angle:  Math.PI,     caption: "Infrastructure" },
  };

  var st = {
    open: false, map: null, eventId: null, frameT: "",
    root: null, svg: null, parts: null, hiRect: null,
    seq: 0, popRetry: null, anim: null, bars: {}, targets: {},
  };

  function el(name, attrs, parent) {
    var n = document.createElementNS(SVGNS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function dir(a) { return [Math.sin(a), -Math.cos(a)]; }
  function perp(v) { return [-v[1], v[0]]; }
  function add(a, b, s) { return [a[0] + b[0] * s, a[1] + b[1] * s]; }
  function mul(v, s) { return [v[0] * s, v[1] * s]; }
  function readable(v) {
    var deg = Math.atan2(v[1], v[0]) * 180 / Math.PI, flip = false;
    if (deg > 90) { deg -= 180; flip = true; }
    else if (deg <= -90) { deg += 180; flip = true; }
    return { deg: deg, flip: flip };
  }
  function txt(parent, cls, s, p, rot, anchor) {
    var t = el("text", { class: cls }, parent);
    t.textContent = s;
    t.setAttribute("transform",
      "translate(" + p[0] + "," + p[1] + ") rotate(" + (rot || 0) + ")");
    if (anchor) t.style.textAnchor = anchor;
    return t;
  }

  // -- construction ---------------------------------------------------------
  function build(mapDiv) {
    // bar registries FIRST: mkBar() fills them while the arms are built
    // (they were reset after construction once — every setBar then threw
    // into the probe's swallow-all catch and the arms stayed blank)
    st.bars = {}; st.targets = {};
    var root = document.createElement("div");
    root.id = "riskview";
    root.innerHTML = "";
    mapDiv.appendChild(root);
    var svg = el("svg", { viewBox: (-G.w / 2) + " " + (-G.h / 2) + " " + G.w + " " + G.h });
    root.appendChild(svg);
    var g = el("g", {}, svg);

    var parts = { arms: {} };
    // central balloon
    parts.balloon = el("circle", { class: "rv-balloon", cx: 0, cy: 0, r: G.r }, g);
    parts.hazClass = txt(g, "rv-haz-class", "—", [0, -8]);
    parts.hazVal = txt(g, "rv-haz-val", "", [0, 14]);
    parts.hazBasis = txt(g, "rv-haz-basis", "", [0, 30]);

    for (var key in ARMS) buildArm(g, parts, key, ARMS[key]);
    st.root = root; st.svg = svg; st.parts = parts;
    fit();
  }

  function slotR(i) { return G.armInner + G.slotStep * i; }

  function buildArm(g, parts, key, spec) {
    var ag = el("g", { class: "rv-arm rv-arm-" + key }, g);
    var d = dir(spec.angle), p = perp(d);
    var nSlots = key === "pop" ? POP_BUCKETS.length : 3;
    el("line", { class: "rv-axis", x1: d[0] * (G.armInner - 14), y1: d[1] * (G.armInner - 14),
                 x2: d[0] * (slotR(nSlots - 1) + 16), y2: d[1] * (slotR(nSlots - 1) + 16) }, ag);
    // curved caption
    var a0 = spec.angle - Math.PI / 2, a1 = spec.angle + Math.PI / 2;
    var pt = function (a, r) { return [r * Math.sin(a), -r * Math.cos(a)]; };
    var up = Math.cos(spec.angle) > 0;
    var s0 = pt(up ? a0 : a1, G.capR), s1 = pt(up ? a1 : a0, G.capR);
    var id = "rv-cap-" + key;
    el("path", { id: id, fill: "none",
      d: "M " + s0[0] + " " + s0[1] + " A " + G.capR + " " + G.capR + " 0 0 " + (up ? 1 : 0) + " " + s1[0] + " " + s1[1] }, ag);
    var ct = el("text", { class: "rv-caption" }, ag);
    var tp = el("textPath", { href: "#" + id, startOffset: "50%" }, ct);
    tp.setAttribute("text-anchor", "middle");
    tp.textContent = spec.caption;

    var arm = { d: d, p: p, spec: spec, slots: [], note: null };
    var rot = readable(d);
    if (key === "pop") {
      for (var i = 0; i < POP_BUCKETS.length; i++) {
        var base = mul(d, slotR(i));
        var lbl = txt(ag, "rv-slot-label", POP_BUCKETS[i], base, rot.deg);
        arm.slots.push({
          base: base,
          left: mkBar(ag, key + "_m" + i, "rv-bar-male"),
          right: mkBar(ag, key + "_f" + i, "rv-bar-female"),
          vLeft: txt(ag, "rv-val rv-val-male", "", base, rot.deg),
          vRight: txt(ag, "rv-val rv-val-female", "", base, rot.deg),
          label: lbl,
        });
      }
      var endP = mul(d, slotR(POP_BUCKETS.length - 1) + 40);
      arm.note = txt(ag, "rv-note", "", endP, rot.deg);
      var mf = add(mul(d, G.armInner - 34), p, -G.barOff - 26);
      var ff = add(mul(d, G.armInner - 34), p, G.barOff + 26);
      txt(ag, "rv-mf", "male", mf, rot.deg, "middle");
      txt(ag, "rv-mf", "female", ff, rot.deg, "middle");
    } else {
      var names = key === "inund" ? ["depth", "speed", "flow dir"]
                                  : ["buildings", "road", "critical"];
      for (var j = 0; j < 3; j++) {
        var b2 = mul(d, slotR(j));
        arm.slots.push({
          base: b2,
          bar: j < 2 || key !== "inund" ? mkBar(ag, key + "_" + j, "rv-bar") : null,
          label: txt(ag, "rv-slot-label", names[j], b2, rot.deg),
          val: txt(ag, "rv-val", "", b2, rot.deg),
        });
      }
      if (key === "inund") {                 // compass glyph on slot 3
        var b3 = mul(d, slotR(2));
        arm.compass = el("g", { class: "rv-compass" }, ag);
        el("circle", { cx: 0, cy: 0, r: 13, class: "rv-compass-ring" }, arm.compass);
        el("path", { class: "rv-compass-arrow",
                     d: "M 0 -10 L 5 6 L 0 2 L -5 6 Z" }, arm.compass);
        arm.compassPos = add(b3, arm.p, G.barOff + 22);
        arm.compass.setAttribute("transform",
          "translate(" + arm.compassPos[0] + "," + arm.compassPos[1] + ")");
        arm.compass.style.display = "none";
      }
      arm.note = txt(ag, "rv-note", "", mul(d, slotR(2) + 44), rot.deg);
    }
    parts.arms[key] = arm;
  }

  function mkBar(parent, id, cls) {
    var ln = el("line", { class: cls, x1: 0, y1: 0, x2: 0, y2: 0 }, parent);
    st.bars[id] = { ln: ln, cur: 0 };
    st.targets[id] = { len: 0, base: null, p: null, side: 1 };
    return ln;
  }

  // -- animation (exponential ease toward targets, like the original) ------
  function animTick() {
    var active = false;
    for (var id in st.targets) {
      var b = st.bars[id], t = st.targets[id];
      if (!t.base) continue;
      var next = Math.abs(t.len - b.cur) < 0.3 ? t.len : b.cur + (t.len - b.cur) * 0.18;
      if (next !== b.cur) { b.cur = next; active = true; }
      var q0 = add(t.base, t.p, t.side * G.barOff);
      var q1 = add(t.base, t.p, t.side * (G.barOff + b.cur));
      b.ln.setAttribute("x1", q0[0]); b.ln.setAttribute("y1", q0[1]);
      b.ln.setAttribute("x2", q1[0]); b.ln.setAttribute("y2", q1[1]);
      b.ln.style.display = b.cur > 0.4 ? "" : "none";
    }
    st.anim = active ? requestAnimationFrame(animTick) : null;
  }
  function setBar(id, base, p, side, frac) {
    var t = st.targets[id];
    t.base = base; t.p = p; t.side = side;
    t.len = Math.max(0, Math.min(1, frac)) * G.barLen;
    if (!st.anim) st.anim = requestAnimationFrame(animTick);
  }

  // -- data -> arms ---------------------------------------------------------
  function fmtN(v, d) { return v === null || v === undefined ? "–" : (+v).toFixed(d); }

  function render(d) {
    var P = st.parts;
    // balloon
    var hz = (d.hazard && HAZ[d.hazard.class]) || HAZ.dry;
    P.balloon.style.fill = hz.color + "2e";
    P.balloon.style.stroke = hz.color;
    P.hazClass.textContent = hz.label.toUpperCase();
    P.hazClass.style.fill = hz.color;
    P.hazVal.textContent = d.hazard && d.hazard.class !== "dry"
      ? "HR " + fmtN(d.hazard.rating, 2) : "";
    P.hazBasis.textContent = d.hazard
      ? (d.hazard.basis === "depth-only" ? "depth-only (pre-flux event)" : "depth × velocity")
      : "";

    // inundation arm
    var A = P.arms.inund, cap = st.depthCap || 3;
    var inu = d.inundation || {};
    setBar("inund_0", A.slots[0].base, A.p, 1, (inu.depth_m || 0) / cap);
    A.slots[0].val.textContent = fmtN(inu.depth_m, 2) + " m" +
      (inu.depth_at === "max depth" ? " (max)" : "");
    place(A.slots[0].val, A, 0);
    setBar("inund_1", A.slots[1].base, A.p, 1, (inu.speed_ms || 0) / 3.0);
    A.slots[1].val.textContent = inu.speed_ms === null || inu.speed_ms === undefined
      ? "n/a" : fmtN(inu.speed_ms, 2) + " m/s peak";
    place(A.slots[1].val, A, 1);
    if (inu.direction) {
      A.compass.style.display = "";
      var deg = ["N","NE","E","SE","S","SW","W","NW"].indexOf(inu.direction) * 45;
      A.compass.setAttribute("transform", "translate(" + A.compassPos[0] + "," +
        A.compassPos[1] + ") rotate(" + deg + ")");
      A.slots[2].val.textContent = inu.direction;
    } else {
      A.compass.style.display = "none";
      A.slots[2].val.textContent = "n/a";
    }
    place(A.slots[2].val, A, 2, 44);
    A.note.textContent = "";

    // population arm (two-sided pyramid, auto-scaled per probe)
    var B = P.arms.pop, pop = d.population || {};
    if (pop.male) {
      var mx = 0.001;
      for (var i = 0; i < pop.male.length; i++)
        mx = Math.max(mx, pop.male[i], pop.female[i]);
      for (var j = 0; j < B.slots.length; j++) {
        var m = pop.male[j] || 0, f = pop.female[j] || 0;
        setBar("pop_m" + j, B.slots[j].base, B.p, -1, m / mx);
        setBar("pop_f" + j, B.slots[j].base, B.p, 1, f / mx);
        B.slots[j].vLeft.textContent = m > 0 ? m.toFixed(2) : "";
        B.slots[j].vRight.textContent = f > 0 ? f.toFixed(2) : "";
        placePop(B, j);
      }
      B.note.textContent = "≈" + fmtN(pop.cell_total, 2) + " people in this cell · WorldPop " +
        pop.year + (pop.source_res === "1km" ? " (1 km source)" : "");
    } else {
      for (var k = 0; k < B.slots.length; k++) {
        setBar("pop_m" + k, B.slots[k].base, B.p, -1, 0);
        setBar("pop_f" + k, B.slots[k].base, B.p, 1, 0);
        B.slots[k].vLeft.textContent = ""; B.slots[k].vRight.textContent = "";
      }
      B.note.textContent = pop.status === "building"
        ? "loading population…" : (pop.status || "population unavailable");
    }

    // infrastructure arm
    var C = P.arms.infra, inf = d.infrastructure || {};
    if (inf.status === "ok") {
      setBar("infra_0", C.slots[0].base, C.p, 1, Math.min(1, (inf.buildings || 0) / 20));
      C.slots[0].val.textContent = (inf.buildings || 0) +
        (Object.keys(inf.building_types || {}).length
          ? " (" + Object.keys(inf.building_types).slice(0, 2).join(", ") + ")" : "");
      place(C.slots[0].val, C, 0);
      setBar("infra_1", C.slots[1].base, C.p, 1, (inf.road_rank || 0) / (inf.road_rank_max || 8));
      C.slots[1].val.textContent = inf.roads && inf.roads.length
        ? inf.roads[0].class + (inf.roads[0].name ? " · " + inf.roads[0].name : "")
        : "none";
      place(C.slots[1].val, C, 1);
      var nc = (inf.critical || []).length;
      setBar("infra_2", C.slots[2].base, C.p, 1, Math.min(1, nc / 4));
      C.slots[2].val.textContent = nc
        ? nc + " · " + inf.critical[0].kind +
          (inf.critical[0].dist_m !== null ? " " + inf.critical[0].dist_m + " m" : "")
        : "none in " + (inf.context_m || 250) + " m";
      place(C.slots[2].val, C, 2);
      C.note.textContent = "OSM · pixel + " + (inf.context_m || 250) + " m ring";
    } else {
      for (var q = 0; q < 3; q++) {
        setBar("infra_" + q, C.slots[q].base, C.p, 1, 0);
        C.slots[q].val.textContent = "";
      }
      C.note.textContent = "OSM " + (inf.status || "unavailable");
    }
  }

  function place(valEl, arm, i, extra) {
    var rot = readable(arm.d);
    var pos = add(add(mul(arm.d, slotR(i)), arm.p, G.barOff + G.barLen + (extra || 8)), arm.d, 0);
    valEl.setAttribute("transform",
      "translate(" + pos[0] + "," + pos[1] + ") rotate(" + rot.deg + ")");
    valEl.style.textAnchor = rot.flip ? "end" : "start";
  }
  function placePop(arm, i) {
    var rot = readable(arm.d);
    var L0 = add(mul(arm.d, slotR(i)), arm.p, -(G.barOff + G.barLen + 8));
    var R0 = add(mul(arm.d, slotR(i)), arm.p, G.barOff + G.barLen + 8);
    arm.slots[i].vLeft.setAttribute("transform",
      "translate(" + L0[0] + "," + L0[1] + ") rotate(" + rot.deg + ")");
    arm.slots[i].vLeft.style.textAnchor = rot.flip ? "start" : "end";
    arm.slots[i].vRight.setAttribute("transform",
      "translate(" + R0[0] + "," + R0[1] + ") rotate(" + rot.deg + ")");
    arm.slots[i].vRight.style.textAnchor = rot.flip ? "end" : "start";
  }

  // -- probing --------------------------------------------------------------
  var debounceT = null;
  function probeSoon() {
    if (debounceT) clearTimeout(debounceT);
    debounceT = setTimeout(probeNow, 260);
  }
  function probeNow() {
    if (!st.open) return;
    var ctr = st.map.getCenter();
    var seq = ++st.seq;
    var url = "/api/riskprobe/" + encodeURIComponent(st.eventId) +
      "?lat=" + ctr.lat.toFixed(6) + "&lon=" + ctr.lng.toFixed(6) +
      (st.frameT ? "&t=" + encodeURIComponent(st.frameT) : "");
    fetch(url).then(function (r) { return r.json(); }).then(function (d) {
      if (!st.open || seq !== st.seq) return;
      if (d.error) {
        st.parts.hazClass.textContent = "—";
        st.parts.hazClass.style.fill = "#8aa2ba";
        st.parts.hazVal.textContent = "";
        st.parts.hazBasis.textContent = d.error.indexOf("outside") >= 0
          ? "outside the simulated basin" : d.error.slice(0, 40);
        if (st.hiRect) { st.hiRect.remove(); st.hiRect = null; }
        return;
      }
      // light up the probed pixel: its exact footprint
      if (d.cell && d.cell.bounds) {
        if (st.hiRect) st.hiRect.setBounds(d.cell.bounds);
        else st.hiRect = L.rectangle(d.cell.bounds, {
          color: "#ffd23f", weight: 2, fill: true,
          fillColor: "#ffd23f", fillOpacity: 0.25, interactive: false,
        }).addTo(st.map);
      }
      render(d);
      // population window still building server-side: poll it in
      if (d.population && d.population.status === "building") {
        if (st.popRetry) clearTimeout(st.popRetry);
        st.popRetry = setTimeout(probeNow, 4000);
      }
    }).catch(function () {});
  }

  function fit() {
    if (!st.root || !st.map) return;
    var sz = st.map.getSize();
    var s = Math.min(1, Math.min(sz.x / G.w, sz.y / G.h) * 0.98);
    st.svg.style.width = (G.w * s) + "px";
    st.svg.style.height = (G.h * s) + "px";
  }

  // -- public API -----------------------------------------------------------
  window.RiskView = {
    isOpen: function () { return st.open; },
    open: function (map, eventId, opts) {
      if (st.open) this.close();
      st.map = map; st.eventId = eventId;
      st.frameT = (opts && opts.frameT) || "";
      st.depthCap = (opts && opts.depthCap) || 3;
      build(map.getContainer());
      st.open = true;
      map.on("moveend zoomend", probeSoon);
      map.on("resize", fit);
      probeNow();
    },
    close: function () {
      if (!st.open) return;
      st.open = false;
      if (st.map) { st.map.off("moveend zoomend", probeSoon); st.map.off("resize", fit); }
      if (st.hiRect) { st.hiRect.remove(); st.hiRect = null; }
      if (st.popRetry) clearTimeout(st.popRetry);
      if (st.anim) cancelAnimationFrame(st.anim);
      st.anim = null;
      if (st.root) st.root.remove();
      st.root = null;
    },
    setEvent: function (eventId, opts) {
      if (!st.open) return;
      st.eventId = eventId;
      st.frameT = (opts && opts.frameT) || "";
      st.depthCap = (opts && opts.depthCap) || st.depthCap;
      probeSoon();
    },
    setFrame: function (t) {
      if (!st.open) return;
      st.frameT = t || "";
      probeSoon();
    },
  };
})();
