"use strict";

// LightHit event viewer: interactive scene, analysis panels and deterministic
// movie export.  The glow of an OM shows the expected (mean) photoelectron
// response on the detector clock; it is never a sampled hit.  Stored arrays
// are read only.
(() => {
  const $ = id => document.getElementById(id);
  const Model = LightHitStudioModel, GL = LightHitStudioGL, Movie = LightHitMovie;
  const C = Model.C_M_PER_NS, TAU = 2 * Math.PI;
  const STORE = "lighthit-viewer-settings-v1";
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = t => t <= 0 ? 0 : t >= 1 ? 1 : 0.5 - 0.5 * Math.cos(Math.PI * t);
  const fmt = (x, n = 4) => Number.isFinite(x) ? Number(x).toLocaleString("en-US", {maximumSignificantDigits: n}) : "—";
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b, s = 1) => [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]];
  const norm = a => Math.hypot(a[0], a[1], a[2]);

  // ------------------------------------------------------------- colours
  // Turbo colour map (Mikhailov 2019, polynomial fit), sRGB in [0, 1].
  function turbo(x) {
    x = clamp(x, 0, 1);
    return [
      0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943)))),
      0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604)))),
      0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973)))),
    ].map(v => clamp(v, 0, 1));
  }
  // Early light red, late light blue, as in neutrino-telescope event displays.
  const timeColour = f => turbo(0.94 - 0.84 * clamp(f, 0, 1));
  const iceColour = f => {
    f = clamp(f, 0, 1);
    return [lerp(0.10, 1.0, f ** 1.6), lerp(0.30, 1.0, f ** 0.9), lerp(0.75, 1.0, f)];
  };
  const COMPONENT = [[1.0, 0.70, 0.30], [0.30, 0.92, 0.68], [0.68, 0.50, 1.0]];
  const COMPONENT_NAMES = ["Unscattered", "One scattering", "Two or more"];
  const linear = c => c.map(v => Math.pow(clamp(v, 0, 1), 2.2));
  const css = c => `rgb(${c.map(v => Math.round(255 * clamp(v, 0, 1))).join(",")})`;

  // ------------------------------------------------------------- settings
  const CONTROLS = {
    part: "rebuild", display: "rebuild", colour: "draw", cluster: "visible",
    threshold: "levels", size: "draw", flash: "draw", "show-front": "draw", "show-source": "static",
    "show-strings": "static", "show-dark": "static", "show-floor": "static", "show-snow": "draw",
    bloom: "draw", exposure: "draw", lambda: "draw", fog: "draw",
    "movie-size": "none", "movie-fps": "none", intro: "none", "event-seconds": "none", outro: "none",
    pacing: "pace", "camera-motion": "none", orbit: "none", zoom: "none",
    "title-text": "draw", "subtitle-text": "draw", "hud-title": "draw", "hud-clock": "draw",
    "hud-legend": "draw", "hud-scale": "draw", "hud-inset": "draw", "hud-note": "draw", "play-seconds": "none",
  };
  const S = {};
  function readControl(id) {
    const node = $(id);
    if (node.type === "checkbox") return node.checked;
    if (node.type === "range") return Number(node.value);
    return node.value;
  }
  function writeControl(id, value) {
    const node = $(id);
    if (node.type === "checkbox") node.checked = !!value; else node.value = value;
  }
  function loadSettings() {
    let saved = {};
    try {saved = JSON.parse(localStorage.getItem(STORE) || "{}") || {}} catch (_) {saved = {}}
    for (const id of Object.keys(CONTROLS)) {
      if (id in saved && id !== "cluster" && !id.endsWith("-text")) {
        try {writeControl(id, saved[id])} catch (_) {/* keep the default */}
      }
      S[id] = readControl(id);
    }
  }
  function saveSettings() {
    try {localStorage.setItem(STORE, JSON.stringify(S))} catch (_) {/* storage is optional */}
  }

  // ------------------------------------------------------------- state
  let data = null, eventIndex = 0, model = null, omState = null, renderer = null;
  let T = 0, selected = 0, visible = [], visibleSet = new Set();
  let threshold = 0.01, colourRange = [0, 1], peakVisible = 1, tRef = 0;
  let staticScene = null, snow = null, lastMats = null, dirty = true;
  let playing = false, playU = 0, pace = null, preview = null, exporting = null, tween = null;
  let keyframes = [], lastMovieUrl = null, hoverIndex = -1, controls = null;
  const smoothCache = new WeakMap();
  const camera = {target: [0, 0, 0], azimuth: -1.2, elevation: 0.25, distance: 800, fov: 34 * Math.PI / 180, follow: false};
  const dpr = () => Math.min(2, window.devicePixelRatio || 1);

  function eye(cam) {
    const ce = Math.cos(cam.elevation);
    return [cam.target[0] + cam.distance * ce * Math.cos(cam.azimuth),
            cam.target[1] + cam.distance * ce * Math.sin(cam.azimuth),
            cam.target[2] + cam.distance * Math.sin(cam.elevation)];
  }
  function glCamera(cam) {
    return {eye: eye(cam), target: cam.target, up: [0, 0, 1], fov: cam.fov,
            near: Math.max(0.5, cam.distance * 0.002), far: cam.distance * 12 + 6000};
  }
  const copyCamera = cam => ({target: cam.target.slice(), azimuth: cam.azimuth, elevation: cam.elevation,
                              distance: cam.distance, fov: cam.fov, follow: !!cam.follow});

  // ------------------------------------------------------------- data
  const det = () => data.detector;
  function moduleName(i) {
    const d = det();
    return `OM ${i} · C${d.cluster_id[i] + 1}/S${d.string_id[i] + 1}/${d.module_id[i] + 1}`;
  }
  function smoothRows(components, edges) {
    const event = data.events[eventIndex];
    let rows = smoothCache.get(event);
    if (!rows) {rows = LightHitViewerSmoothing.smoothComponents(components, edges, 3); smoothCache.set(event, rows)}
    return rows;
  }
  function validate(x) {
    if (!x || x.schema !== "lighthit/event-viewer/1" || !x.detector || !Array.isArray(x.events) || !x.events.length) {
      throw Error("Unsupported LightHit viewer format");
    }
    return x;
  }
  function primarySource() {return model.sources.find(s => s.kind !== "point") || model.sources[0] || null}
  function sourceAxis() {
    const s = primarySource();
    return s && s.direction ? s.direction : null;
  }

  function rebuildModel() {
    model = Model.build(data, eventIndex, {smooth: S.display === "smooth" ? smoothRows : null, part: S.part});
    omState = null;
    const s = primarySource();
    tRef = s?.timing ? s.timing.t0 : s?.emission ? s.emission.t : model.arrivalRange[0];
    pace = model.pacing(S.pacing);
    updateLevels();
    buildStatic();
    fillModules();
    updateInfo();
  }

  function refreshVisible() {
    const d = det(), cluster = S.cluster;
    visible = d.positions_m.map((_, i) => i).filter(i => cluster === "all" || d.cluster_id[i] === Number(cluster));
    visibleSet = new Set(visible);
  }

  // Threshold, colour range and size reference over the visible OMs.
  function updateLevels() {
    threshold = Math.pow(10, S.threshold);
    $("threshold-value").textContent = `${fmt(threshold, 2)} ${data.units.signal === "photoelectrons" ? "p.e." : data.units.signal}`;
    const times = [];
    peakVisible = 1e-300;
    for (const i of model.relevant) {
      if (!visibleSet.has(i)) continue;
      peakVisible = Math.max(peakVisible, model.window[i]);
      if (model.window[i] >= threshold && Number.isFinite(model.first[i])) times.push(model.first[i]);
    }
    times.sort((a, b) => a - b);
    colourRange = times.length ? [times[Math.floor(0.02 * (times.length - 1))], times[Math.ceil(0.98 * (times.length - 1))]] : [model.t0, model.t1];
    if (colourRange[1] - colourRange[0] < 1) colourRange = [colourRange[0] - 0.5, colourRange[0] + 0.5];
    if (!model.relevant.some(i => i === selected) || !visibleSet.has(selected)) {
      let best = -1, value = -Infinity;
      for (const i of model.relevant) if (visibleSet.has(i) && model.window[i] > value) {value = model.window[i]; best = i}
      if (best >= 0) selected = best;
    }
  }

  // Static geometry: strings, unlit OMs, floor grid, source guide.
  function buildStatic() {
    const d = det(), lines = [], beads = [];
    const pos = visible.map(i => d.positions_m[i]);
    const lo = [0, 1, 2].map(k => Math.min(...pos.map(p => p[k]))), hi = [0, 1, 2].map(k => Math.max(...pos.map(p => p[k])));
    const pushLine = (a, b, c0, c1, w) => lines.push(a[0], a[1], a[2], b[0], b[1], b[2], ...c0, ...c1, w);
    const px = dpr();
    if (S["show-strings"]) {
      const strings = new Map();
      for (const i of visible) {
        const key = `${d.cluster_id[i]}-${d.string_id[i]}`;
        if (!strings.has(key)) strings.set(key, []);
        strings.get(key).push(d.positions_m[i]);
      }
      for (const v of strings.values()) {
        v.sort((a, b) => a[2] - b[2]);
        const top = v[v.length - 1], bottom = v[0];
        pushLine(bottom, [top[0], top[1], top[2] + 25], [0.06, 0.10, 0.15, 0.6], [0.05, 0.08, 0.12, 0.35], 1.0 * px);
        pushLine([bottom[0], bottom[1], bottom[2] - 12], bottom, [0.05, 0.08, 0.11, 0.4], [0.06, 0.10, 0.15, 0.6], 1.0 * px);
      }
    }
    if (S["show-floor"]) {
      const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], 100);
      const step = span > 1200 ? 200 : span > 500 ? 100 : 50;
      const pad = 0.15 * span, z = lo[2] - 40;
      const x0 = Math.floor((lo[0] - pad) / step) * step, x1 = Math.ceil((hi[0] + pad) / step) * step;
      const y0 = Math.floor((lo[1] - pad) / step) * step, y1 = Math.ceil((hi[1] + pad) / step) * step;
      const c = [0.05, 0.09, 0.13, 0.55];
      for (let x = x0; x <= x1 + 1e-6; x += step) pushLine([x, y0, z], [x, y1, z], c, c, px);
      for (let y = y0; y <= y1 + 1e-6; y += step) pushLine([x0, y, z], [x1, y, z], c, c, px);
    }
    if (S["show-source"]) {
      for (const s of model.sources) {
        if (s.kind === "point") continue;
        const c = s.kind === "track" ? [0.30, 0.24, 0.12, 0.55] : [0.30, 0.24, 0.12, 0.35];
        pushLine(s.start, s.end, c, c, 1.3 * px);
      }
    }
    if (S["show-dark"]) {
      for (const i of visible) {
        const p = d.positions_m[i];
        beads.push(p[0], p[1], p[2], 1.15, 0.045, 0.065, 0.085, 1);
      }
    }
    staticScene = {lines: new Float32Array(lines), beads: new Float32Array(beads), lo, hi};
    // Seeded suspended particles for depth cues while the camera moves.
    let seed = 12345;
    const rand = () => (seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296;
    const out = [], count = 2600;
    const centre = lo.map((v, k) => (v + hi[k]) / 2), half = lo.map((v, k) => Math.max(60, 0.8 * (hi[k] - v)));
    for (let k = 0; k < count; k++) {
      out.push(centre[0] + (2 * rand() - 1) * half[0], centre[1] + (2 * rand() - 1) * half[1],
               centre[2] + (2 * rand() - 1) * half[2], 0.35, 0.55, 0.70, 0.90, 0.05 + 0.08 * rand());
    }
    snow = new Float32Array(out);
    dirty = true;
  }

  // ------------------------------------------------------------- scene per frame
  function sourceState(s, time) {
    if (s.kind === "track") {
      if (!s.timing) return {kind: "static"};
      const len = norm(sub(s.end, s.start)), first = s.points[0].xyz, last = s.points[s.points.length - 1].xyz;
      const path = norm(sub(last, first)) || len;
      if (time < s.timing.t0) return {kind: "before"};
      const h = Model.headAt(s, time);
      let travelled = norm(sub(h.xyz, first));
      if (h.done) travelled = path + C * (time - s.timing.t1);
      return {kind: "moving", head: h.done ? last : h.xyz, done: h.done, travelled, path, first};
    }
    if (!s.emission) return {kind: "static"};
    if (time < s.emission.t) return {kind: "before"};
    return {kind: "flash", age: time - s.emission.t, at: s.emission.position};
  }

  function compose(time) {
    omState = Model.stateAt(model, time, omState);
    const glows = [], additive = [], shells = [];
    const size = S.size, flash = S.flash, colourMode = S.colour, px = dpr();
    for (const i of model.relevant) {
      if (!visibleSet.has(i)) continue;
      const q = omState.q[i];
      if (!(q >= threshold)) continue;
      const p = det().positions_m[i];
      const r = size * (1.4 + 5.0 * Math.pow(q / peakVisible, 0.45));
      let colour;
      if (colourMode === "components") {
        const a = Math.max(0, omState.q3[i * 3]), b = Math.max(0, omState.q3[i * 3 + 1]), c = Math.max(0, omState.q3[i * 3 + 2]);
        const w = model.weights, total = w[0] * a + w[1] * b + w[2] * c || 1;
        colour = [0, 1, 2].map(k => (w[0] * a * COMPONENT[0][k] + w[1] * b * COMPONENT[1][k] + w[2] * c * COMPONENT[2][k]) / total);
      } else if (colourMode === "charge") {
        colour = iceColour(Math.log10(q / threshold) / Math.max(0.5, Math.log10(peakVisible / threshold)));
      } else {
        colour = timeColour((model.first[i] - colourRange[0]) / (colourRange[1] - colourRange[0]));
      }
      const burst = flash * 2.4 * Math.sqrt(Math.max(0, omState.rate[i]) * 8 / peakVisible);
      const intensity = 0.95 + Math.min(2.5, burst) + (i === selected ? 0.4 : 0);
      const lc = linear(colour);
      glows.push(p[0], p[1], p[2], r, lc[0], lc[1], lc[2], intensity);
    }
    let headPoint = null;
    const lambda = S.lambda;
    if (S["show-source"] || S["show-front"]) {
      for (const s of model.sources) {
        const st = sourceState(s, time);
        if (s.kind === "track" && st.kind === "moving") {
          headPoint = st.head;
          if (S["show-source"]) {
            const n = 48, from = st.first, to = st.head;
            for (let k = 0; k < n; k++) {
              const a = k / n, b = (k + 1) / n;
              const pa = add(from, sub(to, from), a), pb = add(from, sub(to, from), b);
              const fade = st.done ? 0.35 : 1;
              additive.push(...pa, ...pb, 1.0, 0.78, 0.42, fade * (0.06 + 0.9 * a ** 6), 1.0, 0.78, 0.42,
                            fade * (0.06 + 0.9 * b ** 6), 2.6 * px);
            }
            if (!st.done) {
              glows.push(...st.head, 2.4, 0.85, 0.93, 1.0, 7);
              glows.push(...st.head, 9, 0.35, 0.55, 1.0, 0.35);
            }
          }
          if (S["show-front"]) {
            const ng = model.groupIndex, cos = 1 / model.phaseIndex, k1 = 1 - cos / ng;
            const uhMax = Math.min(st.travelled, 6 * lambda * ng), uhMin = Math.max(0, st.travelled - st.path);
            if (uhMax > uhMin) {
              const apex = add(st.first, s.direction, st.travelled);
              shells.push({kind: "cone", origin: apex, axis: s.direction.map(x => -x), tanBeta: Math.tan(model.frontAngle),
                           a0: uhMin * k1, a1: uhMax * k1, k1, ng, color: [0.28, 0.58, 1.0], intensity: 0.42, lambda});
            }
          }
        } else if (st.kind === "flash") {
          headPoint = st.at;
          if (S["show-source"]) {
            const glow = 0.5 + 6 * Math.exp(-st.age / 25);
            if (s.points) {
              for (const p of s.points) if (p.w > 0.02) glows.push(...p.xyz, 1.0 + 3.5 * Math.sqrt(p.w), 1.0, 0.82, 0.55, glow * Math.sqrt(p.w));
            } else {
              glows.push(...st.at, 3, 1.0, 0.86, 0.6, glow);
            }
          }
          if (S["show-front"]) {
            const radius = C / model.groupIndex * st.age;
            if (radius > 0.5 && radius < 8 * lambda) {
              const iso = s.kind === "point";
              shells.push({kind: "sphere", origin: st.at, axis: s.direction || [0, 0, 1], radius,
                           peak: Math.acos(1 / model.phaseIndex), width: 0.33, floor: iso ? 1 : 0.1,
                           color: [0.28, 0.58, 1.0], intensity: iso ? 0.35 : 0.6, lambda});
            }
          }
        } else if (st.kind === "static" && S["show-source"]) {
          const at = s.kind === "point" ? s.position : s.centre;
          glows.push(...at, 2.5, 1.0, 0.8, 0.5, 1.2);
        }
      }
    }
    let allGlows = glows;
    if (S["show-snow"] && snow) {
      allGlows = new Float32Array(glows.length + snow.length);
      allGlows.set(snow, 0);
      allGlows.set(glows, snow.length);
    } else allGlows = new Float32Array(glows);
    const fog = S.fog;
    return {
      head: headPoint,
      scene: {
        background: {top: [0.012, 0.032, 0.060], bottom: [0.0015, 0.004, 0.009], glow: [0.010, 0.030, 0.050], glowAt: [0.5, 1.15]},
        fog: {distance: fog, amount: 0.85, color: [0.004, 0.012, 0.022]},
        lines: staticScene.lines, beads: staticScene.beads, glows: allGlows,
        additiveLines: new Float32Array(additive), shells, halo: 4,
        post: {bloom: S.bloom, exposure: S.exposure, threshold: 0.55, vignette: 0.4},
      },
    };
  }

  // ------------------------------------------------------------- HUD
  function niceStep(span, target) {
    const raw = span / Math.max(1, target), mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / mag;
    return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * mag;
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }

  // Waveform of one OM: per-order density (signal / ns) with a time cursor.
  function plotWaveform(ctx, x, y, w, h, i, rel, k, compact) {
    const edges = model.edges, rows = model.rows[i], widths = edges.slice(1).map((e, j) => e - edges[j]);
    const series = [0, 1, 2].map(p => rows.map((v, j) => v[p] / widths[j]));
    const total = rows.map((v, j) => (v[0] + v[1] + v[2]) / widths[j]);
    let first = 0, last = rows.length - 1, peak = 0;
    for (const v of total) peak = Math.max(peak, Math.abs(v));
    if (compact && peak > 0) {
      while (first < last && Math.abs(total[first]) < 2e-3 * peak) first++;
      while (last > first && Math.abs(total[last]) < 2e-3 * peak) last--;
      first = Math.max(0, first - 2); last = Math.min(rows.length - 1, last + 6);
    }
    const xa = edges[first], xb = edges[last + 1];
    let ylo = 0, yhi = 1e-12;
    for (const s of [...series, total]) for (let j = first; j <= last; j++) {ylo = Math.min(ylo, s[j]); yhi = Math.max(yhi, s[j])}
    yhi *= 1.12; ylo = ylo < 0 ? ylo * 1.12 : 0;
    const pad = {l: 46 * k, r: 10 * k, t: 10 * k, b: 24 * k};
    const X = t => x + pad.l + (t - xa) / (xb - xa) * (w - pad.l - pad.r);
    const Y = v => y + h - pad.b - (v - ylo) / (yhi - ylo) * (h - pad.t - pad.b);
    ctx.save();
    ctx.font = `${11 * k}px Inter, system-ui, sans-serif`;
    ctx.strokeStyle = "rgba(120,150,180,0.16)"; ctx.fillStyle = "rgba(160,185,210,0.8)"; ctx.lineWidth = k;
    const xs = niceStep(xb - xa, 5);
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let t = Math.ceil(xa / xs) * xs; t <= xb; t += xs) {
      ctx.beginPath(); ctx.moveTo(X(t), y + pad.t); ctx.lineTo(X(t), y + h - pad.b); ctx.stroke();
      ctx.fillText(fmt(Math.abs(t) < 1e-9 * xs ? 0 : t, 4), X(t), y + h - pad.b + 5 * k);
    }
    const ys = niceStep(yhi - ylo, 4);
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let v = Math.ceil(ylo / ys) * ys; v <= yhi; v += ys) {
      ctx.beginPath(); ctx.moveTo(x + pad.l, Y(v)); ctx.lineTo(x + w - pad.r, Y(v)); ctx.stroke();
      ctx.fillText(fmt(Math.abs(v) < 1e-9 * ys ? 0 : v, 3), x + pad.l - 6 * k, Y(v));
    }
    const step = (s, colour, width) => {
      ctx.strokeStyle = colour; ctx.lineWidth = width * k; ctx.beginPath();
      for (let j = first; j <= last; j++) {
        if (j === first) ctx.moveTo(X(edges[j]), Y(s[j])); else ctx.lineTo(X(edges[j]), Y(s[j]));
        ctx.lineTo(X(edges[j + 1]), Y(s[j]));
      }
      ctx.stroke();
    };
    const w8 = model.weights;
    [0, 1, 2].forEach(p => {if (w8[p]) step(series[p], css(COMPONENT[p]), 1.4)});
    step(total, "rgba(240,246,252,0.95)", 1.8);
    if (Number.isFinite(rel) && rel >= xa && rel <= xb) {
      ctx.strokeStyle = "rgba(245,200,117,0.9)"; ctx.setLineDash([4 * k, 3 * k]); ctx.lineWidth = 1.2 * k;
      ctx.beginPath(); ctx.moveTo(X(rel), y + pad.t); ctx.lineTo(X(rel), y + h - pad.b); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();
    return {xa, xb};
  }

  function hudInfo(time, cam, mats) {
    const ev = data.events[eventIndex];
    return {
      time, cam, mats,
      title: S["title-text"] || ev.label || ev.event_id,
      subtitle: S["subtitle-text"] || `${det().label}${data.medium?.provenance ? ` · ${data.medium.provenance}` : ""}`,
    };
  }

  function drawHud(ctx, w, h, k, info) {
    ctx.save();
    ctx.textBaseline = "alphabetic";
    const m = 34 * k;
    ctx.shadowColor = "rgba(0,0,0,0.6)"; ctx.shadowBlur = 8 * k;
    if (S["hud-title"]) {
      ctx.fillStyle = "#eef5fc"; ctx.font = `600 ${27 * k}px Inter, system-ui, sans-serif`;
      ctx.fillText(info.title, m, m + 22 * k, w * 0.62);
      ctx.fillStyle = "rgba(170,195,220,0.85)"; ctx.font = `${16 * k}px Inter, system-ui, sans-serif`;
      ctx.fillText(info.subtitle, m, m + 48 * k, w * 0.62);
    }
    if (S["hud-clock"]) {
      const since = info.time - tRef;
      ctx.textAlign = "right";
      ctx.fillStyle = "#f6f9fc"; ctx.font = `500 ${30 * k}px "SF Mono", ui-monospace, Menlo, monospace`;
      const text = `${since >= 0 ? "+" : "−"}${Math.abs(since) >= 1000 ? (Math.abs(since) / 1000).toFixed(3) + " µs" : Math.abs(since).toFixed(0) + " ns"}`;
      ctx.fillText(text, w - m, m + 24 * k);
      ctx.fillStyle = "rgba(170,195,220,0.8)"; ctx.font = `${13 * k}px Inter, system-ui, sans-serif`;
      const s = primarySource();
      ctx.fillText(s?.timing ? "detector clock · since source start" : s?.emission ? "detector clock · since emission (estimated)" : "detector clock · since first light", w - m, m + 46 * k);
      const f = clamp((info.time - model.t0) / (model.t1 - model.t0), 0, 1);
      ctx.shadowBlur = 0;
      ctx.fillStyle = "rgba(120,150,180,0.25)"; ctx.fillRect(w - m - 220 * k, m + 56 * k, 220 * k, 2.5 * k);
      ctx.fillStyle = "rgba(245,200,117,0.95)"; ctx.fillRect(w - m - 220 * k, m + 56 * k, 220 * k * f, 2.5 * k);
      ctx.shadowBlur = 8 * k;
      ctx.textAlign = "left";
    }
    let bottom = h - m;
    if (S["hud-note"]) {
      ctx.fillStyle = "rgba(150,175,200,0.75)"; ctx.font = `${12.5 * k}px Inter, system-ui, sans-serif`;
      const parts = ["LightHit radiative transfer · glow = expected photoelectrons, not sampled hits"];
      if (S["show-front"] && model.sources.length) parts.push("light front illustrative");
      ctx.fillText(parts.join(" · "), m, bottom);
      bottom -= 26 * k;
    }
    if (S["hud-legend"]) {
      ctx.font = `${13 * k}px Inter, system-ui, sans-serif`;
      if (S.colour === "components") {
        let x = m;
        COMPONENT.forEach((c, p) => {
          if (!model.weights[p]) return;
          ctx.fillStyle = css(c); ctx.beginPath(); ctx.arc(x + 6 * k, bottom - 5 * k, 6 * k, 0, TAU); ctx.fill();
          ctx.fillStyle = "rgba(220,232,244,0.9)"; ctx.fillText(COMPONENT_NAMES[p], x + 18 * k, bottom);
          x += ctx.measureText(COMPONENT_NAMES[p]).width + 40 * k;
        });
      } else {
        const bw = 240 * k, bh = 8 * k, y = bottom - bh - 2 * k;
        const g = ctx.createLinearGradient(m, 0, m + bw, 0);
        for (let s = 0; s <= 1.0001; s += 0.1) g.addColorStop(s, css(S.colour === "charge" ? iceColour(s) : timeColour(s)));
        ctx.shadowBlur = 0; ctx.fillStyle = g; ctx.fillRect(m, y, bw, bh); ctx.shadowBlur = 8 * k;
        ctx.fillStyle = "rgba(220,232,244,0.9)";
        if (S.colour === "charge") {
          ctx.fillText(`${fmt(threshold, 2)}`, m, y - 8 * k);
          ctx.textAlign = "right"; ctx.fillText(`${fmt(peakVisible, 3)} p.e.`, m + bw, y - 8 * k); ctx.textAlign = "left";
          ctx.fillStyle = "rgba(150,175,200,0.8)"; ctx.fillText("expected charge (log)", m + bw + 14 * k, y + bh);
        } else {
          ctx.fillText(`${fmt(colourRange[0] - tRef, 4)} ns`, m, y - 8 * k);
          ctx.textAlign = "right"; ctx.fillText(`${fmt(colourRange[1] - tRef, 4)} ns`, m + bw, y - 8 * k); ctx.textAlign = "left";
          ctx.fillStyle = "rgba(150,175,200,0.8)"; ctx.fillText("first light · size ∝ expected charge", m + bw + 14 * k, y + bh);
        }
      }
      bottom -= 44 * k;
    }
    if (S["hud-scale"] && info.mats) {
      const ppm = info.mats.proj[5] * 0.5 * h / info.cam.distance;
      const choices = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000];
      let L = choices[0];
      for (const c of choices) if (c * ppm <= 200 * k) L = c;
      const len = L * ppm, y = bottom;
      ctx.strokeStyle = "rgba(220,232,244,0.85)"; ctx.lineWidth = 2 * k;
      ctx.beginPath(); ctx.moveTo(m, y - 6 * k); ctx.lineTo(m, y); ctx.lineTo(m + len, y); ctx.lineTo(m + len, y - 6 * k); ctx.stroke();
      ctx.fillStyle = "rgba(220,232,244,0.9)"; ctx.font = `${13 * k}px Inter, system-ui, sans-serif`;
      ctx.fillText(`${L} m at the view centre`, m + len + 12 * k, y);
    }
    if (S["hud-inset"] && model.relevant.includes(selected)) {
      const iw = 430 * k, ih = 180 * k, x = w - m - iw, y = h - m - ih;
      ctx.shadowBlur = 0;
      ctx.fillStyle = "rgba(4,10,18,0.72)"; roundRect(ctx, x, y, iw, ih, 8 * k); ctx.fill();
      ctx.strokeStyle = "rgba(90,130,170,0.35)"; ctx.lineWidth = k; ctx.stroke();
      ctx.fillStyle = "rgba(220,232,244,0.9)"; ctx.font = `${12.5 * k}px Inter, system-ui, sans-serif`;
      ctx.fillText(`${moduleName(selected)} · ${data.units.signal} / ns`, x + 12 * k, y + 20 * k);
      plotWaveform(ctx, x + 4 * k, y + 26 * k, iw - 8 * k, ih - 30 * k, selected, info.time - model.origins[selected], k * 0.95, true);
    }
    ctx.restore();
  }

  // ------------------------------------------------------------- drawing
  const glCanvas = $("gl"), hudCanvas = $("hud"), hudCtx = hudCanvas.getContext("2d");

  function render(time, cam, {width, height} = {}) {
    if (width) {glCanvas.width = width; glCanvas.height = height}
    const {scene, head} = compose(time);
    const view = copyCamera(cam);
    if (view.follow && head) view.target = head;
    const mats = renderer.render(scene, glCamera(view));
    return {mats, view};
  }

  function draw() {
    if (!renderer || !model || exporting) return;
    const {mats, view} = render(T, camera);
    lastMats = mats;
    const k = dpr() * clamp(hudCanvas.clientHeight / 950, 0.7, 1.15);
    hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
    drawHud(hudCtx, hudCanvas.width, hudCanvas.height, k, hudInfo(T, view, mats));
    updateClock();
    dirty = false;
  }

  function updateClock() {
    const since = T - tRef;
    $("clock").textContent = `${since >= 0 ? "+" : "−"}${fmt(Math.abs(since), 5)} ns · T=${fmt(T, 6)}`;
    const value = Math.round(1000 * clamp((T - model.t0) / (model.t1 - model.t0), 0, 1));
    if (Number($("time").value) !== value) $("time").value = value;
  }

  function resize() {
    const stage = $("stage"), r = dpr();
    const w = Math.max(2, Math.round(stage.clientWidth * r)), h = Math.max(2, Math.round(stage.clientHeight * r));
    for (const c of [glCanvas, hudCanvas]) if (c.width !== w || c.height !== h) {c.width = w; c.height = h}
    const act = $("activity");
    act.width = Math.round(act.clientWidth * r); act.height = Math.round(act.clientHeight * r);
    drawActivity();
    drawPanels();
    dirty = true;
  }

  function drawActivity() {
    const c = $("activity"), ctx = c.getContext("2d");
    if (!model || !c.width) return;
    ctx.clearRect(0, 0, c.width, c.height);
    const a = model.activity, peak = Math.max(...a) || 1, w = c.width, h = c.height;
    const g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, "rgba(92,198,255,0.55)"); g.addColorStop(1, "rgba(92,198,255,0.05)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.moveTo(0, h);
    a.forEach((v, k) => ctx.lineTo(k / (a.length - 1) * w, h - 2 - (h - 6) * Math.sqrt(v / peak)));
    ctx.lineTo(w, h); ctx.fill();
    ctx.fillStyle = "rgba(120,150,180,0.12)"; ctx.fillRect(0, h - 1, w, 1);
    for (const s of model.sources) {
      const t = s.timing ? s.timing.t0 : s.emission?.t;
      if (!Number.isFinite(t)) continue;
      const x = (t - model.t0) / (model.t1 - model.t0) * w;
      ctx.fillStyle = "rgba(245,200,117,0.8)"; ctx.fillRect(x - 1, 0, 2, h);
    }
  }

  // ------------------------------------------------------------- side panels
  function fillModules() {
    const list = model.relevant.filter(i => visibleSet.has(i)).sort((a, b) => model.window[b] - model.window[a]);
    const node = $("module");
    node.replaceChildren();
    for (const i of list) {
      const o = document.createElement("option");
      o.value = i; o.textContent = `${moduleName(i)} · ${fmt(model.window[i], 3)}`;
      node.append(o);
    }
    if (list.length && !list.includes(selected)) selected = list[0];
    node.value = selected;
  }

  function drawPanels() {
    if (!model || !document.querySelector('.tab.active[data-tab="module"]')) return;
    const r = dpr();
    for (const id of ["waveform", "timemap"]) {
      const c = $(id);
      const w = Math.round(c.clientWidth * r), h = Math.round(c.clientHeight * r);
      if (c.width !== w || c.height !== h) {c.width = w; c.height = h}
    }
    const wf = $("waveform"), wctx = wf.getContext("2d");
    wctx.clearRect(0, 0, wf.width, wf.height);
    if (model.relevant.includes(selected)) plotWaveform(wctx, 0, 0, wf.width, wf.height, selected, T - model.origins[selected], r, false);
    drawTimemap();
    const ev = data.events[eventIndex], q = ev.charge_components[selected];
    const rows = [["Module", moduleName(selected)],
      ["Integrated charge", `${fmt(q[0] + q[1] + q[2])} ${data.units.signal}`],
      ["0 / 1 / ≥2", q.map(v => fmt(v)).join(" / ")],
      ["Selected window charge", `${fmt(model.window[selected])} ${data.units.signal}`],
      ["First light", Number.isFinite(model.first[selected]) ? `${fmt(model.first[selected] - tRef, 5)} ns` : "—"],
      ["Time origin", `${fmt(model.origins[selected], 6)} ns`],
      ["Full time spectrum", !ev.active || ev.active[selected] ? "computed" : "skipped: Q below threshold"]];
    $("details").replaceChildren(...rows.flatMap(([a, b]) => {
      const dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.textContent = a; dd.textContent = b; return [dt, dd];
    }));
  }

  let timemapRows = [];
  function drawTimemap() {
    const c = $("timemap"), ctx = c.getContext("2d"), w = c.width, h = c.height;
    ctx.fillStyle = "#07111b"; ctx.fillRect(0, 0, w, h);
    timemapRows = model.relevant.filter(i => visibleSet.has(i) && model.window[i] >= threshold && Number.isFinite(model.first[i]))
      .sort((a, b) => model.first[a] - model.first[b]);
    if (!timemapRows.length) return;
    const rowH = h / timemapRows.length, span = model.t1 - model.t0;
    let peak = 0;
    for (const i of timemapRows) for (let j = 0; j < model.bins; j++) {
      const v = model.rows[i][j], s = model.weights[0] * v[0] + model.weights[1] * v[1] + model.weights[2] * v[2];
      peak = Math.max(peak, Math.abs(s) / (model.edges[j + 1] - model.edges[j]));
    }
    const scale = peak * 1e-4;
    timemapRows.forEach((i, row) => {
      for (let j = 0; j < model.bins; j++) {
        const v = model.rows[i][j], s = (model.weights[0] * v[0] + model.weights[1] * v[1] + model.weights[2] * v[2]) / (model.edges[j + 1] - model.edges[j]);
        if (!s) continue;
        const f = Math.log10(1 + Math.abs(s) / scale) / Math.log10(1 + peak / scale);
        ctx.fillStyle = s > 0 ? css(iceColour(f)) : `rgba(214,90,150,${0.25 + 0.75 * f})`;
        const x0 = (model.origins[i] + model.edges[j] - model.t0) / span * w, x1 = (model.origins[i] + model.edges[j + 1] - model.t0) / span * w;
        ctx.fillRect(x0, row * rowH, Math.max(1, x1 - x0), Math.max(1, rowH));
      }
      if (i === selected) {ctx.strokeStyle = "rgba(245,200,117,0.9)"; ctx.strokeRect(0.5, row * rowH, w - 1, Math.max(1, rowH))}
    });
    const x = (T - model.t0) / span * w;
    ctx.fillStyle = "rgba(245,200,117,0.9)"; ctx.fillRect(x - 1, 0, 2, h);
  }

  function updateInfo() {
    const ev = data.events[eventIndex], q = ev.charge_components.reduce((a, v) => a.map((x, p) => x + v[p]), [0, 0, 0]);
    const s = primarySource(), out = [];
    const p = (html) => {const node = document.createElement("p"); node.innerHTML = html; out.push(node)};
    const esc = x => String(x).replace(/[&<>"]/g, ch => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"})[ch]);
    p(`<strong>${esc(ev.label || ev.event_id)}</strong><br>${esc(det().label)}${data.medium?.provenance ? ` · ${esc(data.medium.provenance)}` : ""}`);
    p(`Integrated array charge: <strong>${fmt(q[0] + q[1] + q[2])} ${esc(data.units.signal)}</strong> (0 / 1 / ≥2 = ${q.map(v => fmt(v)).join(" / ")}). ` +
      `OMs with stored signal: ${model.relevant.length} of ${model.n}. Computation: ${fmt(ev.diagnostics?.elapsed_seconds || 0)} s.`);
    if (ev.computed_orders && !ev.computed_orders.includes(2)) p("Prompt transport: order ≥2 was not computed (it is not zero).");
    p("The glow of an OM is the <strong>expected</strong> photoelectron response from radiative transfer, accumulated on the detector clock; it is not a sampled hit. " +
      "Glow sizes use the positive part of the selected signal; waveforms and the time map keep signed Fourier bins.");
    if (s) {
      const timing = s.timing ? "emission times from the stored axial timing" : s.emission ? "emission time estimated from direct-light arrivals (display only)" : "no emission time";
      p(`Source: ${esc(s.label)} · ${timing}.`);
    }
    const thetaC = Math.acos(1 / model.phaseIndex) * 180 / Math.PI;
    p(`Light front (illustrative): Cherenkov angle ${fmt(thetaC, 3)}° for n = ${model.phaseIndex}, speed c/n<sub>g</sub> with n<sub>g</sub> = ${fmt(model.groupIndex, 3)}; ` +
      `for a β = 1 track the front is a cone of half-angle ${fmt(model.frontAngle * 180 / Math.PI, 3)}° around the backward axis. Its fade length is a display setting.`);
    if (ev.diagnostics?.note) p(esc(ev.diagnostics.note));
    $("info").replaceChildren(...out);
  }

  // ------------------------------------------------------------- camera
  function fit(view = "auto") {
    const d = det();
    // The event view frames OMs that receive light; "array" frames the detector.
    let ids = visible;
    if (view !== "array") {
      const lit = model.relevant.filter(i => visibleSet.has(i) && model.window[i] >= threshold);
      if (lit.length >= 2) ids = lit;
    }
    const pos = ids.map(i => d.positions_m[i]);
    const lo = [0, 1, 2].map(k => Math.min(...pos.map(p => p[k]))), hi = [0, 1, 2].map(k => Math.max(...pos.map(p => p[k])));
    const centre = lo.map((v, k) => (v + hi[k]) / 2);
    let radius = 0.5 * Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 50;
    if (ids !== visible) radius = Math.max(1.25 * radius, 90);
    const aspect = glCanvas.width / Math.max(1, glCanvas.height);
    const half = Math.min(camera.fov / 2, Math.atan(Math.tan(camera.fov / 2) * aspect));
    const target = {target: centre, distance: radius / Math.sin(half) * 0.92, fov: camera.fov, azimuth: camera.azimuth, elevation: camera.elevation, follow: false};
    const axis = sourceAxis();
    const sideAz = axis && Math.hypot(axis[0], axis[1]) > 0.15 ? Math.atan2(axis[1], axis[0]) - Math.PI / 2 : -Math.PI / 3;
    if (view === "auto" || view === "side" || view === "array") {target.azimuth = sideAz; target.elevation = view === "array" ? 0.32 : 0.18}
    else if (view === "top") {target.elevation = 1.53}
    else if (view === "along" && axis) {
      target.azimuth = Math.atan2(-axis[1], -axis[0]) + 0.25;
      target.elevation = clamp(Math.asin(clamp(-axis[2], -1, 1)) + 0.15, -1.4, 1.4);
    } else if (view === "follow") {
      target.azimuth = sideAz; target.elevation = 0.2; target.distance = Math.min(target.distance, 380); target.follow = true;
    }
    return target;
  }

  function moveCamera(to, seconds = 0.8) {
    stopPreview();
    syncControls();
    const from = copyCamera(camera);
    let da = to.azimuth - from.azimuth;
    da = ((da + Math.PI) % TAU + TAU) % TAU - Math.PI;
    tween = {from, to: {...to, azimuth: from.azimuth + da}, start: performance.now(), seconds};
    if (seconds <= 0) applyTween(1);
  }
  function applyTween(f) {
    const e = ease(f), a = tween.from, b = tween.to;
    camera.target = a.target.map((v, k) => lerp(v, b.target[k], e));
    camera.azimuth = lerp(a.azimuth, b.azimuth, e);
    camera.elevation = lerp(a.elevation, b.elevation, e);
    camera.distance = Math.exp(lerp(Math.log(a.distance), Math.log(b.distance), e));
    camera.follow = f >= 1 ? b.follow : false;
    if (f >= 1) tween = null;
    dirty = true;
  }

  // ------------------------------------------------------------- movie plan
  function moviePlan() {
    const intro = S.intro, event = S["event-seconds"], outro = S.outro;
    return {intro, event, outro, total: intro + event + outro, pace: model.pacing(S.pacing),
            base: copyCamera(camera), motion: S["camera-motion"], orbit: S.orbit * Math.PI / 180,
            zoom: S.zoom, keys: keyframes.map(copyCamera)};
  }
  function catmull(p0, p1, p2, p3, t) {
    const t2 = t * t, t3 = t2 * t;
    return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
  }
  function movieAt(s, plan) {
    const time = s < plan.intro ? model.t0 : s < plan.intro + plan.event ? plan.pace.time((s - plan.intro) / plan.event) : model.t1;
    const g = plan.total > 0 ? clamp(s / plan.total, 0, 1) : 0, e = ease(g);
    const cam = copyCamera(plan.base);
    cam.follow = false;
    if (plan.motion === "orbit" || plan.motion === "follow") {
      cam.azimuth = plan.base.azimuth + plan.orbit * e;
      cam.distance = plan.base.distance * lerp(1, plan.zoom, e);
      if (plan.motion === "follow") cam.follow = true;
    } else if (plan.motion === "keys" && plan.keys.length >= 2) {
      const keys = plan.keys, n = keys.length - 1, x = e * n, k = Math.min(n - 1, Math.floor(x)), t = x - k;
      const at = j => keys[clamp(j, 0, n)];
      const unwrap = [keys[0].azimuth];
      for (let j = 1; j <= n; j++) {
        let da = keys[j].azimuth - unwrap[j - 1];
        da = ((da + Math.PI) % TAU + TAU) % TAU - Math.PI;
        unwrap.push(unwrap[j - 1] + da);
      }
      const az = j => unwrap[clamp(j, 0, n)];
      cam.azimuth = catmull(az(k - 1), az(k), az(k + 1), az(k + 2), t);
      cam.elevation = clamp(catmull(at(k - 1).elevation, at(k).elevation, at(k + 1).elevation, at(k + 2).elevation, t), -1.5, 1.5);
      cam.distance = Math.exp(catmull(...[k - 1, k, k + 1, k + 2].map(j => Math.log(at(j).distance)), t));
      cam.target = [0, 1, 2].map(c => catmull(at(k - 1).target[c], at(k).target[c], at(k + 1).target[c], at(k + 2).target[c], t));
    }
    return {time, cam};
  }

  function startPreview() {
    stopPlay();
    syncControls();
    preview = {plan: moviePlan(), start: performance.now()};
    $("preview").textContent = "Stop preview";
    document.body.classList.add("presenting");
  }
  function stopPreview() {
    if (!preview) return;
    Object.assign(camera, preview.plan.base);
    preview = null;
    $("preview").textContent = "Preview";
    document.body.classList.remove("presenting");
    dirty = true;
  }

  // ------------------------------------------------------------- export
  function frameSize() {
    const [w, h] = S["movie-size"].split("x").map(Number);
    return [w, h];
  }

  function busy(title, text, fraction) {
    $("busy").hidden = false;
    $("busy-title").textContent = title;
    $("busy-text").textContent = text;
    $("busy-bar").style.width = `${Math.round(100 * clamp(fraction, 0, 1))}%`;
  }

  async function renderMovie() {
    if (exporting) return;
    const status = $("movie-status");
    if (!Movie.supported()) {status.textContent = "This browser has no WebCodecs video encoder; use a current Chrome, Edge or Safari."; return}
    stopPlay(); stopPreview();
    const [width, height] = frameSize(), fps = Number(S["movie-fps"]), plan = moviePlan();
    const frames = Math.max(1, Math.round(plan.total * fps));
    const encoder = new Movie.MovieEncoder({width, height, fps, bitrate: Movie.defaultBitrate(width, height, fps)});
    exporting = {cancelled: false};
    const out = document.createElement("canvas");
    out.width = width; out.height = height;
    const ctx = out.getContext("2d");
    const started = performance.now();
    try {
      await encoder.open();
      busy("Rendering movie", "Preparing encoder…", 0);
      for (let f = 0; f < frames; f++) {
        if (exporting.cancelled) throw Error("Rendering cancelled.");
        const st = movieAt(f / fps, plan);
        const {mats, view} = render(st.time, st.cam, {width, height});
        ctx.drawImage(glCanvas, 0, 0);
        drawHud(ctx, width, height, height / 1080 * (width < height ? 1.25 : 1), hudInfo(st.time, view, mats));
        await encoder.addFrame(out);
        if (f % 6 === 0) {
          const elapsed = (performance.now() - started) / 1000, left = elapsed / (f + 1) * (frames - f - 1);
          busy("Rendering movie", `Frame ${f + 1} of ${frames} · ${width}×${height} at ${fps} fps · about ${Math.ceil(left)} s left`, (f + 1) / frames);
          await new Promise(r => setTimeout(r, 0));
        }
      }
      busy("Rendering movie", "Finishing the MP4 file…", 1);
      const blob = await encoder.finish();
      const name = Movie.filename(`${data.events[eventIndex].event_id}-${width}x${height}-${fps}fps`);
      if (lastMovieUrl) URL.revokeObjectURL(lastMovieUrl);
      lastMovieUrl = Movie.downloadBlob(blob, name);
      const link = $("movie-download");
      link.href = lastMovieUrl; link.download = name; link.hidden = false;
      let saved = "";
      if (Movie.localSaveEnabled()) {
        try {saved = ` · saved locally as ${(await Movie.saveLocal(blob, name)).saved_as}`}
        catch (error) {saved = ` · local save failed: ${error.message}`}
      }
      status.textContent = `Ready · ${name} · ${(blob.size / 1048576).toFixed(1)} MB · ${frames} frames in ${((performance.now() - started) / 1000).toFixed(0)} s${saved}`;
      return {name, bytes: blob.size, frames, blob};
    } catch (error) {
      encoder.cancel();
      status.textContent = error.message;
      return {error: error.message};
    } finally {
      exporting = null;
      $("busy").hidden = true;
      resize();
      draw();
    }
  }

  // Composite one frame (scene and overlay) at a given size; returns a canvas.
  function composeFrame(time, cam, width, height) {
    const out = document.createElement("canvas");
    out.width = width; out.height = height;
    const ctx = out.getContext("2d");
    const {mats, view} = render(time, cam, {width, height});
    ctx.drawImage(glCanvas, 0, 0);
    drawHud(ctx, width, height, height / 1080 * (width < height ? 1.25 : 1), hudInfo(time, view, mats));
    return out;
  }

  // Data URL of a still frame, for scripted use; nothing is downloaded.
  async function still({time = T, width, height} = {}) {
    const [mw, mh] = frameSize();
    exporting = {cancelled: false};
    try {return composeFrame(time, camera, width || mw, height || mh).toDataURL("image/png")}
    finally {exporting = null; resize(); draw()}
  }

  async function snapshot() {
    if (exporting) return;
    const [mw, mh] = frameSize(), scale = Math.min(2, 4096 / Math.max(mw, mh));
    const width = Math.round(mw * scale / 2) * 2, height = Math.round(mh * scale / 2) * 2;
    exporting = {cancelled: false};
    try {
      const out = composeFrame(T, camera, width, height);
      const blob = await new Promise(r => out.toBlob(r, "image/png"));
      const name = Movie.filename(`${data.events[eventIndex].event_id}-${Math.round(T)}ns-${width}x${height}`, "png");
      URL.revokeObjectURL(Movie.downloadBlob(blob, name));
      if (Movie.localSaveEnabled()) {try {await Movie.saveLocal(blob, name)} catch (_) {/* download remains */}}
      setStatus(`Saved still · ${name}`);
      return {name, bytes: blob.size};
    } finally {
      exporting = null;
      resize();
      draw();
    }
  }

  // ------------------------------------------------------------- playback
  function setTime(time) {
    T = clamp(time, model.t0, model.t1);
    dirty = true;
  }
  function startPlay() {
    stopPreview();
    pace = model.pacing(S.pacing);
    if (T >= model.t1 - 1e-9) T = model.t0;
    playU = pace.progress(T);
    playing = {last: performance.now()};
    $("play").textContent = "❚❚";
  }
  function stopPlay() {
    playing = false;
    $("play").textContent = "▶";
  }

  function loop(now) {
    if (!exporting) {
      if (steering && !preview) controls.update();  // a gesture and its inertia
      if (tween) applyTween(clamp((now - tween.start) / 1000 / tween.seconds, 0, 1));
      if (playing) {
        const dt = (now - playing.last) / 1000;
        playing.last = now;
        playU += dt / Number(S["play-seconds"]);
        if (playU >= 1) {playU = 1; stopPlay()}
        setTime(pace.time(playU));
      }
      if (preview) {
        const s = (now - preview.start) / 1000;
        const st = movieAt(s, preview.plan);
        T = st.time; Object.assign(camera, st.cam);
        dirty = true;
        if (s > preview.plan.total) stopPreview();
      }
      if (dirty) {draw(); if (playing || preview) drawPanelsThrottled(now)}
    }
    requestAnimationFrame(loop);
  }
  let lastPanels = 0;
  function drawPanelsThrottled(now) {if (now - lastPanels > 120) {lastPanels = now; drawPanels()}}

  // ------------------------------------------------------------- input
  function nearestOM(x, y, radius = 14) {
    if (!lastMats) return -1;
    const r = dpr();
    let best = -1, bestD = radius * r;
    for (const i of visible) {
      const p = GL.project(lastMats, det().positions_m[i]);
      if (!p) continue;
      const d = Math.hypot(p.x - x * r, p.y - y * r);
      const lit = omState && omState.q[i] >= threshold;
      const bias = lit ? 0.6 : 1;
      if (d * bias < bestD) {bestD = d * bias; best = i}
    }
    return best;
  }

  // Orbit, pan and pinch come from three.js OrbitControls acting on a proxy
  // camera.  The view model stays (target, azimuth, elevation, distance), so
  // tweens, keyframes, follow mode and movies are unchanged: the proxy is set
  // from it before a gesture, and only a gesture and its inertia are read back.
  let syncing = false, steering = false, held = false;
  function syncControls() {
    if (!controls) return;
    const proxy = controls.object, stage = $("stage");
    syncing = true;
    steering = false;
    const damping = controls.enableDamping;
    controls.enableDamping = false;  // drop any inertia still in flight
    controls.update();
    controls.enableDamping = damping;
    proxy.fov = camera.fov * 180 / Math.PI;
    proxy.aspect = stage.clientWidth / Math.max(1, stage.clientHeight);
    proxy.updateProjectionMatrix();
    proxy.position.fromArray(eye(camera));
    controls.target.fromArray(camera.target);
    controls.update();
    syncing = false;
  }
  function readControls() {
    if (syncing || !steering) return;
    const p = controls.object.position.toArray(), t = controls.target.toArray(), d = sub(p, t);
    const distance = Math.max(norm(d), 1e-6), elevation = Math.asin(clamp(d[2] / distance, -1, 1));
    let da = Math.atan2(d[1], d[0]) - camera.azimuth;
    da = ((da + Math.PI) % TAU + TAU) % TAU - Math.PI;
    const shift = norm(sub(t, camera.target)) / camera.distance;
    const step = Math.max(Math.abs(da), Math.abs(elevation - camera.elevation), Math.abs(Math.log(distance / camera.distance)), shift);
    if (step < 1e-5) {
      if (!held) steering = false;  // the inertia has died out
      return;
    }
    if (shift > 1e-5) camera.follow = false;  // a pan leaves follow mode
    camera.target = t;
    camera.azimuth += da;
    camera.elevation = elevation;
    camera.distance = distance;
    tween = null;
    dirty = true;
    $("tooltip").hidden = true;
  }

  function setupInput() {
    const stage = $("stage");
    let drag = null;
    const proxy = new THREE.PerspectiveCamera();
    proxy.up.set(0, 0, 1);
    proxy.position.fromArray(eye(camera));
    controls = new THREE.OrbitControls(proxy, stage);
    Object.assign(controls, {enableDamping: true, dampingFactor: 0.14, rotateSpeed: 0.8, panSpeed: 1, zoomSpeed: 1,
      minDistance: 5, maxDistance: 50000, minPolarAngle: Math.PI / 2 - 1.53, maxPolarAngle: Math.PI / 2 + 1.45});
    controls.addEventListener("change", readControls);
    controls.addEventListener("start", () => {steering = true; held = true});
    controls.addEventListener("end", () => {held = false});
    // Capture-phase listeners run before the controls on the same element;
    // buttons and the progress card on the stage never start a gesture.
    const grab = () => {stopPreview(); tween = null; syncControls()};
    const overlays = ".view-buttons, #busy";
    stage.addEventListener("pointerdown", e => {if (!e.target.closest(overlays)) grab()}, {capture: true});
    for (const node of stage.querySelectorAll(overlays)) node.addEventListener("pointerdown", e => e.stopPropagation());
    stage.addEventListener("contextmenu", e => e.preventDefault());
    stage.addEventListener("pointerdown", e => {
      drag = {x: e.clientX, y: e.clientY, moved: 0};
      stage.classList.add("dragging");
    });
    stage.addEventListener("pointermove", e => {
      const rect = stage.getBoundingClientRect();
      if (drag) {
        drag.moved += Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y);
        drag.x = e.clientX; drag.y = e.clientY;
        $("tooltip").hidden = true;
        return;
      }
      const i = nearestOM(e.clientX - rect.left, e.clientY - rect.top);
      const tip = $("tooltip");
      if (i !== hoverIndex) {
        hoverIndex = i;
        if (i < 0) tip.hidden = true;
        else {
          const q = omState ? omState.q[i] : 0, first = model.first[i];
          tip.textContent = `${moduleName(i)} · Q(t) = ${fmt(q, 3)} / ${fmt(model.window[i], 3)}${Number.isFinite(first) ? ` · first light ${fmt(first - tRef, 4)} ns` : ""}`;
          tip.hidden = false;
        }
      }
      tip.style.left = `${e.clientX - rect.left}px`; tip.style.top = `${e.clientY - rect.top}px`;
    });
    const end = e => {
      if (!drag) return;
      const rect = stage.getBoundingClientRect();
      if (drag.moved < 4) {
        const i = nearestOM(e.clientX - rect.left, e.clientY - rect.top);
        if (i >= 0) choose(i);
      }
      drag = null;
      stage.classList.remove("dragging");
    };
    stage.addEventListener("pointerup", end);
    stage.addEventListener("pointercancel", end);
    stage.addEventListener("pointerleave", () => {$("tooltip").hidden = true; hoverIndex = -1});
    stage.addEventListener("dblclick", e => {
      const rect = stage.getBoundingClientRect(), i = nearestOM(e.clientX - rect.left, e.clientY - rect.top);
      if (i < 0) return;
      moveCamera({...copyCamera(camera), target: det().positions_m[i].slice(), distance: Math.min(camera.distance, 220), follow: false});
    });
    // Wheel and trackpad zoom stay proportional to the scroll distance.
    stage.addEventListener("wheel", e => {
      e.preventDefault();
      e.stopImmediatePropagation();
      grab();
      camera.distance = clamp(camera.distance * Math.exp(e.deltaY * (e.deltaMode ? 0.05 : 0.0015)), 5, 50000);
      dirty = true;
    }, {passive: false, capture: true});
    document.querySelectorAll("[data-view]").forEach(b => b.onclick = () => moveCamera(fit(b.dataset.view)));
    window.addEventListener("keydown", e => {
      if (e.target.closest("input[type=text],select") || e.metaKey || e.ctrlKey) return;
      const step = (model.t1 - model.t0) / 200;
      const keys = {
        " ": () => playing ? stopPlay() : startPlay(),
        ArrowRight: () => {stopPlay(); setTime(T + step)}, ArrowLeft: () => {stopPlay(); setTime(T - step)},
        Home: () => {stopPlay(); setTime(model.arrivalRange[0])}, End: () => {stopPlay(); setTime(model.t1)},
        r: () => moveCamera(fit("auto")), 0: () => moveCamera(fit("array")), 1: () => moveCamera(fit("side")), 2: () => moveCamera(fit("top")),
        3: () => moveCamera(fit("along")), 4: () => moveCamera(fit("follow")),
        p: () => snapshot(), f: () => toggleFullscreen(), s: () => toggleSide(),
      };
      const action = keys[e.key] || keys[e.key.toLowerCase()];
      if (action && !(e.target.type === "range" && e.key.startsWith("Arrow"))) {e.preventDefault(); action()}
    });
  }

  function choose(i) {
    selected = i;
    if ([...$("module").options].some(o => Number(o.value) === i)) $("module").value = i;
    dirty = true;
    drawPanels();
  }
  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen(); else $("stage").requestFullscreen?.();
  }
  function toggleSide() {document.body.classList.toggle("side-hidden"); setTimeout(resize, 30)}
  function setStatus(text, error = false) {
    $("status").textContent = text;
    $("status").classList.toggle("error", error);
  }

  function setupControls() {
    for (const [id, kind] of Object.entries(CONTROLS)) {
      const node = $(id), event = node.type === "range" || node.type === "text" ? "input" : "change";
      node.addEventListener(event, () => {
        S[id] = readControl(id);
        saveSettings();
        showValues();
        if (!model) return;
        if (kind === "rebuild") {const keep = T; rebuildModel(); setTime(keep)}
        else if (kind === "visible") {refreshVisible(); updateLevels(); buildStatic(); fillModules()}
        else if (kind === "levels") {updateLevels(); fillModules()}
        else if (kind === "static") buildStatic();
        else if (kind === "pace") pace = model.pacing(S.pacing);
        dirty = true;
        drawPanels();
      });
    }
    $("event").onchange = () => selectEvent(Number($("event").value));
    $("module").onchange = () => choose(Number($("module").value));
    $("time").addEventListener("input", () => {stopPlay(); stopPreview(); setTime(model.t0 + Number($("time").value) / 1000 * (model.t1 - model.t0)); drawPanels()});
    $("play").onclick = () => playing ? stopPlay() : startPlay();
    $("preview").onclick = () => preview ? stopPreview() : startPreview();
    $("render").onclick = () => renderMovie();
    $("snapshot").onclick = () => snapshot();
    $("busy-cancel").onclick = () => {if (exporting) exporting.cancelled = true};
    $("fullscreen").onclick = toggleFullscreen;
    $("toggle-side").onclick = toggleSide;
    $("key-add").onclick = () => {keyframes.push(copyCamera(camera)); $("key-count").textContent = `${keyframes.length} keyframe${keyframes.length === 1 ? "" : "s"}`};
    $("key-clear").onclick = () => {keyframes = []; $("key-count").textContent = "0 keyframes"};
    document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
      document.querySelectorAll(".tabs button, .tab").forEach(n => n.classList.toggle("active", n.dataset.tab === b.dataset.tab));
      drawPanels();
    });
    $("timemap").addEventListener("click", e => {
      const rect = e.target.getBoundingClientRect(), row = Math.floor((e.clientY - rect.top) / rect.height * timemapRows.length);
      if (timemapRows[row] !== undefined) choose(timemapRows[row]);
    });
    $("file").onchange = async e => {
      try {await load(JSON.parse(await e.target.files[0].text()))}
      catch (error) {setStatus(error.message, true)}
    };
    new ResizeObserver(resize).observe($("stage"));
    window.addEventListener("resize", resize);
  }

  function showValues() {
    const put = (id, text) => {const node = $(`${id}-value`); if (node) node.textContent = text};
    put("size", `×${S.size.toFixed(2)}`); put("flash", S.flash.toFixed(2)); put("bloom", S.bloom.toFixed(2));
    put("exposure", S.exposure.toFixed(2)); put("lambda", `${S.lambda}`); put("fog", `${S.fog}`);
    put("intro", `${S.intro}`); put("event-seconds", `${S["event-seconds"]}`); put("outro", `${S.outro}`);
    put("pacing", S.pacing.toFixed(2)); put("orbit", `${S.orbit}`); put("zoom", `×${S.zoom.toFixed(2)}`);
    if (data) $("threshold-value").textContent = `${fmt(Math.pow(10, S.threshold), 2)} p.e.`;
  }

  // ------------------------------------------------------------- loading
  async function selectEvent(index) {
    stopPlay(); stopPreview();
    eventIndex = index;
    $("event").value = index;
    rebuildModel();
    T = model.t1;
    camera.follow = false;
    moveCamera(fit("auto"), 0);
    drawActivity();
    dirty = true;
    drawPanels();
  }

  async function load(x) {
    data = validate(x);
    $("event").replaceChildren(...data.events.map((e, i) => {
      const o = document.createElement("option"); o.value = i; o.textContent = e.label || e.event_id; return o;
    }));
    const clusters = [...new Set(data.detector.cluster_id)].sort((a, b) => a - b);
    $("cluster").replaceChildren(...[["all", "All clusters"], ...clusters.map(c => [c, `Cluster ${c + 1}`])].map(([v, t]) => {
      const o = document.createElement("option"); o.value = v; o.textContent = t; return o;
    }));
    S.cluster = "all";
    refreshVisible();
    const params = new URLSearchParams(location.search);
    const wanted = params.get("event");
    const index = Math.max(0, data.events.findIndex(e => e.event_id === wanted));
    await selectEvent(index);
    setStatus(`${det().label} · ${data.events.length} event${data.events.length === 1 ? "" : "s"} · ${renderer.hdr ? "HDR" : "LDR"} rendering`);
    showValues();
  }

  function start() {
    loadSettings();
    showValues();
    try {renderer = GL.create(glCanvas)}
    catch (error) {setStatus(`${error.message} The legacy viewer remains available from Python.`, true); return}
    setupControls();
    setupInput();
    resize();
    requestAnimationFrame(loop);
    const ready = load(JSON.parse($("result").textContent)).catch(error => setStatus(error.message, true));
    // Small automation surface for scripted renders (e.g. a headless browser).
    window.LightHitViewer = {
      ready, renderMovie, snapshot, still, view: v => moveCamera(fit(v), 0),
      setTime: t => setTime(t), get time() {return T}, get range() {return model && [model.t0, model.t1]},
      selectEvent: id => selectEvent(Math.max(0, data.events.findIndex(e => e.event_id === id))),
      settings: S, camera,
    };
  }

  start();
})();
