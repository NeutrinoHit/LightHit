"use strict";

// Display-only animation of a track or shower axis on the detector clock.
// The stored OM time origins live on the same clock as the source emission
// times, so a bin of OM i covers  origin_i + [edge_j, edge_j+1)  in detector
// time.  Nothing here changes a stored value.
const LightHitViewerSource = (() => {
  const BACKGROUND = [16, 31, 48], GOLD = [245, 200, 117], HOT = [255, 246, 220];
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const mix = (a, b, t) => a.map((x, i) => Math.round(x + (b[i] - x) * t));
  const css = c => `rgb(${c[0]},${c[1]},${c[2]})`;

  // Timing of one source: emission time t(s) at axial position s (metres from
  // source.position_m along source.direction).  Points are ordered by time, so
  // ``reversed`` says the particle moves against source.direction.
  function prepare(source) {
    const timing = source && source.timing;
    if (!timing || !timing.s_m || timing.s_m.length < 2) return null;
    const s = timing.s_m, t = timing.t_ns, n = s.length;
    const weight = timing.weight || s.map(() => 1);
    const reversed = t[n - 1] < t[0];
    const order = [...Array(n).keys()];
    if (reversed) order.reverse();
    const points = order.map(k => ({
      t: t[k], w: weight[k], s: s[k],
      xyz: source.position_m.map((x, j) => x + s[k] * source.direction[j])}));
    const t0 = points[0].t, t1 = points[n - 1].t, spindle = source.shape === "spindle";
    // afterglow: how long an emitted point stays visible (display only)
    return {points, reversed, spindle, t0, t1,
            tau: Math.max(40, (spindle ? 1.5 : 0.35) * (t1 - t0))};
  }

  // Position of the moving front at detector time T, or null outside [t0, t1].
  function head(prep, T) {
    const p = prep.points;
    if (T < prep.t0 || T > prep.t1) return null;
    let lo = 0, hi = p.length - 1;
    while (hi - lo > 1) {const mid = (lo + hi) >> 1; if (p[mid].t <= T) lo = mid; else hi = mid}
    const span = p[hi].t - p[lo].t, u = span > 0 ? (T - p[lo].t) / span : 0;
    return p[lo].xyz.map((x, j) => x + (p[hi].xyz[j] - x) * u);
  }

  // Marker arrays for one frame.  The last point is the muon head (hidden for
  // a spindle).  Fading blends towards the scene background instead of alpha.
  function frame(prep, T) {
    const out = {x: [], y: [], z: [], size: [], color: []};
    const push = (xyz, size, colour) => {
      out.x.push(xyz[0]); out.y.push(xyz[1]); out.z.push(xyz[2]);
      out.size.push(size); out.color.push(css(colour));
    };
    for (const p of prep.points) {
      const age = T - p.t;
      if (age < 0 || age >= prep.tau) {push(p.xyz, 0, BACKGROUND); continue}
      const fade = age / prep.tau;
      const colour = mix(mix(HOT, GOLD, clamp(age / 8, 0, 1)), BACKGROUND, 0.8 * fade);
      push(p.xyz, (prep.spindle ? 4 + 20 * Math.sqrt(Math.max(p.w, 0)) : 7) * (1 - 0.5 * fade), colour);
    }
    const front = prep.spindle ? null : head(prep, T);
    if (front) push(front, 18, HOT); else push(prep.points[prep.points.length - 1].xyz, 0, BACKGROUND);
    return out;
  }

  // Index j of the bin with edges[j] <= rel < edges[j+1]; -1 before the first
  // edge and edges.length - 1 (the bin count) at or after the last.
  function binAt(edges, rel) {
    if (rel < edges[0]) return -1;
    const last = edges.length - 1;
    if (rel >= edges[last]) return last;
    let lo = 0, hi = last;
    while (hi - lo > 1) {const mid = (lo + hi) >> 1; if (edges[mid] <= rel) lo = mid; else hi = mid}
    return lo;
  }

  // Detector-clock interval worth playing: from the source start to the time by
  // which 99.5 % of the absolute signal has arrived, padded a little.
  function clockRange(origins, rows, centers, sourceRange, resolution = 2) {
    let lo = Infinity, hi = -Infinity;
    rows.forEach((row, i) => row.forEach((v, j) => {
      if (v) {const T = origins[i] + centers[j]; lo = Math.min(lo, T); hi = Math.max(hi, T)}}));
    if (!isFinite(lo)) {lo = sourceRange ? sourceRange[0] : 0; hi = sourceRange ? sourceRange[1] : 1}
    const count = clamp(Math.ceil((hi - lo) / resolution) + 1, 1, 20000);
    const width = count > 1 ? (hi - lo) / (count - 1) : 1, hist = new Float64Array(count);
    rows.forEach((row, i) => row.forEach((v, j) => {
      if (v) hist[Math.round((origins[i] + centers[j] - lo) / width)] += Math.abs(v)}));
    const total = hist.reduce((a, b) => a + b, 0);
    let run = 0, first = lo, last = hi, gotFirst = false;
    for (let k = 0; k < count && total > 0; k++) {
      run += hist[k];
      if (!gotFirst && run >= 5e-4 * total) {first = lo + k * width; gotFirst = true}
      if (run >= 0.995 * total) {last = lo + k * width; break}
    }
    if (sourceRange) {first = Math.min(first, sourceRange[0]); last = Math.max(last, sourceRange[1])}
    return {t0: first - 10, t1: last + 20};
  }

  return {prepare, head, frame, binAt, clockRange};
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = LightHitViewerSource;
}
