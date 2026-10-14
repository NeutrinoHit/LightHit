"use strict";

// Display model of one event on the detector clock.  Bin j of OM i covers
// origin_i + [edge_j, edge_j+1) in detector time; within a bin the stored
// integral is spread uniformly.  Every quantity here is derived for display
// only: stored components, signs and charges are never modified.
const LightHitStudioModel = (() => {
  const C_M_PER_NS = 0.299792458;
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const PARTS = {all: [1, 1, 1], prompt: [1, 1, 0], 0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]};

  function binAt(edges, rel) {
    if (rel < edges[0]) return -1;
    const last = edges.length - 1;
    if (rel >= edges[last]) return last;
    let lo = 0, hi = last;
    while (hi - lo > 1) {const mid = (lo + hi) >> 1; if (edges[mid] <= rel) lo = mid; else hi = mid}
    return lo;
  }

  // Cumulative per-order sums of every OM: cum[(i*(B+1)+j)*3+p] is the
  // integral of order p over bins [0, j).
  function cumulate(rows, bins) {
    const n = rows.length, cum = new Float64Array(n * (bins + 1) * 3);
    for (let i = 0; i < n; i++) {
      const row = rows[i];
      let base = i * (bins + 1) * 3;
      for (let j = 0; j < bins; j++, base += 3) {
        for (let p = 0; p < 3; p++) cum[base + 3 + p] = cum[base + p] + row[j][p];
      }
    }
    return cum;
  }

  // First-light time: where the running positive part of the selected signal
  // reaches `level` of its total, interpolated inside the bin.
  function arrival(row, weights, edges, origin, level = 0.1) {
    let total = 0;
    for (const v of row) total += Math.max(0, weights[0] * v[0] + weights[1] * v[1] + weights[2] * v[2]);
    if (!(total > 0)) return NaN;
    let run = 0;
    for (let j = 0; j < row.length; j++) {
      const v = Math.max(0, weights[0] * row[j][0] + weights[1] * row[j][1] + weights[2] * row[j][2]);
      if (run + v >= level * total) {
        const f = v > 0 ? (level * total - run) / v : 0;
        return origin + edges[j] + f * (edges[j + 1] - edges[j]);
      }
      run += v;
    }
    return origin + edges[row.length];
  }

  // Monotone map from movie progress u in [0, 1] to detector time.  With
  // strength a the clock spends a share a of the playback where light arrives
  // and 1 - a uniformly; a = 0 is a linear clock.
  function pacing(grid, activity, strength) {
    const k = grid.length, density = new Float64Array(k);
    let total = 0;
    for (let i = 0; i < k; i++) total += activity[i];
    for (let i = 0; i < k; i++) density[i] = (1 - strength) / k + (total > 0 ? strength * activity[i] / total : strength / k);
    const cdf = new Float64Array(k + 1);
    for (let i = 0; i < k; i++) cdf[i + 1] = cdf[i] + density[i];
    for (let i = 0; i <= k; i++) cdf[i] /= cdf[k];
    const t0 = grid[0], step = k > 1 ? grid[1] - grid[0] : 1;
    return {
      time(u) {
        u = clamp(u, 0, 1);
        let lo = 0, hi = k;
        while (hi - lo > 1) {const mid = (lo + hi) >> 1; if (cdf[mid] <= u) lo = mid; else hi = mid}
        const span = cdf[lo + 1] - cdf[lo], f = span > 0 ? (u - cdf[lo]) / span : 0;
        return t0 - step / 2 + (lo + f) * step;
      },
      progress(T) {
        const x = clamp((T - (t0 - step / 2)) / step, 0, k), lo = Math.min(k - 1, Math.floor(x));
        return cdf[lo] + (x - lo) * (cdf[lo + 1] - cdf[lo]);
      },
    };
  }

  function gaussianSmooth(values, sigmaCells) {
    if (!(sigmaCells > 0)) return Float64Array.from(values);
    const n = values.length, out = new Float64Array(n), reach = Math.ceil(3 * sigmaCells);
    const kernel = [];
    for (let d = -reach; d <= reach; d++) kernel.push(Math.exp(-0.5 * (d / sigmaCells) ** 2));
    for (let i = 0; i < n; i++) {
      let s = 0, w = 0;
      for (let d = -reach; d <= reach; d++) {
        const j = i + d;
        if (j >= 0 && j < n) {s += kernel[d + reach] * values[j]; w += kernel[d + reach]}
      }
      out[i] = s / w;
    }
    return out;
  }

  // Display source description: a timed axis gives a moving head, an untimed
  // source may get an emission time estimated from direct-light arrivals.
  function sourceInfo(source) {
    if (!source) return null;
    if (source.type === "point") return {kind: "point", position: source.position_m, label: source.label || "Point source"};
    const shape = source.shape || "line", dir = source.direction, half = source.extent_m || 40;
    const timing = source.timing && source.timing.s_m && source.timing.s_m.length >= 2 ? source.timing : null;
    const info = {kind: shape === "spindle" ? "shower" : "track", label: source.label || (shape === "spindle" ? "Shower" : "Track"),
                  centre: source.position_m, direction: dir.slice(), half, timing: null};
    if (timing) {
      const n = timing.s_m.length, reversed = timing.t_ns[n - 1] < timing.t_ns[0];
      const order = [...Array(n).keys()];
      if (reversed) order.reverse();
      info.direction = reversed ? dir.map(x => -x) : dir.slice();
      info.points = order.map(k => ({t: timing.t_ns[k], w: timing.weight ? timing.weight[k] : 1,
        xyz: source.position_m.map((x, j) => x + timing.s_m[k] * dir[j])}));
      info.timing = {t0: info.points[0].t, t1: info.points[n - 1].t};
    }
    const sign = info.direction;
    info.start = source.position_m.map((x, j) => x - half * sign[j]);
    info.end = source.position_m.map((x, j) => x + half * sign[j]);
    return info;
  }

  function headAt(info, T) {
    const p = info.points;
    if (!p || T < p[0].t) return null;
    if (T >= p[p.length - 1].t) return {xyz: p[p.length - 1].xyz, done: true};
    let lo = 0, hi = p.length - 1;
    while (hi - lo > 1) {const mid = (lo + hi) >> 1; if (p[mid].t <= T) lo = mid; else hi = mid}
    const span = p[hi].t - p[lo].t, u = span > 0 ? (T - p[lo].t) / span : 0;
    return {xyz: p[lo].xyz.map((x, j) => x + (p[hi].xyz[j] - x) * u), done: false};
  }

  // Emission time of an untimed compact source from direct-light (order 0)
  // arrivals: the median of t_arrival - distance * n_g / c.  Display only.
  function estimateEmission(positions, rows, edges, origins, centre, groupIndex) {
    const guesses = [];
    for (let i = 0; i < rows.length; i++) {
      const t = arrival(rows[i], [1, 0, 0], edges, origins[i], 0.05);
      if (!Number.isFinite(t)) continue;
      const d = Math.hypot(positions[i][0] - centre[0], positions[i][1] - centre[1], positions[i][2] - centre[2]);
      guesses.push(t - d * groupIndex / C_M_PER_NS);
    }
    if (!guesses.length) return NaN;
    guesses.sort((a, b) => a - b);
    return guesses[guesses.length >> 1];
  }

  // Half-opening angle of the direct-light front behind a beta=1 particle:
  // light leaves at the Cherenkov angle (phase index) and travels at c/n_g.
  function frontAngle(phaseIndex, groupIndex) {
    const c = 1 / phaseIndex, s = Math.sqrt(Math.max(0, 1 - c * c));
    return Math.atan2(s / groupIndex, 1 - c / groupIndex);
  }

  function build(data, eventIndex, {smooth = null, part = "all", groupIndex, phaseIndex = 1.34} = {}) {
    const event = data.events[eventIndex], det = data.detector, edges = data.readout.relative_time_edges_ns;
    const n = det.positions_m.length, bins = edges.length - 1;
    const rows = smooth ? smooth(event.components, edges) : event.components;
    const weights = PARTS[part] || PARTS.all;
    const origins = event.time_origin_ns;
    const ng = groupIndex || data.medium?.group_index || 1.36;
    const cum = cumulate(rows, bins);
    // Only OMs with any stored signal take part in per-frame work.
    const relevant = [];
    for (let i = 0; i < n; i++) {
      let any = false;
      for (let j = 0; j < bins && !any; j++) any = rows[i][j][0] !== 0 || rows[i][j][1] !== 0 || rows[i][j][2] !== 0;
      if (any) relevant.push(i);
    }
    const window = new Float64Array(n), first = new Float64Array(n).fill(NaN);
    for (const i of relevant) {
      const end = (i * (bins + 1) + bins) * 3;
      window[i] = weights[0] * cum[end] + weights[1] * cum[end + 1] + weights[2] * cum[end + 2];
      first[i] = arrival(rows[i], weights, edges, origins[i]);
    }
    const sources = (event.sources || []).map(sourceInfo).filter(Boolean);
    for (const s of sources) {
      if (s.kind === "point" || (s.kind === "shower" && !s.timing)) {
        const centre = s.kind === "point" ? s.position : s.centre;
        const t = estimateEmission(det.positions_m, rows, edges, origins, centre, ng);
        if (Number.isFinite(t)) s.emission = {t, estimated: true, position: centre};
      } else if (s.kind === "shower" && s.timing) {
        s.emission = {t: s.timing.t0, estimated: false, position: s.points[0].xyz};
      }
    }
    // Clock range: from first to last appreciable light, widened to the source.
    let lo = Infinity, hi = -Infinity;
    for (const i of relevant) {
      const row = rows[i];
      let total = 0;
      for (const v of row) total += Math.abs(weights[0] * v[0] + weights[1] * v[1] + weights[2] * v[2]);
      if (!(total > 0)) continue;
      let run = 0;
      for (let j = 0; j < bins; j++) {
        run += Math.abs(weights[0] * row[j][0] + weights[1] * row[j][1] + weights[2] * row[j][2]);
        if (run >= 1e-3 * total) {lo = Math.min(lo, origins[i] + edges[j]); break}
      }
      run = 0;
      for (let j = bins - 1; j >= 0; j--) {
        run += Math.abs(weights[0] * row[j][0] + weights[1] * row[j][1] + weights[2] * row[j][2]);
        if (run >= 2e-3 * total) {hi = Math.max(hi, origins[i] + edges[j + 1]); break}
      }
    }
    if (!Number.isFinite(lo)) {lo = Math.min(...origins) + edges[0]; hi = Math.max(...origins) + edges[bins]}
    for (const s of sources) {
      if (s.timing) {lo = Math.min(lo, s.timing.t0); hi = Math.max(hi, Math.min(s.timing.t1, hi + 0.25 * (hi - lo)))}
      if (s.emission) lo = Math.min(lo, s.emission.t);
    }
    const pad = 0.03 * (hi - lo) + 5;
    const t0 = lo - pad, t1 = hi + pad;
    // Activity on the clock: positive selected signal per grid cell.
    const cells = 720, step = (t1 - t0) / cells, grid = new Float64Array(cells), activity = new Float64Array(cells);
    for (let k = 0; k < cells; k++) grid[k] = t0 + (k + 0.5) * step;
    for (const i of relevant) {
      for (let j = 0; j < bins; j++) {
        const v = weights[0] * rows[i][j][0] + weights[1] * rows[i][j][1] + weights[2] * rows[i][j][2];
        if (!(v > 0)) continue;
        const a = origins[i] + edges[j], b = origins[i] + edges[j + 1];
        const ka = clamp(Math.floor((a - t0) / step), 0, cells - 1), kb = clamp(Math.floor((b - t0) / step), 0, cells - 1);
        for (let k = ka; k <= kb; k++) activity[k] += v / (kb - ka + 1);
      }
    }
    const smoothActivity = gaussianSmooth(activity, 4);
    const finite = [...first].filter(Number.isFinite).sort((a, b) => a - b);
    const quantile = q => finite.length ? finite[clamp(Math.round(q * (finite.length - 1)), 0, finite.length - 1)] : t0;
    let peak = 0;
    for (const i of relevant) peak = Math.max(peak, window[i]);
    return {
      event, n, bins, edges, origins, rows, cum, weights, relevant, window, first, peak, sources,
      t0, t1, grid, activity: smoothActivity, groupIndex: ng, phaseIndex,
      frontAngle: frontAngle(phaseIndex, ng),
      arrivalRange: [quantile(0.02), quantile(0.98)],
      pacing: strength => pacing(grid, smoothActivity, strength),
    };
  }

  // State of every relevant OM at detector time T: cumulative selected charge,
  // per-order cumulative charges and current selected rate (per ns).
  function stateAt(model, T, out) {
    const {bins, edges, origins, cum, weights} = model;
    if (!out || out.n !== model.n) {
      out = {n: model.n, q: new Float64Array(model.n), q3: new Float64Array(model.n * 3), rate: new Float64Array(model.n)};
    }
    for (const i of model.relevant) {
      const rel = T - origins[i], j = binAt(edges, rel), base = i * (bins + 1) * 3;
      let a, b, f = 0, width = 1;
      if (j < 0) {a = base; b = base}
      else if (j >= bins) {a = base + bins * 3; b = a}
      else {a = base + j * 3; b = a + 3; width = edges[j + 1] - edges[j]; f = (rel - edges[j]) / width}
      let q = 0, r = 0;
      for (let p = 0; p < 3; p++) {
        const lo = cum[a + p], hi = cum[b + p], value = lo + f * (hi - lo);
        out.q3[i * 3 + p] = value;
        q += weights[p] * value;
        r += weights[p] * (hi - lo) / width;
      }
      out.q[i] = q;
      out.rate[i] = j >= 0 && j < bins ? r : 0;
    }
    return out;
  }

  // Selected-component rows of one OM for plotting: per-order values per bin.
  function series(model, i) {
    const {rows, edges} = model;
    return rows[i].map((v, j) => ({t0: edges[j], t1: edges[j + 1], v}));
  }

  return {C_M_PER_NS, PARTS, binAt, cumulate, arrival, pacing, gaussianSmooth, sourceInfo, headAt,
          estimateEmission, frontAngle, build, stateAt, series};
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = LightHitStudioModel;
}
