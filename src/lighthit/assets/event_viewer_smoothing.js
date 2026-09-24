"use strict";

const LightHitViewerSmoothing = (() => {
  const SQRT_TWO_PI = Math.sqrt(2 * Math.PI);
  const nodes = [-0.9602898564975363, -0.7966664774136267,
    -0.5255324099163290, -0.1834346424956498, 0.1834346424956498,
    0.5255324099163290, 0.7966664774136267, 0.9602898564975363];
  const quadratureWeights = [0.1012285362903763, 0.2223810344533745,
    0.3137066458778873, 0.3626837833783620, 0.3626837833783620,
    0.3137066458778873, 0.2223810344533745, 0.1012285362903763];

  function operator(edges, sigmaNs = 3) {
    if (!Array.isArray(edges) || edges.length < 2 ||
        !edges.every(Number.isFinite) || !Number.isFinite(sigmaNs) || sigmaNs <= 0) {
      throw Error("Gaussian display requires finite time edges and positive sigma");
    }
    const rows = [];
    for (let i = 0; i < edges.length - 1; i++) {
      if (edges[i + 1] <= edges[i]) throw Error("Time edges must increase");
      const weights = [];
      for (let j = 0; j < edges.length - 1; j++) {
        if (edges[j + 1] <= edges[j]) throw Error("Time edges must increase");
        if (edges[i] - edges[j + 1] > 8 * sigmaNs ||
            edges[j] - edges[i + 1] > 8 * sigmaNs) continue;
        const targetMid = (edges[i] + edges[i + 1]) / 2;
        const sourceMid = (edges[j] + edges[j + 1]) / 2;
        const targetHalf = (edges[i + 1] - edges[i]) / 2;
        const sourceHalf = (edges[j + 1] - edges[j]) / 2;
        let integral = 0;
        for (let a = 0; a < nodes.length; a++) {
          const target = targetMid + targetHalf * nodes[a];
          for (let b = 0; b < nodes.length; b++) {
            const z = (target - sourceMid - sourceHalf * nodes[b]) / sigmaNs;
            integral += quadratureWeights[a] * quadratureWeights[b] * Math.exp(-z * z / 2);
          }
        }
        const weight = targetHalf * integral / (2 * sigmaNs * SQRT_TWO_PI);
        if (weight > 1e-10) weights.push([j, weight]);
      }
      rows.push(weights);
    }
    return rows;
  }

  function smoothComponents(components, edges, sigmaNs = 3) {
    const matrix = operator(edges, sigmaNs);
    const bins = edges.length - 1;
    return components.map(module => {
      if (module.length !== bins) throw Error("Components and time edges differ");
      return matrix.map(weights => {
        const output = [0, 0, 0];
        for (const [j, weight] of weights) {
          for (let order = 0; order < 3; order++) output[order] += weight * module[j][order];
        }
        return output;
      });
    });
  }

  return {operator, smoothComponents};
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = LightHitViewerSmoothing;
}
