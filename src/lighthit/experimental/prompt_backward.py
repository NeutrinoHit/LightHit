"""Backward (adjoint) next-event Monte Carlo for exactly one scattering.

Research prototype, compared against :func:`lighthit.prompt.transport_prompt`.
Nothing in the production path uses it.

Geometry is centred on the module at ``R`` with look direction ``n``.  The
scatter point is ``y = R - b s2``: ``s2`` is the direction the once-scattered
photon travels to the module and ``b`` the distance it covers.  With
``dV = b^2 db dOmega`` the ``1/b^2`` of the second leg cancels, and

    N1 = A_eff int dOmega(s2) A(s2.n) int db e^{-mu_t b} mu_s
         sum_i B_i(y) p_HG(s1_i . s2).

``b`` is sampled from ``mu* exp(-mu* b)`` with ``mu* = min_lambda mu_t``, so
every wavelength reuses the sample with the bounded weight
``mu_s e^{-(mu_t - mu*) b} / mu*``.  ``s2`` is sampled from the defensive
mixture

    q(s2) = beta A(s2.n) / int A + (1 - beta) sum_j w_j p_HG(d_j . s2),

where ``d_j`` are the directions in which direct Cherenkov light reaches the
module (weights ``w_j`` from the direct field at ``R``): forward scattering
close to a direct ray is where ``p_HG`` peaks, so plain acceptance sampling
(``beta = 1``) has a large variance for ``g = 0.9``.  Every sample carries the
exact ratio ``A / q``, so the estimator is unbiased for any lobes.

The direct field of a straight segment (start ``x0``, direction ``u``,
length ``h``, cone ``cos theta_C``) at ``y`` is analytic: with
``z = (y - x0).u``, ``rho = |(y - x0) - z u|``,

    xi* = z - rho cot(theta_C),   a* = rho / sin(theta_C),
    B   = Y e^{-mu_t a*} / (2 pi rho sin theta_C)     if 0 <= xi* < h,

``s1 = (y - x(xi*)) / a*`` and the photon arrives at
``t = t0 + xi* dt/dxi + (a* + b) / v_g``.  ``Y`` is the per-metre spectral
yield of the same two Cherenkov fields as the prompt path.

For a shower the segments sit in a bounding-volume tree whose nodes bound
both position (a ball) and direction (a cone); a node is skipped for ``y``
when no direction within its angular radius can make the Cherenkov angle
with any direction from its ball to ``y``.  The test is conservative, so the estimator
stays unbiased.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import njit, prange

from .._prompt_numba import _find_bin, _hg
from ..prompt import (_element_segments, _spectral_tables, _track_segments,
                      acceptance_monomials, acceptance_polynomial)
from ..sources import CherenkovTrack

__all__ = ["BackwardTree", "build_tree", "direct_lobes", "backward_order1"]


# --------------------------------------------------------------- grouping

@dataclass(frozen=True)
class BackwardTree:
    """Bounding-volume tree over segments: position ball x direction cone.

    Node ``k`` covers ``order[node_range[k, 0]:node_range[k, 1]]``; a leaf has
    ``node_child[k] == (-1, -1)``.  ``node_bound[k]`` is (centre xyz, radius,
    axis xyz, angular spread) and ``node_cone[k]`` the Cherenkov-angle range.
    """
    order: np.ndarray
    node_bound: np.ndarray
    node_cone: np.ndarray
    node_child: np.ndarray
    node_range: np.ndarray
    lobe_nodes: np.ndarray      # a frontier of mid-size nodes, for the lobes


def _bounds(start, end, direction, theta, members):
    """Ball about the bounding-box centre and a cone of directions."""
    ends = np.concatenate([start[members], end[members]])
    centre = 0.5 * (ends.min(axis=0) + ends.max(axis=0))
    radius = float(np.sqrt(((ends - centre) ** 2).sum(axis=1).max()))
    axis = direction[members].sum(axis=0)
    norm = np.linalg.norm(axis)
    axis = axis / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
    spread = float(np.arccos(np.clip(direction[members] @ axis, -1, 1)).max())
    return (np.concatenate([centre, [radius], axis, [spread]]),
            np.array([theta[members].min(), theta[members].max()]))


def build_tree(segments, leaf_size=8, distance_scale_m=30.0, lobe_node_size=512):
    """Recursive median splits in position or direction, whichever is wider.

    A node's angular spread counts ``distance_scale_m`` times its angle, so
    that at typical module distances the split reduces the larger of the two
    errors of the conservative cone test.
    """
    start = np.asarray(segments.start, float)
    direction = np.asarray(segments.direction, float)
    end = start + np.asarray(segments.length, float)[:, None] * direction
    theta = np.arccos(np.clip(segments.cone, -1, 1))
    middle = 0.5 * (start + end)
    order = np.arange(len(start))
    bounds, cones, children, ranges = [], [], [], []
    stack = [(0, len(order), -1, 0)]
    while stack:
        low, high, parent, side = stack.pop()
        node = len(bounds)
        members = order[low:high]
        b, c = _bounds(start, end, direction, theta, members)
        bounds.append(b)
        cones.append(c)
        children.append([-1, -1])
        ranges.append([low, high])
        if parent >= 0:
            children[parent][side] = node
        if high - low <= leaf_size:
            continue
        box = middle[members].max(axis=0) - middle[members].min(axis=0)
        if b[7] * distance_scale_m > box.max():
            spread = direction[members] - direction[members].mean(axis=0)
            _, _, vt = np.linalg.svd(spread, full_matrices=False)
            key = direction[members] @ vt[0]
        else:
            key = middle[members, int(np.argmax(box))]
        half = (high - low) // 2
        split = np.argpartition(key, half)
        order[low:high] = members[split]
        stack.append((low + half, high, node, 1))
        stack.append((low, low + half, node, 0))
    ranges = np.asarray(ranges, np.int64)
    children = np.asarray(children, np.int64)
    count = ranges[:, 1] - ranges[:, 0]
    parent_count = np.full(len(count), np.iinfo(np.int64).max)
    for k in range(len(children)):
        for child in children[k]:
            if child >= 0:
                parent_count[child] = count[k]
    lobe_nodes = np.flatnonzero((count <= lobe_node_size) & (parent_count > lobe_node_size))
    return BackwardTree(order.astype(np.int64), np.asarray(bounds), np.asarray(cones),
                        children, ranges, lobe_nodes.astype(np.int64))


# ------------------------------------------------------------- the kernel

@njit(cache=True, nogil=True)
def _next(state):
    """splitmix64 step; returns a uniform in (0, 1)."""
    state[0] += np.uint64(0x9E3779B97F4A7C15)
    z = state[0]
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (float(z >> np.uint64(11)) + 0.5) * (1.0 / 9007199254740992.0)


@njit(cache=True, nogil=True)
def _sample_cosine(apoly, cdf_total, u):
    """x in [-1, 1] with density A(x) / int A, by Newton-bisection on the CDF."""
    target = u * cdf_total
    low, high = -1.0, 1.0
    x = 0.0
    for _ in range(60):
        # CDF(x) = int_{-1}^x A
        value = 0.0
        power_x = x
        power_m = -1.0
        for k in range(apoly.shape[0]):
            value += apoly[k] * (power_x - power_m) / (k + 1)
            power_x *= x
            power_m *= -1.0
        density = 0.0
        for k in range(apoly.shape[0] - 1, -1, -1):
            density = density * x + apoly[k]
        if value > target:
            high = x
        else:
            low = x
        step = (value - target) / density if density > 0 else 0.0
        candidate = x - step
        if not (low < candidate < high) or density <= 0:
            candidate = 0.5 * (low + high)
        if abs(candidate - x) < 1e-15:
            return candidate
        x = candidate
    return x


@njit(cache=True, nogil=True)
def _passes(bound, cone, y0, y1, y2):
    """False only if no segment in the bound can light ``y`` on its cone."""
    d0 = y0 - bound[0]
    d1 = y1 - bound[1]
    d2 = y2 - bound[2]
    dist = math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    radius = bound[3]
    if dist <= radius * 1.000001 + 1e-9:
        return True
    cosine = (d0 * bound[4] + d1 * bound[5] + d2 * bound[6]) / dist
    alpha = math.acos(min(1.0, max(-1.0, cosine)))
    slack = bound[7] + math.asin(radius / dist) + 1e-9
    return cone[0] - slack <= alpha <= cone[1] + slack


@njit(cache=True, nogil=True)
def _task(R, look, e1, e2, area, origin, start, direction, length, q0, q2, cone,
          t0, dtda, order, node_bound, node_cone, node_child, node_range,
          g, mus, mut, speed, s0w, s2w, mu_star,
          apoly, cdf_total, lobes, lobe_cdf, beta, gamma, near_m, edges, samples, seed,
          bins, stats, trace):
    """``samples`` backward histories for one module; accumulates in place.

    ``stats``: [sum w, sum w^2, sum w in window, sum w^2 in window,
    node tests, segment tests, hits].  ``trace`` (``(samples, 6)`` or empty)
    records ``(weight, b, mixture component, smallest rho hit, node tests,
    segment tests)`` per sample.
    """
    L = mus.shape[0]
    nb = edges.shape[0] - 1
    state = np.empty(1, np.uint64)
    state[0] = np.uint64(seed)
    lam_weight = np.empty(L)
    stack = np.empty(256, np.int64)
    J = lobes.shape[0]
    log_span = np.empty(J)
    for j in range(J):
        log_span[j] = math.log(max(lobes[j, 3], 2.0 * near_m) / near_m)
    for sample in range(samples):
        kind = 0
        if J > 0 and _next(state) >= beta:
            kind = 1 if _next(state) >= gamma else 2
        if kind == 0:
            x = _sample_cosine(apoly, cdf_total, _next(state))
            phi = 2.0 * math.pi * _next(state)
            sx = math.sqrt(max(0.0, 1.0 - x * x))
            c = math.cos(phi) * sx
            s = math.sin(phi) * sx
            s20 = x * look[0] + c * e1[0] + s * e2[0]
            s21 = x * look[1] + c * e1[1] + s * e2[1]
            s22 = x * look[2] + c * e1[2] + s * e2[2]
        else:
            pick = _next(state) * lobe_cdf[J - 1]
            j = 0
            while j < J - 1 and lobe_cdf[j] < pick:
                j += 1
            w = (1.0 - g * g) / (1.0 - g + 2.0 * g * _next(state))
            mu = (1.0 + g * g - w * w) / (2.0 * g)
            mu = min(1.0, max(-1.0, mu))
            phi = 2.0 * math.pi * _next(state)
            d0, d1, d2 = lobes[j, 0], lobes[j, 1], lobes[j, 2]
            if abs(d2) < 0.9:
                f0, f1, f2 = d1, -d0, 0.0
            else:
                f0, f1, f2 = 0.0, d2, -d1
            fn = math.sqrt(f0 * f0 + f1 * f1 + f2 * f2)
            f0 /= fn
            f1 /= fn
            f2 /= fn
            h0 = d1 * f2 - d2 * f1
            h1 = d2 * f0 - d0 * f2
            h2 = d0 * f1 - d1 * f0
            st = math.sqrt(max(0.0, 1.0 - mu * mu))
            c = math.cos(phi) * st
            s = math.sin(phi) * st
            s20 = mu * d0 + c * f0 + s * h0
            s21 = mu * d1 + c * f1 + s * h1
            s22 = mu * d2 + c * f2 + s * h2
            x = s20 * look[0] + s21 * look[1] + s22 * look[2]
        if kind == 2:
            # distance to the lobe's source, log-uniform in [near_m, L_j]
            span = max(lobes[j, 3], 2.0 * near_m)
            b = span - near_m * math.exp(log_span[j] * _next(state))
        else:
            b = -math.log(_next(state)) / mu_star
        accept = 0.0
        for k in range(apoly.shape[0] - 1, -1, -1):
            accept = accept * x + apoly[k]
        p_exp = mu_star * math.exp(-mu_star * b)
        density = beta * accept / (2.0 * math.pi * cdf_total) * p_exp
        if J > 0:
            mix = 0.0
            for j in range(J):
                prev = lobe_cdf[j - 1] if j > 0 else 0.0
                span = max(lobes[j, 3], 2.0 * near_m)
                p_near = (1.0 / ((span - b) * log_span[j])
                          if b <= span - near_m else 0.0)
                mix += (lobe_cdf[j] - prev) * _hg(
                    lobes[j, 0] * s20 + lobes[j, 1] * s21 + lobes[j, 2] * s22, g) * (
                    (1.0 - gamma) * p_exp + gamma * p_near)
            density += (1.0 - beta) * mix / lobe_cdf[J - 1]
        y0 = R[0] - b * s20
        y1 = R[1] - b * s21
        y2 = R[2] - b * s22
        for lam in range(L):
            lam_weight[lam] = (area * accept / density * mus[lam]
                               * math.exp(-mut[lam] * b))
        total = 0.0
        window = 0.0
        rho_min = 1e300
        tests0 = stats[4]
        tests1 = stats[5]
        stack[0] = 0
        top = 1
        while top > 0:
            top -= 1
            node = stack[top]
            stats[4] += 1
            if not _passes(node_bound[node], node_cone[node], y0, y1, y2):
                continue
            if node_child[node, 0] >= 0:
                stack[top] = node_child[node, 0]
                stack[top + 1] = node_child[node, 1]
                top += 2
                continue
            for k in range(node_range[node, 0], node_range[node, 1]):
                    i = order[k]
                    stats[5] += 1
                    v0 = y0 - start[i, 0]
                    v1 = y1 - start[i, 1]
                    v2 = y2 - start[i, 2]
                    z = v0 * direction[i, 0] + v1 * direction[i, 1] + v2 * direction[i, 2]
                    rho2 = v0 * v0 + v1 * v1 + v2 * v2 - z * z
                    if rho2 <= 1e-24:
                        continue
                    rho = math.sqrt(rho2)
                    sin_c = math.sqrt(1.0 - cone[i] * cone[i])
                    xi = z - rho * cone[i] / sin_c
                    if xi < 0.0 or xi >= length[i]:
                        continue
                    stats[6] += 1
                    rho_min = min(rho_min, rho)
                    a = rho / sin_c
                    s10 = (v0 - xi * direction[i, 0]) / a
                    s11 = (v1 - xi * direction[i, 1]) / a
                    s12 = (v2 - xi * direction[i, 2]) / a
                    geometry = (_hg(s10 * s20 + s11 * s21 + s12 * s22, g)
                                / (2.0 * math.pi * rho * sin_c))
                    emit = t0[i] + xi * dtda[i] - origin
                    for lam in range(L):
                        value = (lam_weight[lam] * geometry
                                 * (q0[i] * s0w[lam] + q2[i] * s2w[lam])
                                 * math.exp(-mut[lam] * a))
                        total += value
                        kb = _find_bin(edges, emit + (a + b) / speed[lam])
                        if 0 <= kb < nb:
                            bins[kb] += value
                            window += value
        if trace.shape[0] > 0:
            trace[sample, 0] = total
            trace[sample, 1] = b
            trace[sample, 2] = kind
            trace[sample, 3] = rho_min
            trace[sample, 4] = stats[4] - tests0
            trace[sample, 5] = stats[5] - tests1
        stats[0] += total
        stats[1] += total * total
        stats[2] += window
        stats[3] += window * window


@njit(cache=True, parallel=True)
def _run(receivers, looks, frames, areas, origins, start, direction, length, q0, q2,
         cone, t0, dtda, order, node_bound, node_cone, node_child, node_range, g, mus, mut, speed, s0w, s2w, mu_star, apoly,
         cdf_total, lobes, lobe_cdf, lobe_count, beta, gamma, near_m, edges, samples,
         batches, seed):
    n = receivers.shape[0]
    nb = edges.shape[0] - 1
    bins = np.zeros((n, batches, nb))
    stats = np.zeros((n, batches, 7))
    for task in prange(n * batches):
        m = task // batches
        k = task % batches
        _task(receivers[m], looks[m], frames[m, 0], frames[m, 1], areas[m], origins[m],
              start, direction, length, q0, q2, cone, t0, dtda, order, node_bound,
              node_cone, node_child, node_range, g, mus, mut, speed, s0w, s2w, mu_star, apoly, cdf_total,
              lobes[m, :lobe_count[m]], lobe_cdf[m, :lobe_count[m]], beta, gamma,
              near_m, edges,
              samples, seed * 1000003 + task * 7919 + 1, bins[m, k], stats[m, k],
              np.zeros((0, 6)))
    return bins, stats


# ------------------------------------------------------------ the driver

def _frame(look):
    helper = np.array([0.0, 0.0, 1.0]) if abs(look[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(look, helper)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(look, e1)])


def direct_lobes(segments, tree, receiver, look, apoly, mu, max_lobes=256,
                 exact_below=256):
    """Directions (unit, plus source distance) along which direct light arrives.

    Few segments: the exact cone points whose light reaches ``receiver``,
    weighted by their direct field.  Many (short) segments: one lobe per
    tree node of the lobe frontier, from its centre, weighted by its yield,
    attenuation and a Gaussian in the mismatch between the node's view angle
    and the Cherenkov angle (width: its angular spread plus apparent size).  Only the
    ``max_lobes`` largest are kept; any choice keeps the estimator unbiased.
    """
    if len(segments.length) <= exact_below:
        v = receiver[None, :] - segments.start
        z = (v * segments.direction).sum(axis=1)
        rho = np.sqrt(np.maximum((v * v).sum(axis=1) - z * z, 1e-300))
        sine = np.sqrt(1 - segments.cone ** 2)
        xi = z - rho * segments.cone / sine
        hit = (xi >= 0) & (xi < segments.length) & (rho > 1e-9)
        if not np.any(hit):
            # no direct light: aim at the closest point of each segment
            xi = np.clip(z, 0, segments.length)
            hit = np.ones(len(xi), bool)
        point = segments.start[hit] + xi[hit, None] * segments.direction[hit]
        weight_geometry = np.abs(segments.q0[hit]) / np.maximum(rho[hit], 1e-3)
    else:
        bound = tree.node_bound[tree.lobe_nodes]
        point = bound[:, :3]
        towards = receiver[None, :] - point
        distance = np.linalg.norm(towards, axis=1)
        cosine = (towards * bound[:, 4:7]).sum(axis=1) / distance
        view = np.arccos(np.clip(cosine, -1, 1))
        theta = tree.node_cone[tree.lobe_nodes].mean(axis=1)
        width = np.hypot(bound[:, 7] + 0.02, bound[:, 3] / distance)
        cumulative = np.concatenate([[0.0], np.cumsum(
            np.abs(segments.q0 * segments.length)[tree.order])])
        span = tree.node_range[tree.lobe_nodes]
        member_yield = cumulative[span[:, 1]] - cumulative[span[:, 0]]
        weight_geometry = (member_yield * np.exp(-0.5 * ((view - theta) / width) ** 2)
                           / (width * distance))
    towards = receiver[None, :] - point
    distance = np.linalg.norm(towards, axis=1)
    towards /= distance[:, None]
    accept = np.polynomial.polynomial.polyval(towards @ look, apoly)
    weight = weight_geometry * np.exp(-mu * distance) * np.maximum(accept, 1e-3)
    keep = np.argsort(-weight)[:max_lobes]
    keep = keep[weight[keep] > 1e-6 * weight[keep[0]]]
    return np.concatenate([towards[keep], distance[keep, None]], axis=1), weight[keep]


def backward_order1(kernel, source, modules, origins, *, samples=100_000, batches=16,
                    seed=1, tree=None, tree_options=None, beta=0.3, gamma=0.5,
                    near_m=0.05, max_lobes=256):
    """Order 1 at ``modules`` by the backward next-event estimator.

    ``origins`` are the per-module time origins (ns) the bins are relative to,
    e.g. ``transport_prompt(...).time_origin_ns``.  ``samples`` per module are
    split into ``batches`` independent streams for the bin errors.  Returns a
    dict with ``charge``, ``charge_se``, ``window``, ``bins`` and ``bins_se``
    (per module), the counters, and the timings.
    """
    started = perf_counter()
    tables = _spectral_tables(kernel)
    alpha = np.asarray(kernel.acceptance()["alpha"], float)
    apoly = acceptance_polynomial(*acceptance_monomials(alpha))
    x = np.linspace(-1, 1, 20001)
    if np.polynomial.polynomial.polyval(x, apoly).min() < 0:
        raise ValueError("prototype samples s2 from A; A must be non-negative")
    integral = np.polynomial.polynomial.polyint(apoly)
    cdf_total = float(np.polynomial.polynomial.polyval(1.0, integral)
                      - np.polynomial.polynomial.polyval(-1.0, integral))
    if isinstance(source, CherenkovTrack):
        segments = _track_segments(source)
    else:
        elements = source.elements if hasattr(source, "elements") else source
        valid = elements.beta * float(np.min(tables["phase_index"])) > 1
        segments = _element_segments(elements.subset(valid) if not np.all(valid)
                                     else elements)
    stage = perf_counter()
    if tree is None:
        tree = build_tree(segments, **(tree_options or {}))
    tree_seconds = perf_counter() - stage
    modules = np.asarray(modules, np.int64)
    detector = kernel.detector
    receivers = np.ascontiguousarray(detector.positions_m[modules], float)
    looks = np.ascontiguousarray(-detector.orientations[modules], float)
    looks /= np.linalg.norm(looks, axis=1)[:, None]
    frames = np.ascontiguousarray(np.stack([_frame(v) for v in looks]))
    areas = np.asarray(detector.effective_area_m2, float)[modules]
    edges = np.asarray(kernel.config.relative_time_edges_ns, float)
    lobe_count = np.zeros(len(modules), np.int64)
    lobes = np.zeros((len(modules), max_lobes, 4))
    lobe_cdf = np.zeros((len(modules), max_lobes))
    if beta < 1:
        for j in range(len(modules)):
            d, w = direct_lobes(segments, tree, receivers[j], looks[j], apoly,
                                float(tables["mut"].min()), max_lobes)
            lobe_count[j] = len(w)
            lobes[j, :len(w)] = d
            lobe_cdf[j, :len(w)] = np.cumsum(w)
    per_batch = int(math.ceil(samples / batches))
    stage = perf_counter()
    bins, stats = _run(
        receivers, looks, frames, np.ascontiguousarray(areas), np.ascontiguousarray(
            np.asarray(origins, float)), np.ascontiguousarray(segments.start),
        np.ascontiguousarray(segments.direction), np.asarray(segments.length, float),
        np.asarray(segments.q0, float), np.asarray(segments.q2, float),
        np.asarray(segments.cone, float), np.asarray(segments.t0, float),
        np.asarray(segments.dtda, float), tree.order, tree.node_bound,
        tree.node_cone, tree.node_child, tree.node_range, float(kernel.medium.g),
        tables["mus"], tables["mut"], tables["speed"], tables["s0w"], tables["s2w"],
        float(tables["mut"].min()), apoly, cdf_total, lobes, lobe_cdf, lobe_count,
        float(beta), float(gamma), float(near_m), edges, per_batch, batches, int(seed))
    run_seconds = perf_counter() - stage
    count = per_batch * batches
    total = stats[:, :, 0].sum(axis=1)
    square = stats[:, :, 1].sum(axis=1)
    charge = total / count
    charge_se = np.sqrt(np.maximum(square / count - charge ** 2, 0) / (count - 1))
    window = stats[:, :, 2].sum(axis=1) / count
    batch_bins = bins / per_batch
    return {"charge": charge, "charge_se": charge_se, "window": window,
            "bins": batch_bins.mean(axis=1),
            "bins_se": batch_bins.std(axis=1, ddof=1) / np.sqrt(batches),
            "samples_per_module": count,
            "node_tests_per_sample": stats[:, :, 4].sum(axis=1) / count,
            "segment_tests_per_sample": stats[:, :, 5].sum(axis=1) / count,
            "hits_per_sample": stats[:, :, 6].sum(axis=1) / count,
            "nodes": len(tree.node_range), "lobe_nodes": len(tree.lobe_nodes),
            "timings": {"tree_seconds": tree_seconds, "run_seconds": run_seconds,
                        "total_seconds": perf_counter() - started}}
