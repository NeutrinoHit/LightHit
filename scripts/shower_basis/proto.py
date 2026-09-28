# ruff: noqa: F403, F405  (prototype scripts share the helpers of common/proto)
"""Prototype order >=2 driver with explicit representation knobs.

Mirrors the hybrid >=2 arithmetic (two fields, frozen cone, 9 wavelengths,
detector scale, OM time origins, cubic readout 65->321) but exposes:
  cell_z, cell_t (or axial_only), source degree L, azimuthal M, frequency
subset.  Returns ge2 charge at all OMs (omega=0 pass over 'inside') and ge2
bins for the given active set.  Prepared kernels are cached in-process.
"""
import os
import sys, time
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import *
from lighthit.experimental.axial_source import AxisFrame
from lighthit.experimental.axial_fast import compile_axial_source_fast
from lighthit._directional_numba import PreparedDirectionalKernel
from lighthit.readout import inverse_bins
from scipy.interpolate import CubicSpline

_PREPARED = {}


def prepared(k, lam, degree, indices):
    key = (lam, degree, tuple(indices))
    if key not in _PREPARED:
        cache = k._directional_cache_for(float(k.wavelength.wavelength_nm[lam]))
        _PREPARED[key] = PreparedDirectionalKernel.from_cache(
            cache, degree=degree, frequency_indices=np.asarray(indices))
        k._directional_caches.clear()
    return _PREPARED[key]


def fields(k, source):
    sample = k.medium.sample(k.wavelength.wavelength_nm)
    el = k._source_elements(source)
    valid = el.beta * float(np.min(sample["phase_index"])) > 1
    if not np.all(valid):
        el = el.subset(valid)
    return el, el.field(0), el.field(2), sample


def origins_of(k, el, sample):
    first = int(np.argmin(el.start_ns))
    fastest = float(np.min(sample["group_index"]))
    pos = k.detector.positions_m
    return float(el.start_ns[first]) + np.linalg.norm(pos - el.start_m[first], axis=1) / (0.299792458 / fastest)


def ge2(k, source, active, *, cell_z=0.12, cell_t=None, axial_only=False, degree=32,
        M=4, indices=None, zero_targets=None, timings=None, receiver_block=1,
        slices=None, stencil_m=0.2, time_step_ns=0.02, axial_order=1,
        element_order=2, stencil_threshold_m=0.0):
    """Return (charge_ge2 (N,), bins_ge2 (N, nb), omega_used)."""
    timings = {} if timings is None else timings
    t0 = time.perf_counter()
    omega_all = k.config.omega_per_ns
    indices = np.arange(len(omega_all)) if indices is None else np.asarray(indices)
    omega = omega_all[indices]
    el, f0, f2, sample = fields(k, source)
    frame = AxisFrame.of(f0)
    kw = dict(azimuthal_degree=M, cell_m=cell_z, transverse_m=cell_t, element_order=2,
              frame=frame, axial_only=axial_only)
    if slices is None:
        s0 = compile_axial_source_fast(f0, degree, omega, **kw)
        s2 = compile_axial_source_fast(f2, degree, omega, **kw)
    else:
        from lighthit._shower_basis import compile_axial_slices
        s0, s2 = compile_axial_slices([f0, f2], degree, omega, azimuthal_degree=M, cell_m=cell_z,
                                      frame=frame, transverse=slices, stencil_m=stencil_m,
                                      time_step_ns=time_step_ns, axial_order=axial_order,
                                      element_order=element_order, stencil_threshold_m=stencil_threshold_m)
    timings["compile"] = time.perf_counter() - t0
    timings["cells"] = len(s0.z_m)
    alpha = np.ascontiguousarray(k.acceptance()["alpha"], float)
    pos = k.detector.positions_m
    looks = np.ascontiguousarray(-k.detector.orientations, float)
    area = np.asarray(k.detector.effective_area_m2, float)
    n = len(pos)
    zero_targets = np.arange(n) if zero_targets is None else np.asarray(zero_targets)
    charge = np.zeros(n)
    spectrum = np.zeros((len(omega), n), complex)
    z0 = replace(s0, channels=np.ascontiguousarray(s0.channels[:, :, :1]))
    z2 = replace(s2, channels=np.ascontiguousarray(s2.channels[:, :, :1]))
    tp = ta0 = ta = 0.0
    for lam, (w, wt, ph) in enumerate(zip(k.wavelength.wavelength_nm, k.wavelength.weight_nm,
                                          sample["phase_index"])):
        a = 1.0 / w ** 2
        b = 1.0 / (w ** 2 * ph ** 2)
        scale = wt * area * float(k.detector.spectral_weight(float(w)))
        t = time.perf_counter()
        pz = prepared(k, lam, degree, [indices[0]])
        pf = prepared(k, lam, degree, indices)
        tp += time.perf_counter() - t
        t = time.perf_counter()
        r = pz.apply(z0, pos[zero_targets], looks[zero_targets], alpha,
                     source_omega_per_ns=omega[:1], receiver_block=4, second_source=z2,
                     first_scale=a, second_scale=b, ge2_only=True)
        charge[zero_targets] += scale[zero_targets] * r[0, :, 1].real
        ta0 += time.perf_counter() - t
        t = time.perf_counter()
        r = pf.apply(s0, pos[active], looks[active], alpha, source_omega_per_ns=omega,
                     receiver_block=receiver_block, second_source=s2, first_scale=a,
                     second_scale=b, ge2_only=True)
        spectrum[:, active] += scale[active][None, :] * r[:, :, 1]
        ta += time.perf_counter() - t
    timings.update(prepare=tp, zero_apply=ta0, full_apply=ta)
    t = time.perf_counter()
    origins = origins_of(k, el, sample)
    bins = np.zeros((n, len(k.config.relative_time_edges_ns) - 1))
    spectrum[0] = charge
    rel = spectrum[:, active] * np.exp(-1j * omega[:, None] * origins[None, active])
    out = k.config.readout_omega_per_ns
    rel = CubicSpline(omega, rel.real, axis=0)(out) + 1j * CubicSpline(omega, rel.imag, axis=0)(out)
    bins[active] = inverse_bins(out, rel, k.config.relative_time_edges_ns)
    timings["readout"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t0
    return charge, bins


def errors(ref, charge, bins, active, label=""):
    """ref: dict from run_hybrid npz. Reports ge2 and total metrics."""
    rq = ref["charge"][:, 2]; rb = ref["bins"][:, :, 2]
    idx = np.flatnonzero(active)
    tot_ref = ref["charge"][idx].sum(1)
    order = np.argsort(-tot_ref)

    d = charge - rq
    l1 = np.abs(bins[idx] - rb[idx]).sum(1) / np.abs(rb[idx]).sum(1)
    tb_ref = ref["bins"][idx].sum(2)
    tb_new = ref["bins"][idx, :, :2].sum(2) + bins[idx]
    l1t = np.abs(tb_new - tb_ref).sum(1) / np.abs(tb_ref).sum(1)
    out = {"ge2_rel_max": float(np.max(np.abs(d[idx]) / np.abs(rq[idx]))),
           "ge2_abs_max_all": float(np.max(np.abs(d))),
           "ge2_L1_max": float(l1.max()), "ge2_L1_bright_max": float(l1[order[:10]].max()),
           "total_rel_max": float(np.max(np.abs(d[idx]) / tot_ref)),
           "total_L1_max": float(l1t.max()), "total_L1_bright_max": float(l1t[order[:10]].max())}
    print(label, " ".join(f"{k}={v:.2e}" for k, v in out.items()), flush=True)
    return out


# ------------------------------------------------------------------ basis tools
def basis_sources(k, source, *, omega=None, cell_z=0.25, axial_order=3, transverse="stencil",
                  M=None, stencil_threshold_m=0.01, directions=None):
    """Compile the two basis fields; ``directions`` optionally replaces element directions."""
    from lighthit._shower_basis import compile_axial_slices
    el, f0, f2, sample = fields(k, source)
    if directions is not None:
        f0 = replace(f0, direction=np.ascontiguousarray(directions(f0)))
        f2 = replace(f2, direction=f0.direction)
    frame = AxisFrame.of(fields(k, source)[1])
    omega = k.config.omega_per_ns if omega is None else omega
    M = k.config.azimuthal_degree if M is None else M
    s0, s2 = compile_axial_slices([f0, f2], k.config.source_degree, omega, azimuthal_degree=M,
                                  cell_m=cell_z, axial_order=axial_order, transverse=transverse,
                                  stencil_threshold_m=stencil_threshold_m, frame=frame)
    return s0, s2, el, sample


def apply_ge2(k, s0, s2, targets, sample, store=None):
    """Order >=2 spectrum (frequency, len(targets)) summed over wavelengths."""
    store = k.ge2_kernel_store() if store is None else store
    idx = np.arange(len(k.config.omega_per_ns))
    pos = k.detector.positions_m[targets]
    looks = np.ascontiguousarray(-k.detector.orientations[targets], float)
    area = np.asarray(k.detector.effective_area_m2, float)[targets]
    alpha = np.ascontiguousarray(k.acceptance()["alpha"], float)
    out = np.zeros((s0.channels.shape[2], len(targets)), complex)
    for w, wt, ph in zip(k.wavelength.wavelength_nm, k.wavelength.weight_nm, sample["phase_index"]):
        prepared = store.kernel(k, w, idx)
        r = prepared.apply(s0, pos, looks, alpha, source_omega_per_ns=k.config.omega_per_ns,
                           receiver_block=1, second_source=s2, first_scale=1 / w ** 2,
                           second_scale=1 / (w ** 2 * ph ** 2), ge2_only=True)
        out += wt * area[None, :] * float(k.detector.spectral_weight(float(w))) * r[:, :, 1]
    return out


def readout(k, spectrum, origins):
    """Bins (targets, nb) from a >=2 spectrum on the cache grid, hybrid readout."""
    omega = k.config.omega_per_ns
    rel = spectrum * np.exp(-1j * omega[:, None] * origins[None, :])
    out = k.config.readout_omega_per_ns
    rel = CubicSpline(omega, rel.real, axis=0)(out) + 1j * CubicSpline(omega, rel.imag, axis=0)(out)
    return inverse_bins(out, rel, k.config.relative_time_edges_ns)


def metrics(q, b, q_ref, b_ref, total_ref=None):
    """Per-OM relative/absolute charge and L1 errors of order >=2."""
    rel = np.abs(q - q_ref) / np.abs(q_ref)
    l1 = np.abs(b - b_ref).sum(1) / np.abs(b_ref).sum(1)
    out = {"ge2_rel_max": float(rel.max()), "ge2_abs_max": float(np.abs(q - q_ref).max()),
           "ge2_L1_max": float(l1.max()), "ge2_L1_median": float(np.median(l1))}
    if total_ref is not None:
        out["total_rel_max"] = float((np.abs(q - q_ref) / total_ref).max())
    return out
