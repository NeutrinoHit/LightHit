"""Shower-basis route for scattering order >=2 in the hybrid transport.

``TransportKernel.transport_hybrid(source, ge2_basis=ShowerBasis())`` keeps
the direct prompt orders 0+1 unchanged and replaces only the order >=2
source representation: the shower's cloud-in-cell lattice of thousands of 3D
cells is replaced by an axial-slice basis of ~10^2 points (cubic Lagrange
slices along the shower axis, a five-point transverse moment stencil, exact
real harmonics ``l <= L``, ``|m| <= M`` in the axial frame); see
:mod:`lighthit._shower_basis` for the formula and
``docs/research/shower-basis.md`` for the measured errors.

The RTE Green function is the existing directional cache.  Its radial
spline for order >=2 is prepared once per wavelength and frequency set and
kept in a :class:`Ge2KernelStore` (in memory, optionally persisted as a
memory-mapped pack), which removes the per-event spline construction and
cache reload of the default hybrid.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .sources import CherenkovTrack, G4Shower, SpectralLightElements, SyntheticShower

__all__ = ["ShowerBasis", "Ge2KernelStore"]

PACK_SCHEMA = "lighthit/ge2-kernel-pack/1"


@dataclass(frozen=True)
class ShowerBasis:
    """Order >=2 source basis: axial slices x harmonics x transverse moments.

    ``cell_m``: slice spacing along the shower axis.  ``axial_order``: 3
    (cubic Lagrange, fourth order) or 1 (linear).  ``transverse``:
    ``"stencil"`` (zeroth, first and diagonal second transverse moments
    exact) or ``"collapse"`` (every point on the axis).  Slices whose rms
    transverse extent is below ``stencil_threshold_m`` keep only the axis
    point.  ``time_step_ns``: lattice for ``exp(i omega t)`` (error at most
    ``(omega dt)^2 / 8``).  The angular degree and azimuthal band are those
    of the kernel configuration (``source_degree``, ``azimuthal_degree``).
    """
    cell_m: float = 0.25
    axial_order: int = 3
    transverse: str = "stencil"
    stencil_m: float = 0.2
    stencil_threshold_m: float = 0.03
    time_step_ns: float = 0.02

    def __post_init__(self):
        if not (self.cell_m > 0 and self.stencil_m > 0 and self.time_step_ns > 0
                and self.stencil_threshold_m >= 0):
            raise ValueError("ShowerBasis lengths and time step must be positive")
        if self.axial_order not in (1, 3):
            raise ValueError("axial_order must be 1 or 3")
        if self.transverse not in ("stencil", "collapse"):
            raise ValueError("transverse must be 'stencil' or 'collapse'")


class Ge2KernelStore:
    """Prepared order >=2 radial splines for all wavelengths of a kernel.

    ``PreparedDirectionalKernel.from_cache`` builds a cubic spline over the
    whole table, for both scattered orders, on every call.  The store builds
    it once per (wavelength set, degree, frequency set), keeps only the order
    >=2 coefficients stacked over wavelengths -- a *bundle* of shape
    (wavelength, radial interval, degree, pair, cubic power, frequency) --
    and reuses it for every event.  ``save`` writes the bundles as ``.npy``
    files; ``load`` memory-maps them.

    The coefficients are stored as ``complex64`` by default (half the memory;
    the apply still evaluates in double precision).  On the BGVD 100 GeV event
    this changed the order >=2 charge by 5e-9 and the bins by 6e-10 relative
    to ``complex128``.
    """

    def __init__(self, dtype=np.complex64):
        self.dtype = np.dtype(dtype)
        self._bundles = {}

    def __len__(self):
        return len(self._bundles)

    @property
    def nbytes(self):
        return int(sum(bundle["coefficients"].nbytes for bundle in self._bundles.values()))

    @staticmethod
    def _key(kernel, indices):
        return (tuple(kernel._directional_cache_key(float(w))
                      for w in kernel.wavelength.wavelength_nm),
                int(kernel.config.source_degree), tuple(int(i) for i in indices))

    def bundle(self, kernel, indices):
        """The stacked order >=2 spline of every kernel wavelength (built once)."""
        from ._directional_numba import PreparedDirectionalKernel
        key = self._key(kernel, indices)
        if key in self._bundles:
            return self._bundles[key]
        coefficients, absorption, speed = None, [], []
        for lam, wavelength in enumerate(kernel.wavelength.wavelength_nm):
            cache = kernel._directional_cache_for(float(wavelength))
            full = PreparedDirectionalKernel.from_cache(
                cache, degree=int(kernel.config.source_degree),
                frequency_indices=np.asarray(indices))
            kernel._directional_caches.pop(
                kernel._directional_cache_key(float(wavelength)), None)
            if coefficients is None:
                coefficients = np.empty((len(kernel.wavelength.wavelength_nm),)
                                        + full.coefficients[:, :, :, :, 1, :].shape,
                                        self.dtype)
                shared = dict(degree=full.degree, acceptance_degree=full.acceptance_degree,
                              pair_lambda=full.pair_lambda, pair_mu=full.pair_mu,
                              omega=full.omega_per_ns, log_r=full.log_r,
                              phase=full.radial_phase)
            elif (not np.array_equal(full.log_r, shared["log_r"])
                  or full.radial_phase != shared["phase"]):
                raise ValueError("wavelength caches differ in their radial grid")
            coefficients[lam] = full.coefficients[:, :, :, :, 1, :]
            absorption.append(full.absorption_per_m)
            speed.append(full.speed_m_per_ns)
            full = cache = None
        bundle = dict(shared, coefficients=coefficients,
                      absorption=np.asarray(absorption, float), speed=np.asarray(speed, float))
        self._bundles[key] = bundle
        return bundle

    def kernel(self, kernel, wavelength_nm, indices, *, frequencies=None):
        """A per-wavelength ``PreparedDirectionalKernel`` for order >=2 only.

        A zero-copy view of the bundle (its order axis has zero stride, and the
        fused apply reads only order >=2).  ``frequencies`` optionally selects
        positions within ``indices``.
        """
        from ._directional_numba import PreparedDirectionalKernel
        bundle = self.bundle(kernel, indices)
        lam = int(np.flatnonzero(np.isclose(kernel.wavelength.wavelength_nm,
                                            float(wavelength_nm), rtol=0, atol=1e-9))[0])
        coefficients = bundle["coefficients"][lam]
        omega = bundle["omega"]
        if frequencies is not None:
            coefficients = coefficients[..., frequencies]
            omega = omega[frequencies]
        shape, strides = coefficients.shape, coefficients.strides
        view = np.lib.stride_tricks.as_strided(
            coefficients, shape=shape[:4] + (2,) + shape[4:],
            strides=strides[:4] + (0,) + strides[4:], writeable=False)
        return PreparedDirectionalKernel(
            int(bundle["degree"]), int(bundle["acceptance_degree"]), bundle["pair_lambda"],
            bundle["pair_mu"], np.asarray(omega), bundle["log_r"], view,
            float(bundle["absorption"][lam]), float(bundle["speed"][lam]),
            str(bundle["phase"]))

    def save(self, directory):
        """Write every bundle as ``<n>.npy`` plus ``pack.json``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        index = []
        for number, (key, bundle) in enumerate(self._bundles.items()):
            np.save(directory / f"{number}.npy", bundle["coefficients"])
            meta = {name: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for name, value in bundle.items() if name != "coefficients"}
            index.append({"file": f"{number}.npy",
                          "key": [list(key[0]), key[1], list(key[2])], "meta": meta})
        (directory / "pack.json").write_text(json.dumps(
            {"schema": PACK_SCHEMA, "dtype": self.dtype.name, "bundles": index}))

    @classmethod
    def load(cls, directory, *, mmap=True):
        directory = Path(directory)
        header = json.loads((directory / "pack.json").read_text())
        if header.get("schema") != PACK_SCHEMA:
            raise ValueError("unsupported order >=2 kernel pack")
        store = cls(header["dtype"])
        for item in header["bundles"]:
            meta = dict(item["meta"])
            for name in ("pair_lambda", "pair_mu"):
                meta[name] = np.ascontiguousarray(meta[name], np.int64)
            for name in ("omega", "log_r", "absorption", "speed"):
                meta[name] = np.asarray(meta[name], float)
            key = (tuple(item["key"][0]), int(item["key"][1]), tuple(item["key"][2]))
            store._bundles[key] = dict(
                meta, coefficients=np.load(directory / item["file"],
                                           mmap_mode="r" if mmap else None))
        return store


def _frequency_indices(kernel, frequency_nodes):
    grid = kernel.config.omega_per_ns
    if not isinstance(frequency_nodes, int) or isinstance(frequency_nodes, bool):
        raise TypeError("frequency_nodes must be an integer")
    if not 4 <= frequency_nodes <= len(grid):
        raise ValueError("frequency_nodes must lie between 4 and the cache grid size")
    indices = np.unique(np.rint(np.linspace(0, len(grid) - 1, frequency_nodes)).astype(int))
    if len(indices) != frequency_nodes:
        raise ValueError("frequency_nodes exceed distinct cache frequencies")
    return indices


def transport_hybrid_basis(kernel, source, basis, *, frequency_nodes=65,
                           prompt_config=None, store=None, representation="basis"):
    """Prompt orders 0+1 and order >=2 on the axial-slice shower basis.

    Same inputs, active-OM rule, time origins and readout as
    :meth:`TransportKernel.transport_hybrid`; only the order >=2 source is
    represented on ``basis``.  ``representation="cells"`` instead compiles
    the reference cloud-in-cell lattice of the default hybrid
    (``KernelConfig.cell_m``) and runs it through the same driver: a
    validation reference for sources the default hybrid rejects (long muons).
    """
    if representation not in ("basis", "cells"):
        raise ValueError("representation must be 'basis' or 'cells'")
    from ._directional_numba import PreparedDirectionalKernel  # noqa: F401 (numba check)
    from ._shower_basis import apply_multi, compile_axial_slices
    from .experimental.axial_source import AxisFrame
    from .readout import inverse_bins
    from .transport import HybridTransportResponse

    if not isinstance(source, (G4Shower, SyntheticShower, SpectralLightElements)):
        raise TypeError("the shower basis requires a shower source")
    if isinstance(source, CherenkovTrack):  # pragma: no cover - excluded above
        raise TypeError("use the track route for a straight track")
    if not isinstance(basis, ShowerBasis):
        raise TypeError("ge2_basis must be a ShowerBasis")
    began = perf_counter()
    timings = {}
    store = kernel._ge2_store if store is None else store
    indices = _frequency_indices(kernel, frequency_nodes)
    omega = kernel.config.omega_per_ns[indices]
    output_omega = (kernel.config.readout_omega_per_ns
                    if kernel.config.readout_omega_per_ns is not None else omega)

    stage = perf_counter()
    elements = kernel._source_elements(source)
    if elements.cone_model == "spectral":
        raise NotImplementedError("the shower basis supports the frozen Cherenkov cone")
    sample = kernel.medium.sample(kernel.wavelength.wavelength_nm)
    valid = elements.beta * float(np.min(sample["phase_index"])) > 1
    if not np.any(valid):
        raise ValueError("source is below Cherenkov threshold over the wavelength range")
    original_coefficient = float(elements.coefficient0.sum())
    if not np.all(valid):
        elements = elements.subset(valid)
    field0, field2 = elements.field(0), elements.field(2)
    frame = AxisFrame.of(field0)
    degree = int(kernel.config.source_degree)
    if representation == "basis":
        source0, source2 = compile_axial_slices(
            [field0, field2], degree, omega,
            azimuthal_degree=int(kernel.config.azimuthal_degree),
            cell_m=basis.cell_m, axial_order=basis.axial_order, transverse=basis.transverse,
            stencil_m=basis.stencil_m, stencil_threshold_m=basis.stencil_threshold_m,
            time_step_ns=basis.time_step_ns, frame=frame)
    else:
        from .experimental.axial_fast import compile_axial_source_fast
        source0, source2 = (compile_axial_source_fast(
            item, degree, omega, azimuthal_degree=int(kernel.config.azimuthal_degree),
            cell_m=kernel.config.cell_m, element_order=2, frame=frame)
            for item in (field0, field2))
    timings["compile_seconds"] = perf_counter() - stage

    positions = kernel.detector.positions_m
    looks = np.ascontiguousarray(-kernel.detector.orientations, float)
    area = np.asarray(kernel.detector.effective_area_m2, float)
    alpha = np.ascontiguousarray(kernel.acceptance()["alpha"], float)
    low, high = kernel.config.radial_range_m
    # Radial range, per OM and exactly: every basis point (where the order
    # >=2 source now sits) must lie inside the cache range.  An elongated
    # source such as a muon is not judged by centroid distance +- extent.
    points = source0.points_m()
    point_distance = np.linalg.norm(positions[:, None, :] - points[None, :, :], axis=2)
    inside = ((point_distance.min(axis=1) >= low)
              & (point_distance.max(axis=1) <= high))
    # conservative distance to the true emission points for the safety bound
    offset = elements.midpoints_m - frame.centre_m
    along = offset @ frame.axis
    spread = float(np.max(np.linalg.norm(offset - along[:, None] * frame.axis, axis=1)))
    margin = spread + 2 * basis.cell_m + basis.stencil_m + float(np.max(elements.length_m))
    emitted = float(elements.coefficient0.sum() * np.sum(
        kernel.wavelength.weight_nm / kernel.wavelength.wavelength_nm ** 2)
        + elements.coefficient2.sum() * np.sum(
            kernel.wavelength.weight_nm
            / (kernel.wavelength.wavelength_nm ** 2 * sample["phase_index"] ** 2)))
    peak_acceptance = float(np.max(np.abs(np.asarray(
        kernel.detector.angular_acceptance(np.linspace(-1, 1, 257)), float))))
    lower = np.maximum(point_distance.min(axis=1) - margin, 1e-3)
    upper = (10 * max(emitted, 0.0) * area * peak_acceptance
             * float(np.max(kernel.detector.spectral_weight(kernel.wavelength.wavelength_nm)))
             * np.exp(-float(np.min(sample["absorption_per_m"])) * lower)
             / (4 * np.pi * lower ** 2))
    if np.any((~inside) & (upper >= kernel.config.threshold_pe)):
        raise ValueError("OMs outside the validated directional radial range have a "
                         "conservative estimate above threshold")
    work = np.flatnonzero(inside)
    first = int(np.argmin(elements.start_ns))
    fastest = float(np.min(sample["group_index"]))
    origins = (float(elements.start_ns[first])
               + np.linalg.norm(positions - elements.start_m[first], axis=1)
               / (0.299792458 / fastest))

    charge = np.zeros((len(positions), 3))
    zero0 = replace(source0, channels=np.ascontiguousarray(source0.channels[:, :, :1]))
    zero2 = replace(source2, channels=np.ascontiguousarray(source2.channels[:, :, :1]))
    wavelengths = kernel.wavelength.wavelength_nm
    scale0 = 1.0 / wavelengths ** 2
    scale2 = 1.0 / (wavelengths ** 2 * sample["phase_index"] ** 2)
    efficiency = np.array([float(kernel.detector.spectral_weight(float(w)))
                           for w in wavelengths])
    out_scale = (kernel.wavelength.weight_nm * efficiency)[:, None] * area[None, :]
    stage = perf_counter()
    bundle = store.bundle(kernel, indices)
    prepare = perf_counter() - stage
    stage = perf_counter()
    if len(work):
        charge[work, 2] = apply_multi(bundle, zero0, zero2, positions[work], looks[work],
                                      alpha, scale0, scale2, out_scale[:, work],
                                      frequencies=slice(0, 1))[0].real
    zero_apply = perf_counter() - stage
    timings.update(ge2_prepare_seconds=prepare, zero_apply_seconds=zero_apply)

    stage = perf_counter()
    prompt = kernel.transport_prompt(source, prompt_config,
                                     selection_extra_charge=charge[:, 2])
    timings["prompt_elapsed_seconds"] = perf_counter() - stage
    if not np.array_equal(prompt.relative_time_edges_ns, kernel.config.relative_time_edges_ns):
        raise ValueError("prompt and RTE time-bin edges differ")
    if not np.allclose(prompt.time_origin_ns, origins, rtol=0, atol=1e-10):
        raise ValueError("prompt and RTE time origins differ")
    charge[:, :2] = prompt.charge_orders_pe
    if np.any(prompt.charge_pe[~inside] >= kernel.config.threshold_pe):
        raise ValueError("prompt-active OMs lie outside the validated RTE radial range")
    active = inside & (charge.sum(axis=1) >= kernel.config.threshold_pe)
    if kernel.config.threshold_pe == 0:
        active = inside

    spectrum = np.zeros((len(omega), len(positions)), complex)
    components = np.zeros((len(positions), len(kernel.config.relative_time_edges_ns) - 1, 3))
    stage = perf_counter()
    if np.any(active):
        spectrum[:, active] = apply_multi(bundle, source0, source2, positions[active],
                                          looks[active], alpha, scale0, scale2,
                                          out_scale[:, active])
    timings["full_apply_seconds"] = perf_counter() - stage

    stage = perf_counter()
    spectrum[0] = charge[:, 2]
    if not np.all(prompt.computed[active]):
        raise ValueError("prompt did not compute every hybrid-active OM")
    components[active, :, :2] = prompt.bins_orders_pe[active]
    if np.any(active):
        relative = spectrum[:, active] * np.exp(-1j * omega[:, None] * origins[None, active])
        if len(omega) != len(output_omega):
            from scipy.interpolate import CubicSpline
            relative = (CubicSpline(omega, relative.real, axis=0)(output_omega)
                        + 1j * CubicSpline(omega, relative.imag, axis=0)(output_omega))
        components[active, :, 2] = inverse_bins(output_omega, relative,
                                                kernel.config.relative_time_edges_ns)
    timings["readout_seconds"] = perf_counter() - stage
    elapsed = perf_counter() - began
    metadata = {
        **timings,
        "elapsed_seconds": elapsed,
        "rte_elapsed_seconds": elapsed - timings["prompt_elapsed_seconds"],
        "ge2_source": ("axial-slice shower basis" if representation == "basis"
                       else "reference cloud-in-cell cells"),
        "ge2_basis": asdict(basis),
        "ge2_basis_summary": {**{key: value for key, value in source0.summary.items()
                                 if key != "photons"}, "cells": len(source0.z_m)},
        "ge2_kernel_store_bytes": store.nbytes,
        "source_elements": len(elements),
        "source_coefficient_fraction_dropped":
            1 - float(elements.coefficient0.sum()) / original_coefficient,
        "active_modules": int(active.sum()),
        "threshold_pe": kernel.config.threshold_pe,
        "wavelength_nodes": len(wavelengths),
        "azimuthal_degree": int(kernel.config.azimuthal_degree),
        "source_degree": degree,
        "spectrum_computed_orders": [2],
        "frequency_nodes_computed": len(omega),
        "frequency_nodes_readout": len(output_omega),
        "frequency_interpolation": "cubic after OM time-origin demodulation",
        "prompt_bins": "orders 0 and 1 integrated in physical time",
        "prompt_algorithm": prompt.metadata["prompt_algorithm"],
        "prompt_order1_level": prompt.metadata["order1_level"],
        "inactive_order1_charge_may_be_coarse": bool(np.any(
            np.asarray(prompt.metadata["order1_level"]) == 1)),
    }
    return HybridTransportResponse(
        kernel.detector, omega.copy(), kernel.config.relative_time_edges_ns.copy(),
        spectrum.copy(), components, charge, origins, active, "hybrid-basis", metadata)
