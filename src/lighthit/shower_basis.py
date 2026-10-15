"""Spectral axial-source basis for order >=2 in hybrid transport.

``TransportKernel.transport_hybrid(source, ge2_basis=ShowerBasis())`` combines
physical-time prompt orders 0+1 with the directional RTE response for order
>=2. The source uses cubic axial slices, transverse moments and retained real
angular harmonics. ``Ge2KernelStore`` prepares the radial splines once and
supports memory-mapped reuse.

Its accuracy against finer bases and the per-band reference is summarised in
docs/accuracy.md; the fast shower route is :mod:`lighthit.shower`.
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
_SPECTRAL_APPLY_POINTS = 2048


@dataclass(frozen=True)
class ShowerBasis:
    """Order >=2 source basis: axial slices x harmonics x transverse moments.

    ``cell_m``: slice spacing along the shower axis.  ``axial_order``: 3
    (cubic Lagrange, fourth order) or 1 (linear).  ``transverse``:
    ``"stencil"`` (zeroth, first and diagonal second transverse moments
    exact) or ``"collapse"`` (every point on the axis).  Slices whose rms
    transverse extent is below ``stencil_threshold_m`` keep a single point at the total-light
    centroid. ``time_step_ns``: lattice for ``exp(i omega t)``.
    ``time_deposit="linear"`` has phase error at most ``(omega dt)^2 / 8``;
    optional ``"cubic"`` uses four ticks and the bound ``3 (omega dt)^4 / 128``.
    These are pointwise source-phase bounds, not relative detector-error bounds.
    The angular degree and azimuthal band are those
    of the kernel configuration (``source_degree``, ``azimuthal_degree``).
    ``radial_high_m`` optionally narrows the per-OM source window inside the
    cache; the omitted zero-frequency light is screened against
    ``radial_tail_tolerance_pe``.  ``track_uid`` explicitly selects a nearly
    straight primary track for an approximate axial ``m=0`` component;
    remaining charged elements retain the full shower angular basis.  The
    component cell sizes default to 0.5 m and can be refined separately.
    ``spectral_fusion=True`` reuses source-to-OM geometry across wavelength
    bands with identical points, at the cost of temporary ``complex64``
    source blocks; it is an explicit speed/memory tradeoff.
    ``apply_tolerance_pe`` optionally skips negligible source/receiver pairs
    in the spectral route using an angular/spline norm bound. It bounds the
    omitted numerical spectrum at each computed frequency, up to floating
    roundoff, relative to the same cache and source basis. It does not bound
    transport discretisation or Fourier interpolation error. Zero disables it.
    """
    cell_m: float = 0.25
    axial_order: int = 3
    transverse: str = "stencil"
    stencil_m: float = 0.2
    stencil_threshold_m: float = 0.03
    time_step_ns: float = 0.02
    radial_high_m: float | None = None
    radial_tail_tolerance_pe: float = 1e-4
    track_uid: int | None = None
    track_cell_m: float = 0.5
    track_transverse: str = "stencil"
    secondary_cell_m: float = 0.5
    spectral_fusion: bool = False
    apply_tolerance_pe: float = 0.0
    time_deposit: str = "linear"

    def __post_init__(self):
        if self.time_deposit not in ("linear", "nearest", "cubic"):
            raise ValueError("time_deposit must be linear, nearest or cubic")
        if not (self.cell_m > 0 and self.stencil_m > 0 and self.time_step_ns > 0
                and self.stencil_threshold_m >= 0):
            raise ValueError("ShowerBasis lengths and time step must be positive")
        if self.axial_order not in (1, 3):
            raise ValueError("axial_order must be 1 or 3")
        if self.transverse not in ("stencil", "collapse"):
            raise ValueError("transverse must be 'stencil' or 'collapse'")
        if self.radial_high_m is not None and (not np.isfinite(self.radial_high_m)
                                               or self.radial_high_m <= 0):
            raise ValueError("radial_high_m must be positive and finite")
        if (not np.isfinite(self.radial_tail_tolerance_pe)
                or self.radial_tail_tolerance_pe < 0):
            raise ValueError("radial_tail_tolerance_pe must be finite and nonnegative")
        if (self.track_uid is not None
                and (not isinstance(self.track_uid, (int, np.integer))
                     or isinstance(self.track_uid, (bool, np.bool_)))):
            raise ValueError("track_uid must be an integer or None")
        if (not np.isfinite(self.track_cell_m) or self.track_cell_m <= 0
                or not np.isfinite(self.secondary_cell_m)
                or self.secondary_cell_m <= 0):
            raise ValueError("track and secondary cell sizes must be positive and finite")
        if self.track_transverse not in ("stencil", "collapse"):
            raise ValueError("track_transverse must be 'stencil' or 'collapse'")
        if not isinstance(self.spectral_fusion, (bool, np.bool_)):
            raise ValueError("spectral_fusion must be a boolean")
        if not np.isfinite(self.apply_tolerance_pe) or self.apply_tolerance_pe < 0:
            raise ValueError("apply_tolerance_pe must be finite and nonnegative")


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
        self._adopted = set()

    def __len__(self):
        return len(self._bundles)

    @property
    def nbytes(self):
        total = 0
        seen = set()
        for bundle in self._bundles.values():
            base = bundle["coefficients"]
            while isinstance(base.base, np.ndarray):
                base = base.base
            if id(base) not in seen:
                seen.add(id(base))
                total += base.nbytes
        return int(total)

    @staticmethod
    def _key(kernel, indices):
        return (tuple(kernel._directional_cache_key(float(w))
                      for w in kernel.wavelength.wavelength_nm),
                int(kernel.config.source_degree), tuple(int(i) for i in indices))

    def _adopt(self, kernel):
        """Re-key bundles saved under earlier names of this kernel's tables.

        Packs written by LightHit <= 0.3.0a1 name each wavelength by a
        directional key that also hashed the medium's provenance. A bundle
        whose every key is an alias of this kernel's table (legacy key, or the
        name of the directional table found for it) is moved to the current key.
        """
        aliases = getattr(kernel, "_directional_cache_aliases", None)
        if aliases is None or not self._bundles:
            return
        current = self._key(kernel, ())[0]
        if current in self._adopted:
            return
        self._adopted.add(current)
        names = [aliases(float(w)) for w in kernel.wavelength.wavelength_nm]
        for key in list(self._bundles):
            if (key[0] != current and len(key[0]) == len(names)
                    and all(name in known for name, known in zip(key[0], names))):
                self._bundles.setdefault((current,) + key[1:], self._bundles.pop(key))

    def bundle(self, kernel, indices):
        """The stacked order >=2 spline of every kernel wavelength (built once)."""
        from ._directional_numba import PreparedDirectionalKernel
        self._adopt(kernel)
        key = self._key(kernel, indices)
        if key in self._bundles:
            return self._bundles[key]
        # A prepared dense pack can serve a sparse frequency subset without
        # rebuilding splines or copying their large coefficient array.
        for old_key, prepared in list(self._bundles.items()):
            if old_key[:2] != key[:2]:
                continue
            old_indices = np.asarray(old_key[2], np.int64)
            requested = np.asarray(indices, np.int64)
            locations = np.searchsorted(old_indices, requested)
            if (np.any(locations >= len(old_indices))
                    or not np.array_equal(old_indices[locations], requested)):
                continue
            if len(locations) > 1 and np.all(np.diff(locations) == locations[1] - locations[0]):
                select = slice(int(locations[0]), int(locations[-1]) + 1,
                               int(locations[1] - locations[0]))
            else:
                # Nonuniform subsets need a compact copy; regular grids above
                # remain memory-mapped views of the original pack.
                select = locations
            subset = dict(prepared,
                          coefficients=prepared["coefficients"][..., select],
                          omega=prepared["omega"][select])
            subset["coefficients"].setflags(write=False)
            self._bundles[key] = subset
            return subset
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
        coefficients.setflags(write=False)
        bundle = dict(shared, coefficients=coefficients, _norm_cache={},
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
        self._adopt(kernel)
        selected_indices=np.atleast_1d(np.asarray(indices) if frequencies is None else np.asarray(indices)[frequencies])
        compact=self._bundles.get(self._key(kernel,selected_indices)) if frequencies is not None else None
        bundle = compact if compact is not None else self.bundle(kernel, indices)
        lam = int(np.flatnonzero(np.isclose(kernel.wavelength.wavelength_nm,
                                            float(wavelength_nm), rtol=0, atol=1e-9))[0])
        coefficients = bundle["coefficients"][lam]
        omega = bundle["omega"]
        if frequencies is not None and compact is None:
            coefficients=coefficients[...,frequencies];omega=omega[frequencies]
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
                    for name, value in bundle.items() if name != "coefficients" and not name.startswith("_")}
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
                                           mmap_mode="r" if mmap else None), _norm_cache={})
            store._bundles[key]["coefficients"].setflags(write=False)
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


def _radial_window_report(kernel, source0, source2, positions, low, high):
    """Per-OM source windows and a deliberately loose estimate of the far tail.

    The estimate follows the existing directional radial check, but uses only
    omitted compiled points and their absolute zero-frequency source weights.
    It is a screening estimate, not a mathematical error bound on the RTE.
    A separate cutoff-convergence comparison is needed for a new medium.
    """
    points = source0.points_m()
    wavelengths = kernel.wavelength.wavelength_nm
    weights = kernel.wavelength.weight_nm
    sample = kernel.medium.sample(wavelengths)
    # Cubic axial interpolation has signed weights.  Include every retained
    # angular channel so cancellations in Y_00 cannot hide an omitted point.
    amplitude0 = np.max(np.abs(source0.channels[:, :, 0]), axis=1)
    amplitude2 = np.max(np.abs(source2.channels[:, :, 0]), axis=1)
    photons = np.sqrt(4 * np.pi) * (
        amplitude0 * np.sum(weights / wavelengths ** 2)
        + amplitude2 * np.sum(weights / (wavelengths ** 2 * sample["phase_index"] ** 2)))
    return _window_report_from_photons(
        kernel, points, photons, positions, low, high,
        float(np.min(sample["absorption_per_m"])),
        float(np.max(kernel.detector.spectral_weight(wavelengths))))


def _window_report_from_photons(kernel, points, photons, positions, low, high,
                                absorption, efficiency):
    """Count usable basis points and estimate the omitted far contribution."""
    peak_acceptance = float(np.max(np.abs(np.asarray(
        kernel.detector.angular_acceptance(np.linspace(-1, 1, 257)), float))))
    area = np.asarray(kernel.detector.effective_area_m2, float)
    used = np.zeros(len(positions), np.int64)
    far = np.zeros(len(positions), np.int64)
    near = np.zeros(len(positions), np.int64)
    tail = np.zeros(len(positions), float)
    # Compact quasi-showers otherwise spend most of this pass in one Python
    # call per OM. Bound the temporary coordinate array to about 24 MiB.
    block = max(1, min(len(positions), 1_000_000 // max(len(points), 1)))
    for start in range(0, len(positions), block):
        stop = min(start + block, len(positions))
        radius = np.linalg.norm(points[None] - positions[start:stop, None], axis=2)
        near[start:stop] = np.count_nonzero(radius < low, axis=1)
        far_mask = radius > high
        far[start:stop] = np.count_nonzero(far_mask, axis=1)
        used[start:stop] = len(points) - near[start:stop] - far[start:stop]
        distance = np.where(far_mask, radius, 1.)
        tail[start:stop] = (10 * area[start:stop] * peak_acceptance * efficiency
                            * np.sum(np.where(far_mask,
                                photons[None] * np.exp(-absorption * distance)
                                / (4 * np.pi * distance ** 2), 0.), axis=1))
    return used, far, near, tail


def transport_hybrid_basis(kernel, source, basis, *, frequency_nodes=65,
                           prompt_config=None, store=None, representation="basis",
                           source_partition=None, track_stencil_center="axis"):
    """Prompt orders 0+1 and order >=2 on the axial-slice shower basis.

    The prompt orders and readout grid follow
    :meth:`TransportKernel.transport_hybrid`; only the order >=2 source is
    represented on ``basis``.  A long source gets a per-OM local time origin
    shared by prompt and RTE.  ``representation="cells"`` instead compiles
    the reference cloud-in-cell lattice of the default hybrid
    (``KernelConfig.cell_m``) and runs it through the same driver: a
    validation reference for sources the default hybrid rejects (long muons).
    The optional ``track_uid`` decomposition leaves the prompt calculation
    on all original segments and applies only to ``representation="basis"``.
    """
    if representation not in ("basis", "cells"):
        raise ValueError("representation must be 'basis' or 'cells'")
    from ._directional_numba import PreparedDirectionalKernel  # noqa: F401 (numba check)
    from ._shower_basis import (apply_multi, compile_axial_slices,
                                earliest_element_arrival)
    from .axial_source import AxisFrame
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
        return _transport_hybrid_spectral_basis(
            kernel, source, basis, frequency_nodes=frequency_nodes,
            prompt_config=prompt_config, store=store,
            representation=representation, source_partition=source_partition,
            track_stencil_center=track_stencil_center)
    if source_partition is not None or track_stencil_center != "axis":
        raise ValueError("component source policies require spectral transport")
    if basis.apply_tolerance_pe:
        raise ValueError("apply_tolerance_pe requires the spectral cone model")
    sample = kernel.medium.sample(kernel.wavelength.wavelength_nm)
    valid = elements.beta * float(np.min(sample["phase_index"])) > 1
    if not np.any(valid):
        raise ValueError("source is below Cherenkov threshold over the wavelength range")
    original_coefficient = float(elements.coefficient0.sum())
    if not np.all(valid):
        elements = elements.subset(valid)
    degree = int(kernel.config.source_degree)
    pieces = [("whole source", elements, int(kernel.config.azimuthal_degree))]
    if basis.track_uid is not None:
        if representation != "basis":
            raise ValueError("track_uid is supported only by the shower basis")
        track = elements.uid == basis.track_uid
        if not np.any(track):
            raise ValueError(f"track_uid={basis.track_uid} is absent from the source")
        pieces = [("axial m=0 track", elements.subset(track), 0)]
        if np.any(~track):
            pieces.append(("secondary particles", elements.subset(~track),
                           int(kernel.config.azimuthal_degree)))
    compiled = []
    for label, piece, azimuthal_degree in pieces:
        field0, field2 = piece.field(0), piece.field(2)
        frame = AxisFrame.of(field0)
        if representation == "basis":
            cell_m = (basis.track_cell_m if label == "axial m=0 track" else
                      basis.secondary_cell_m if label == "secondary particles"
                      else basis.cell_m)
            source0, source2 = compile_axial_slices(
                [field0, field2], degree, omega,
                azimuthal_degree=azimuthal_degree,
                cell_m=cell_m, axial_order=basis.axial_order,
                transverse=(basis.track_transverse if label == "axial m=0 track"
                            else basis.transverse), stencil_m=basis.stencil_m,
                stencil_threshold_m=basis.stencil_threshold_m,
                time_step_ns=basis.time_step_ns, time_deposit=basis.time_deposit, frame=frame)
        else:
            from .axial_fast import compile_axial_source_fast
            source0, source2 = (compile_axial_source_fast(
                item, degree, omega, azimuthal_degree=azimuthal_degree,
                cell_m=kernel.config.cell_m, element_order=2, frame=frame)
                for item in (field0, field2))
        compiled.append((label, source0, source2, len(piece)))
    timings["compile_seconds"] = perf_counter() - stage

    positions = kernel.detector.positions_m
    looks = np.ascontiguousarray(-kernel.detector.orientations, float)
    area = np.asarray(kernel.detector.effective_area_m2, float)
    alpha = np.ascontiguousarray(kernel.acceptance()["alpha"], float)
    low, cache_high = kernel.config.radial_range_m
    high = cache_high if basis.radial_high_m is None else basis.radial_high_m
    if not low < high <= cache_high:
        raise ValueError("basis radial_high_m must lie inside the kernel radial range")
    # A long track needs one local source window per OM, not an all-points
    # inside check.  The zero-frequency pass below sums every retained point
    # before the OM threshold is applied; later frequencies use the same mask.
    used = np.zeros(len(positions), np.int64)
    far = np.zeros(len(positions), np.int64)
    near = np.zeros(len(positions), np.int64)
    tail = np.zeros(len(positions), float)
    windows = []
    for _, source0, source2, _ in compiled:
        part_used, part_far, part_near, part_tail = _radial_window_report(
            kernel, source0, source2, positions, low, high)
        windows.append(part_used)
        used += part_used
        far += part_far
        near += part_near
        tail += part_tail
    if np.any(near):
        raise ValueError(
            f"{int(np.count_nonzero(near))} OMs have source points closer than "
            f"the validated radial minimum {low:g} m")
    if np.any(tail > basis.radial_tail_tolerance_pe):
        raise ValueError(
            f"{int(np.count_nonzero(tail > basis.radial_tail_tolerance_pe))} OMs "
            f"have estimated light beyond {high:g} m above "
            f"{basis.radial_tail_tolerance_pe:g} pe; largest estimate "
            f"{float(np.max(tail)):g} pe")
    first = int(np.argmin(elements.start_ns))
    fastest = float(np.min(sample["group_index"]))
    time_width = float(np.ptp(kernel.config.relative_time_edges_ns))
    source_time_span = float(np.max(elements.end_ns) - np.min(elements.start_ns))
    source_extent = float(np.linalg.norm(np.ptp(elements.midpoints_m, axis=0)))
    local_origin = (source_extent > high or source_time_span > 0.5 * time_width)
    if local_origin:
        origins = earliest_element_arrival(
            np.ascontiguousarray(positions, float),
            np.ascontiguousarray(elements.start_m, float),
            np.ascontiguousarray(elements.direction, float),
            np.ascontiguousarray(elements.length_m, float),
            np.ascontiguousarray(elements.start_ns, float), fastest)
    else:
        origins = (float(elements.start_ns[first])
                   + np.linalg.norm(positions - elements.start_m[first], axis=1)
                   / (0.299792458 / fastest))

    charge = np.zeros((len(positions), 3))
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
    for (_, source0, source2, _), part_used in zip(compiled, windows, strict=True):
        part_work = np.flatnonzero(part_used > 0)
        if len(part_work):
            zero0 = replace(source0, channels=np.ascontiguousarray(
                source0.channels[:, :, :1]))
            zero2 = replace(source2, channels=np.ascontiguousarray(
                source2.channels[:, :, :1]))
            charge[part_work, 2] += apply_multi(
                bundle, zero0, zero2, positions[part_work], looks[part_work],
                alpha, scale0, scale2, out_scale[:, part_work],
                frequencies=slice(0, 1), radial_window_m=(low, high))[0].real
    zero_apply = perf_counter() - stage
    timings.update(ge2_prepare_seconds=prepare, zero_apply_seconds=zero_apply)

    stage = perf_counter()
    prompt = kernel.transport_prompt(
        source, prompt_config, selection_extra_charge=charge[:, 2],
        time_origin_ns=origins if local_origin else None)
    timings["prompt_elapsed_seconds"] = perf_counter() - stage
    if not np.array_equal(prompt.relative_time_edges_ns, kernel.config.relative_time_edges_ns):
        raise ValueError("prompt and RTE time-bin edges differ")
    if not np.allclose(prompt.time_origin_ns, origins, rtol=0, atol=1e-10):
        raise ValueError("prompt and RTE time origins differ")
    charge[:, :2] = prompt.charge_orders_pe
    computable = (used > 0) | (prompt.charge_pe > 0)
    active = computable & (charge.sum(axis=1) >= kernel.config.threshold_pe)
    if kernel.config.threshold_pe == 0:
        active = computable

    spectrum = np.zeros((len(omega), len(positions)), complex)
    components = np.zeros((len(positions), len(kernel.config.relative_time_edges_ns) - 1, 3))
    stage = perf_counter()
    for (_, source0, source2, _), part_used in zip(compiled, windows, strict=True):
        part_active = active & (part_used > 0)
        if np.any(part_active):
            spectrum[:, part_active] += apply_multi(
                bundle, source0, source2, positions[part_active],
                looks[part_active], alpha, scale0, scale2,
                out_scale[:, part_active], radial_window_m=(low, high))
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
        "ge2_source": ("axial-slice shower basis with axial m=0 track"
                       if basis.track_uid is not None else
                       "axial-slice shower basis" if representation == "basis"
                       else "reference cloud-in-cell cells"),
        "ge2_basis": asdict(basis),
        "ge2_basis_summary": (
            {**{key: value for key, value in compiled[0][1].summary.items()
                if key != "photons"}, "cells": len(compiled[0][1].z_m)}
            if len(compiled) == 1 else
            {"cells": sum(len(item[1].z_m) for item in compiled),
             "components": [{"name": label, "elements": count,
                             "cells": len(source0.z_m),
                             "azimuthal_degree": source0.azimuthal_degree}
                            for label, source0, _, count in compiled]}),
        "ge2_kernel_store_bytes": store.nbytes,
        "radial_window_m": [float(low), float(high)],
        "radial_windowed_modules": int(np.count_nonzero(far)),
        "radial_source_points_skipped": int(far.sum()),
        "radial_source_points_used_min": int(used.min()),
        "radial_source_points_used_max": int(used.max()),
        "radial_tail_estimate_max_pe": float(tail.max()),
        "radial_tail_tolerance_pe": basis.radial_tail_tolerance_pe,
        "time_origin": ("earliest local segment and fastest group flight"
                        if local_origin else "earliest element start and fastest group flight"),
        "source_extent_m": source_extent,
        "source_time_span_ns": source_time_span,
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
        "prompt_timings": prompt.metadata.get("timings"),
        "prompt_batch_screen": prompt.metadata.get("order1_batch_screen"),
        "prompt_order1_backward_profile": prompt.metadata.get("order1_backward_profile"),
        "inactive_order1_charge_may_be_coarse": bool(np.any(
            np.asarray(prompt.metadata["order1_level"]) == 1)),
    }
    return HybridTransportResponse(
        kernel.detector, omega.copy(), kernel.config.relative_time_edges_ns.copy(),
        spectrum.copy(), components, charge, origins, active, "hybrid-basis", metadata)


def _transport_hybrid_spectral_basis(kernel, source, basis, *, frequency_nodes,
                                     prompt_config, store, representation,
                                     source_partition=None, track_stencil_center="axis"):
    """Compile the wavelength-dependent cone one band at a time.

    The two wavelength-independent Cherenkov yield fields are combined only
    after applying each band's threshold and phase index.  A zero-frequency
    pass selects OMs; a second pass compiles full frequencies only for active
    OMs.  This bounds source memory independently of the wavelength count.
    """
    from ._shower_basis import (apply_multi, apply_multi_spectral, compile_axial_slices,
                                earliest_element_arrival)
    from .axial_source import AxisFrame
    from .readout import inverse_bins
    from .transport import HybridTransportResponse

    began = perf_counter()
    store = kernel.ge2_kernel_store() if store is None else store
    indices = _frequency_indices(kernel, frequency_nodes)
    omega = kernel.config.omega_per_ns[indices]
    output_omega = (kernel.config.readout_omega_per_ns
                    if kernel.config.readout_omega_per_ns is not None else omega)
    elements = kernel._source_elements(source)
    sample = kernel.medium.sample(kernel.wavelength.wavelength_nm)
    max_phase = float(np.max(sample["phase_index"]))
    valid = elements.beta * max_phase > 1
    if not np.any(valid):
        raise ValueError("source is below Cherenkov threshold over the wavelength range")
    original_coefficient = float(elements.coefficient0.sum())
    if not np.all(valid):
        elements = elements.subset(valid)
    if basis.track_uid is not None and representation != "basis":
        raise ValueError("track_uid is supported only by the shower basis")
    pieces = [("whole source", elements, int(kernel.config.azimuthal_degree),
               basis.cell_m, basis.transverse)]
    if basis.track_uid is not None:
        track = elements.uid == basis.track_uid
        if not np.any(track):
            raise ValueError(f"track_uid={basis.track_uid} is absent from the source")
        pieces = [("axial m=0 track", elements.subset(track), 0,
                   basis.track_cell_m, basis.track_transverse)]
        if np.any(~track):
            pieces.append(("secondary particles", elements.subset(~track),
                           int(kernel.config.azimuthal_degree),
                           basis.secondary_cell_m, basis.transverse))
    if source_partition is not None:
        pieces = list(source_partition(elements, basis))
        rows = np.concatenate([piece.row_index for _, piece, *_ in pieces])
        if (not np.array_equal(np.sort(rows), np.sort(elements.row_index))
                or len({label for label, *_ in pieces}) != len(pieces)):
            raise ValueError("source partition must cover each element once with unique labels")
    component_share = {label: (float(piece.coefficient0.sum()/elements.coefficient0.sum())
                              if source_partition is not None else 1./len(pieces))
                       for label, piece, *_ in pieces}
    pieces = [(label, piece, azimuthal, cell_m, transverse,
               AxisFrame.of(piece.field(0, phase_index=max_phase)))
              for label, piece, azimuthal, cell_m, transverse in pieces]
    geometry_reference = {
        label: piece.field(0, phase_index=max_phase)
        for label, piece, *_ in pieces
        if (track_stencil_center == "centroid" and basis.track_uid is not None
            and np.all(piece.uid == basis.track_uid))}
    def band_pieces():
        bands = list(enumerate(zip(wavelengths, sample["phase_index"], strict=True)))
        if source_partition is not None:
            for info in pieces:
                for lam, (wavelength, phase) in bands:
                    yield lam, wavelength, phase, *info
        else:
            for lam, (wavelength, phase) in bands:
                for info in pieces:
                    yield lam, wavelength, phase, *info

    wavelengths = kernel.wavelength.wavelength_nm
    weights = kernel.wavelength.weight_nm
    efficiency = np.asarray(kernel.detector.spectral_weight(wavelengths), float)
    positions = np.asarray(kernel.detector.positions_m, float)
    looks = np.ascontiguousarray(-kernel.detector.orientations, float)
    area = np.asarray(kernel.detector.effective_area_m2, float)
    alpha = np.ascontiguousarray(kernel.acceptance()["alpha"], float)
    degree = int(kernel.config.source_degree)
    low, cache_high = kernel.config.radial_range_m
    high = cache_high if basis.radial_high_m is None else basis.radial_high_m
    if not low < high <= cache_high:
        raise ValueError("basis radial_high_m must lie inside the kernel radial range")

    stage = perf_counter()
    bundle = store.bundle(kernel, indices)
    prepare_seconds = perf_counter() - stage

    def one_band(lam):
        return dict(bundle, coefficients=bundle["coefficients"][lam:lam + 1],
                    absorption=bundle["absorption"][lam:lam + 1],
                    speed=bundle["speed"][lam:lam + 1])

    def source_field(piece, phase, wavelength):
        mask = piece.beta * phase > 1
        if not np.any(mask):
            return None
        subset = piece if np.all(mask) else piece.subset(mask)
        first = subset.field(0, phase_index=phase)
        second = subset.field(2, phase_index=phase)
        first_scale = 1.0 / wavelength ** 2
        second_scale = first_scale / phase ** 2
        return replace(first, photons=np.ascontiguousarray(
            first_scale * first.photons - second_scale * second.photons))

    def compile_field(field, frame, azimuthal, cell_m, transverse, frequencies,
                      point_selection=None, reference_geometry=None):
        if representation == "cells":
            from .axial_fast import compile_axial_source_fast
            compiled = compile_axial_source_fast(
                field, degree, frequencies, azimuthal_degree=azimuthal,
                cell_m=kernel.config.cell_m, element_order=2, frame=frame)
            if point_selection is not None:
                compiled = replace(compiled, z_m=compiled.z_m[point_selection],
                                   transverse_m=compiled.transverse_m[point_selection],
                                   channels=compiled.channels[point_selection])
            return compiled
        return compile_axial_slices(
            field, degree, frequencies, azimuthal_degree=azimuthal,
            cell_m=cell_m, axial_order=basis.axial_order,
            transverse=transverse, stencil_m=basis.stencil_m,
            stencil_threshold_m=basis.stencil_threshold_m,
            time_step_ns=basis.time_step_ns, time_deposit=basis.time_deposit, frame=frame,
            point_selection=point_selection,
            stencil_center="centroid" if reference_geometry is not None else "axis",
            **({"geometry_reference": reference_geometry} if reference_geometry is not None else {}),
            channel_dtype=np.complex64 if basis.spectral_fusion else np.complex128)

    shared_points = separate_points = 0

    def apply_spectral_group(entries, target, point_fraction=1.0):
        """Apply consecutive bands with shared points to selected OMs."""
        nonlocal shared_points, separate_points
        first_lam, first_source, _ = entries[0]
        count = len(first_source.z_m)
        if count > _SPECTRAL_APPLY_POINTS:
            # Leading-row slices are views. Split BEFORE selecting shared
            # wavelength points, whose fancy indexing otherwise copies entire
            # multi-GB source blocks, followed by another stacked copy in apply.
            # Scaling epsilon with point count preserves the per-pair budget.
            for start in range(0, count, _SPECTRAL_APPLY_POINTS):
                stop = min(start + _SPECTRAL_APPLY_POINTS, count)
                part = [(lam, replace(item, z_m=item.z_m[start:stop],
                         transverse_m=item.transverse_m[start:stop],
                         channels=item.channels[start:stop]), used)
                        for lam, item, used in entries]
                apply_spectral_group(part, target, point_fraction * (stop-start)/count)
            return
        last_lam = entries[-1][0]
        if len(entries) > 1:
            points = first_source.points_m()
            common = np.ones(len(points), bool)
            for _, item, _ in entries[1:]:
                common &= np.all(item.points_m() == points, axis=1)
            if not np.all(common):
                # A few wavelength-dependent thin-slice centroids used to
                # disable fusion for the entire track. Share the identical
                # points and evaluate only the exceptions separately; no
                # coordinate or spectral approximation is introduced.
                def subset(item, mask):
                    return replace(item, z_m=item.z_m[mask],
                                   transverse_m=item.transverse_m[mask],
                                   channels=np.ascontiguousarray(item.channels[mask]))
                fraction = float(common.mean())
                if np.any(common):
                    apply_spectral_group([(lam, subset(item, common), used)
                                          for lam, item, used in entries], target,
                                         point_fraction * fraction)
                for lam, item, used in entries:
                    apply_spectral_group([(lam, subset(item, ~common), used)], target,
                                         point_fraction * (1 - fraction))
                return
            shared_points += len(points) * len(entries)
        else:
            separate_points += len(first_source.z_m)
        is_zero = target.ndim == 1
        frequency_slice = slice(0, 1) if is_zero else slice(None)
        prepared=store._bundles.get(store._key(kernel,indices[:1]),bundle) if is_zero else bundle
        band = dict(
            prepared,
            coefficients=prepared["coefficients"][first_lam:last_lam + 1][
                ..., frequency_slice],
            absorption=bundle["absorption"][first_lam:last_lam + 1],
            speed=bundle["speed"][first_lam:last_lam + 1],
            omega=omega[frequency_slice])
        selected = np.zeros(len(positions), bool)
        for _, _, part_used in entries:
            selected |= part_used > 0
        if not is_zero:
            selected &= active
        work = np.flatnonzero(selected)
        if not len(work):
            return
        out_scale = np.stack([
            weights[lam] * efficiency[lam] * area[work]
            for lam, _, _ in entries])
        if len(entries) == 1:
            response = apply_multi(
                band, first_source, first_source, positions[work], looks[work],
                alpha, np.ones(1), np.zeros(1), out_scale,
                radial_window_m=(low, high),
                absolute_tolerance_pe=basis.apply_tolerance_pe * point_fraction * len(entries)
                * first_source.summary.get("component_budget_share", 1./len(pieces)) / len(wavelengths))
        else:
            response = apply_multi_spectral(
                band, [item[1] for item in entries], positions[work],
                looks[work], alpha, out_scale, radial_window_m=(low, high),
                absolute_tolerance_pe=basis.apply_tolerance_pe * point_fraction * len(entries)
                * first_source.summary.get("component_budget_share", 1./len(pieces)) / len(wavelengths))
        if is_zero:
            target[work] += response[0].real
        else:
            target[:, work] += response

    def spectral_groups(entries, max_bands):
        pending = []
        for entry in entries:
            lam, compiled, _ = entry
            if pending and (len(pending) >= max_bands
                            or lam != pending[-1][0] + 1
                            or compiled.points_m().shape != pending[0][1].points_m().shape):
                yield pending
                pending = []
            pending.append(entry)
        if pending:
            yield pending

    used = np.zeros(len(positions), np.int64)
    far = np.zeros(len(positions), np.int64)
    near = np.zeros(len(positions), np.int64)
    tail = np.zeros(len(positions), float)
    charge = np.zeros((len(positions), 3))
    summaries = []
    zero_windows = {}
    zero_points = {}
    zero_entries = {label: [] for label, *_ in pieces}
    compile_zero_seconds = zero_apply_seconds = 0.0
    previous_component = None
    for lam, wavelength, phase, label, piece, azimuthal, cell_m, transverse, frame in band_pieces():
        if source_partition is not None and basis.spectral_fusion and previous_component != label:
            if previous_component is not None:
                stage = perf_counter()
                for group in spectral_groups(zero_entries.pop(previous_component), 9):
                    apply_spectral_group(group, charge[:, 2])
                zero_apply_seconds += perf_counter() - stage
                group = None
            previous_component = label
        band = one_band(lam)
        field = source_field(piece, float(phase), float(wavelength))
        if field is None:
            continue
        stage = perf_counter()
        compiled = compile_field(field, frame, azimuthal, cell_m, transverse,
                                 omega[:1], reference_geometry=geometry_reference.get(label))
        compiled.summary["component_budget_share"] = component_share[label]
        compile_zero_seconds += perf_counter() - stage
        photons = (np.sqrt(4 * np.pi)
                   * np.max(np.abs(compiled.channels[:, :, 0]), axis=1)
                   * weights[lam])
        points = compiled.points_m()
        part_used, part_far, part_near, part_tail = _window_report_from_photons(
            kernel, points, photons, positions, low, high,
            float(sample["absorption_per_m"][lam]), float(efficiency[lam]))
        zero_windows[(lam, label)] = part_used
        zero_points[(lam, label)] = points
        used += part_used
        far += part_far
        near += part_near
        tail += part_tail
        if np.any(part_near):
            raise ValueError(
                f"{int(np.count_nonzero(part_near))} OMs have source points closer than "
                f"the validated radial minimum {low:g} m")
        summaries.append({"wavelength_nm": float(wavelength), "name": label,
                          "elements": len(field), "cells": len(compiled.z_m),
                          "azimuthal_degree": azimuthal})
        if basis.spectral_fusion:
            compiled = replace(compiled, channels=np.ascontiguousarray(
                compiled.channels, dtype=np.complex64))
            zero_entries[label].append((lam, compiled, part_used))
            continue
        work = np.flatnonzero(part_used > 0)
        if len(work):
            stage = perf_counter()
            detector_scale = (weights[lam] * efficiency[lam] * area[work])[None, :]
            charge[work, 2] += apply_multi(
                band, compiled, compiled, positions[work], looks[work], alpha,
                np.ones(1), np.zeros(1), detector_scale,
                frequencies=slice(0, 1), radial_window_m=(low, high),
                absolute_tolerance_pe=basis.apply_tolerance_pe * component_share[label] / len(wavelengths))[0].real
            zero_apply_seconds += perf_counter() - stage
    if basis.spectral_fusion:
        for entries in zero_entries.values():
            for group in spectral_groups(entries, len(entries)):
                stage = perf_counter()
                apply_spectral_group(group, charge[:, 2])
                zero_apply_seconds += perf_counter() - stage
    # These angular tables were needed only for the charge pass. Keeping all
    # wavelengths alive through prompt processing and the full-frequency pass
    # unnecessarily increases the peak memory of long events.
    zero_entries.clear()
    entries = group = compiled = field = None
    if np.any(near):
        raise ValueError(
            f"{int(np.count_nonzero(near))} OMs have source points closer than "
            f"the validated radial minimum {low:g} m")
    if np.any(tail > basis.radial_tail_tolerance_pe):
        raise ValueError(
            f"{int(np.count_nonzero(tail > basis.radial_tail_tolerance_pe))} OMs "
            f"have estimated light beyond {high:g} m above "
            f"{basis.radial_tail_tolerance_pe:g} pe; largest estimate "
            f"{float(np.max(tail)):g} pe")

    fastest = float(np.min(sample["group_index"]))
    time_width = float(np.ptp(kernel.config.relative_time_edges_ns))
    source_time_span = float(np.max(elements.end_ns) - np.min(elements.start_ns))
    source_extent = float(np.linalg.norm(np.ptp(elements.midpoints_m, axis=0)))
    local_origin = (source_extent > high or source_time_span > 0.5 * time_width)
    if local_origin:
        origins = earliest_element_arrival(
            np.ascontiguousarray(positions),
            np.ascontiguousarray(elements.start_m),
            np.ascontiguousarray(elements.direction),
            np.ascontiguousarray(elements.length_m),
            np.ascontiguousarray(elements.start_ns), fastest)
    else:
        first = int(np.argmin(elements.start_ns))
        origins = (float(elements.start_ns[first])
                   + np.linalg.norm(positions - elements.start_m[first], axis=1)
                   / (0.299792458 / fastest))
    stage = perf_counter()
    prompt = kernel.transport_prompt(
        source, prompt_config, selection_extra_charge=charge[:, 2] + basis.apply_tolerance_pe,
        time_origin_ns=origins if local_origin else None)
    prompt_seconds = perf_counter() - stage
    if not np.array_equal(prompt.relative_time_edges_ns, kernel.config.relative_time_edges_ns):
        raise ValueError("prompt and RTE time-bin edges differ")
    if not np.allclose(prompt.time_origin_ns, origins, rtol=0, atol=1e-10):
        raise ValueError("prompt and RTE time origins differ")
    charge[:, :2] = prompt.charge_orders_pe
    if basis.apply_tolerance_pe and np.any(
            np.abs(charge.sum(axis=1) - kernel.config.threshold_pe) <= basis.apply_tolerance_pe):
        # No ambiguous threshold decisions: recompute without pair screening.
        return _transport_hybrid_spectral_basis(
            kernel, source, replace(basis, apply_tolerance_pe=0.),
            frequency_nodes=frequency_nodes, prompt_config=prompt_config,
            store=store, representation=representation, source_partition=source_partition,
            track_stencil_center=track_stencil_center)
    computable = (used > 0) | (prompt.charge_pe > 0)
    active = computable & (charge.sum(axis=1) >= kernel.config.threshold_pe)
    if kernel.config.threshold_pe == 0:
        active = computable

    spectrum = np.zeros((len(omega), len(positions)), complex)
    components = np.zeros((len(positions), len(kernel.config.relative_time_edges_ns) - 1, 3))
    compile_full_seconds = full_apply_seconds = 0.0
    pending_by_label = {label: [] for label, *_ in pieces}
    # Full frequencies are needed only where an active OM can see a point
    # inside the existing radial window. Charge and the far-tail diagnostic
    # above still used every point. Union wavelength masks with equal layouts
    # so this exact sparsity reduction does not disable spectral fusion.
    full_selection = {}
    for (_, label), points in zero_points.items():
        key = (label, len(points))
        selected = full_selection.setdefault(key, np.zeros(len(points), bool))
        for receiver in positions[active]:
            # Include a roundoff guard for the rotation to the local frame
            # used by the apply kernel at the radial boundary.
            selected |= np.sum((points - receiver) ** 2, axis=1) <= (high + 1e-8) ** 2
    previous_component = None
    for lam, wavelength, phase, label, piece, azimuthal, cell_m, transverse, frame in band_pieces():
        if source_partition is not None and basis.spectral_fusion and previous_component != label:
            if previous_component is not None:
                pending = pending_by_label.pop(previous_component)
                if pending:
                    stage = perf_counter()
                    apply_spectral_group(pending, spectrum)
                    full_apply_seconds += perf_counter() - stage
                pending = []
            previous_component = label
        band = one_band(lam)
        field = source_field(piece, float(phase), float(wavelength))
        if field is None:
            continue
        stage = perf_counter()
        selected_points = full_selection[(label, len(zero_points[(lam, label)]))]
        if not np.any(selected_points):
            continue
        if basis.spectral_fusion:
            pending = pending_by_label[label]
            max_bands = 9 if azimuthal == 0 else 2
            if pending and (len(pending) >= max_bands
                            or lam != pending[-1][0] + 1
                            or int(selected_points.sum()) != len(pending[0][1].z_m)):
                # Consume complete bands before allocating the next one.
                # Otherwise a two-band contraction retains a third full
                # source block throughout its application.
                stage_apply = perf_counter()
                apply_spectral_group(pending, spectrum)
                full_apply_seconds += perf_counter() - stage_apply
                pending_by_label[label] = []
                pending = []
                compiled = None
            stage = perf_counter()
        compiled = compile_field(field, frame, azimuthal, cell_m, transverse,
                                 omega, point_selection=selected_points,
                                 reference_geometry=geometry_reference.get(label))
        compiled.summary["component_budget_share"] = component_share[label]
        compile_full_seconds += perf_counter() - stage
        if not np.array_equal(compiled.points_m(), zero_points[(lam, label)][selected_points]):
            raise ValueError("spectral source points differ between zero and full frequency passes")
        if basis.spectral_fusion:
            compiled = replace(compiled, channels=np.ascontiguousarray(
                compiled.channels, dtype=np.complex64))
            pending = pending_by_label[label]
            max_bands = 9 if azimuthal == 0 else 2
            if pending and (len(pending) >= max_bands
                            or lam != pending[-1][0] + 1
                            or compiled.points_m().shape != pending[0][1].points_m().shape):
                stage = perf_counter()
                apply_spectral_group(pending, spectrum)
                full_apply_seconds += perf_counter() - stage
                pending = []
            pending.append((lam, compiled, zero_windows[(lam, label)]))
            pending_by_label[label] = pending
            continue
        if not np.any(active):
            continue
        work = np.flatnonzero(active & (zero_windows[(lam, label)] > 0))
        if len(work):
            stage = perf_counter()
            detector_scale = (weights[lam] * efficiency[lam] * area[work])[None, :]
            spectrum[:, work] += apply_multi(
                band, compiled, compiled, positions[work], looks[work], alpha,
                np.ones(1), np.zeros(1), detector_scale,
                radial_window_m=(low, high),
                absolute_tolerance_pe=basis.apply_tolerance_pe * component_share[label] / len(wavelengths))
            full_apply_seconds += perf_counter() - stage
    if basis.spectral_fusion:
        for pending in pending_by_label.values():
            if pending:
                stage = perf_counter()
                apply_spectral_group(pending, spectrum)
                full_apply_seconds += perf_counter() - stage
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
    bin_error_bound = np.zeros(len(kernel.config.relative_time_edges_ns) - 1)
    if basis.apply_tolerance_pe:
        interpolation = np.eye(len(omega))
        if len(omega) != len(output_omega):
            from scipy.interpolate import CubicSpline
            interpolation = CubicSpline(omega, interpolation, axis=0)(output_omega)
        # Induced norm of the actual cubic interpolation + bin inversion.
        # The origin demodulation has unit modulus and needs no extra factor.
        real_map = inverse_bins(output_omega, interpolation, kernel.config.relative_time_edges_ns)
        imag_map = inverse_bins(output_omega, 1j * interpolation, kernel.config.relative_time_edges_ns)
        bin_error_bound = basis.apply_tolerance_pe * np.hypot(real_map, imag_map).sum(axis=0)
    readout_seconds = perf_counter() - stage
    elapsed = perf_counter() - began
    metadata = {
        "elapsed_seconds": elapsed, "rte_elapsed_seconds": elapsed - prompt_seconds,
        "apply_spectrum_absolute_error_bound_pe": basis.apply_tolerance_pe,
        "apply_bin_absolute_error_bound_pe": float(bin_error_bound.max()),
        "apply_bins_l1_error_bound_pe": float(bin_error_bound.sum()),
        "apply_bound_scope": "omitted pairs at computed frequencies, same numerical cache; floating-roundoff guard",
        "spectral_geometry_shared_points": shared_points,
        "spectral_geometry_separate_points": separate_points,
        "full_source_points_available": sum(len(mask) for mask in full_selection.values()),
        "full_source_points_kept": sum(int(mask.sum()) for mask in full_selection.values()),
        "ge2_source": "wavelength-dependent axial-slice shower basis",
        "source_partition_components": len(pieces),
        "track_stencil_center": track_stencil_center,
        "cone_model": "spectral", "ge2_basis": asdict(basis),
        "ge2_basis_summary": {"max_cells_per_band": max(
            (item["cells"] for item in summaries), default=0),
            "bands": summaries},
        "ge2_kernel_store_bytes": store.nbytes,
        "ge2_prepare_seconds": prepare_seconds,
        "compile_zero_seconds": compile_zero_seconds,
        "zero_apply_seconds": zero_apply_seconds,
        "compile_full_seconds": compile_full_seconds,
        "full_apply_seconds": full_apply_seconds,
        "prompt_elapsed_seconds": prompt_seconds,
        "readout_seconds": readout_seconds,
        "radial_window_m": [float(low), float(high)],
        "radial_windowed_modules": int(np.count_nonzero(far)),
        "radial_source_points_skipped": int(far.sum()),
        "radial_tail_estimate_max_pe": float(tail.max()),
        "radial_tail_tolerance_pe": basis.radial_tail_tolerance_pe,
        "time_origin": ("earliest local segment and fastest group flight"
                        if local_origin else "earliest element start and fastest group flight"),
        "source_extent_m": source_extent, "source_time_span_ns": source_time_span,
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
        "prompt_timings": prompt.metadata.get("timings"),
        "prompt_batch_screen": prompt.metadata.get("order1_batch_screen"),
        "prompt_order1_backward_profile": prompt.metadata.get("order1_backward_profile"),
        "inactive_order1_charge_may_be_coarse": bool(np.any(
            np.asarray(prompt.metadata["order1_level"]) == 1)),
    }
    return HybridTransportResponse(
        kernel.detector, omega.copy(), kernel.config.relative_time_edges_ns.copy(),
        spectrum.copy(), components, charge, origins, active, "hybrid-basis", metadata)
