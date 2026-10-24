"""Order >=2 kernels folded onto a few wavelength node bands.

A Cherenkov source compiled in band ``lambda`` has angular coefficients

    c_lambda,l = sum_i Q_i(lambda) P_l(1 / (beta_i n_lambda)) G_i,l,

where ``G`` holds everything that does not depend on the wavelength (chord
geometry, real harmonics of the step direction, emission clock) and
``Q_i(lambda) = C0_i (1 - 1/(beta_i n_lambda)^2) / lambda^2`` above threshold.
With the detector scale ``a_lambda = w_lambda * eta_lambda`` the response is
``sum_lambda a_lambda K_lambda c_lambda``.  The phase index changes by about
1% over the spectrum, so for every ``l`` the vector over the nodes
``F_l(lambda; beta) = a_lambda Q(lambda; beta) P_l(1/(beta n_lambda))`` lies,
for every ``beta``, close to the span of its values in three bands ``J``:

    a_lambda c_lambda,l  ~=  sum_j L_l[lambda, j] a_j c_j,l ,
    L_l = F_l F_l[J]^+  (least squares on a grid of beta).

``L`` and ``J`` depend on the medium and detector spectral response only, not
on the event.  The response becomes ``sum_j Kt_j,l (a_j c_j,l)`` with folded
kernels ``Kt_j,l = sum_lambda L_l[lambda, j] exp(-mua_lambda r)
exp(i w r / v_lambda) K_lambda,l``; a source is compiled in the ``J`` bands
only and applied with ``len(J)`` instead of all bands.  ``fold_bundle``
re-fits the folded kernels as cubics in log r on the radial intervals of the
pack (optionally subdivided), relative to a common attenuation and flight
phase, in the format of :class:`lighthit.shower_basis.Ge2KernelStore` bundles.

Approximations, each measured on G4 showers (docs/accuracy.md):
the band interpolation (relative column residual ``residual(beta)``, at most
1.4e-4 for beta > 0.8 with the default three bands, up to a few per cent only
at the Cherenkov threshold), the cubic re-fit of the folded radial kernels,
and complex64 storage.  ``source_residual`` returns the light-weighted
residual of a given source as an a-priori relative source error.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from math import comb
from pathlib import Path

import numpy as np

__all__ = ["BandFold", "fold_bundle", "save_folded", "load_folded"]

FORMAT = "lighthit/ge2-band-fold/1"
#: Largest number of band subsets searched exhaustively (3 of 32 nodes: 4960).
EXHAUSTIVE_SUBSETS = 10_000


def _legendre_table(degree, x):
    """P_l(x) for l = 0..degree on an array x; shape (degree+1,) + x.shape."""
    out = np.empty((degree + 1,) + np.shape(x))
    out[0] = 1.0
    if degree:
        out[1] = x
    for ell in range(2, degree + 1):
        out[ell] = ((2 * ell - 1) * x * out[ell - 1] - (ell - 1) * out[ell - 2]) / ell
    return out


@dataclass(frozen=True)
class BandFold:
    """Node bands ``bands`` and per-degree interpolation ``L[l, lambda, j]``.

    ``scale[lambda]`` is the band's detector scale ``w * eta`` (wavelength
    weight times spectral efficiency): the folded source of node ``j`` is
    ``scale[j]`` times its ordinary compiled band source. ``residual_beta``
    and ``residual`` tabulate the worst relative column residual over all
    degrees as a function of beta.
    """
    bands: tuple
    interpolation: np.ndarray
    scale: np.ndarray
    phase_index: np.ndarray
    residual_beta: np.ndarray
    residual: np.ndarray
    beta_design_min: float

    @classmethod
    def of(cls, kernel, *, rank=3, degree=None, beta_design_min=0.8, bands=None):
        """Choose ``rank`` node bands and their interpolation for a kernel.

        The beta grid is dense in ``1/beta - 1`` from 0 to the threshold of
        the largest phase index. Bands are chosen to minimise the worst
        relative residual over all degrees for ``beta > beta_design_min``
        (exhaustive over band subsets up to ``EXHAUSTIVE_SUBSETS`` of them,
        otherwise adding the best band one at a time), unless given explicitly.
        """
        degree = int(kernel.config.source_degree if degree is None else degree)
        wavelength = np.asarray(kernel.wavelength.wavelength_nm, float)
        weight = np.asarray(kernel.wavelength.weight_nm, float)
        efficiency = np.asarray(kernel.detector.spectral_weight(wavelength), float)
        phase = np.asarray(kernel.medium.sample(wavelength)["phase_index"], float)
        scale = weight * efficiency
        nband = len(wavelength)
        if not 1 <= rank <= nband:
            raise ValueError("rank must lie between 1 and the number of bands")
        epsilon = np.r_[0.0, np.geomspace(1e-7, phase.max() - 1.0, 800)]
        beta = 1.0 / (1.0 + epsilon)
        cosine = 1.0 / (beta[None, :] * phase[:, None])
        yield_ = (scale / wavelength ** 2)[:, None] * np.clip(1.0 - cosine ** 2, 0.0, None)
        legendre = _legendre_table(degree, np.clip(cosine, -1.0, 1.0))
        columns = legendre * yield_[None]                   # (l, lambda, beta)
        norm = np.maximum(np.linalg.norm(yield_, axis=0), 1e-300)
        design = beta > beta_design_min

        def residuals(subset):
            nodes = columns[:, list(subset), :]
            mixing = columns @ np.linalg.pinv(nodes)        # (l, lambda, j)
            remainder = columns - mixing @ nodes
            return mixing, np.linalg.norm(remainder, axis=1) / norm[None, :]

        def worst(subset):
            return float(residuals(subset)[1][:, design].max())

        if bands is None and comb(nband, rank) <= EXHAUSTIVE_SUBSETS:
            bands = min(combinations(range(nband), rank), key=worst)
        elif bands is None:
            # Exhaustive search is out of reach (rank 9 of 32 nodes: 2.8e7
            # subsets); grow the subset greedily instead.
            bands = ()
            for _ in range(rank):
                bands = min((tuple(sorted(bands + (band,))) for band in range(nband)
                             if band not in bands), key=worst)
        bands = tuple(int(b) for b in bands)
        mixing, residual = residuals(bands)
        return cls(bands, np.ascontiguousarray(mixing), scale, phase, beta,
                   residual.max(axis=0), float(beta_design_min))

    def source_residual(self, beta, weight):
        """Light-weighted relative band-interpolation residual of a source.

        ``weight`` are nonnegative emitted-light weights (e.g. ``C0``). The
        result bounds the relative error of the interpolated source in the
        detector-weighted spectral norm, element by element; it is not a
        bound on the RTE response itself.
        """
        beta = np.asarray(beta, float)
        weight = np.asarray(weight, float)
        order = np.argsort(self.residual_beta)
        values = np.interp(beta, self.residual_beta[order], self.residual[order])
        return float(np.sum(weight * values) / max(np.sum(weight), 1e-300))

    def metadata(self):
        """JSON-serialisable description (bands, design range, worst residual)."""
        design = self.residual_beta > self.beta_design_min
        return dict(bands=list(self.bands), rank=len(self.bands),
                    beta_design_min=self.beta_design_min,
                    worst_residual_design=float(self.residual[design].max()),
                    worst_residual_all=float(self.residual.max()))


def fold_bundle(bundle, fold, *, subdivide=1, dtype=np.complex64):
    """Folded order >=2 bundle with ``len(fold.bands)`` virtual bands.

    ``bundle`` is a ``Ge2KernelStore`` bundle (all its frequencies). Each
    radial interval is split into ``subdivide`` parts; on each part the folded
    kernel ``sum_lambda L[l, lambda, j] exp(-(mua_lambda - mua_ref) r)
    exp(i w r (1/v_lambda - 1/v_ref)) S_lambda(t)`` is interpolated at four
    equally spaced nodes in log r. ``mua_ref`` is the smallest absorption and
    ``v_ref`` the mean group speed; the returned bundle stores them as its
    common attenuation and flight speed.
    """
    coefficients = bundle["coefficients"]
    nband, nint, ndeg, npair, _, nfreq = coefficients.shape
    if fold.interpolation.shape[:2] != (ndeg, nband):
        raise ValueError("band fold and kernel bundle differ in degree or bands")
    if bundle.get("phase") != "flight":
        raise ValueError("band folding requires flight-phase radial splines")
    log_r = np.asarray(bundle["log_r"], float)
    omega = np.asarray(bundle["omega"], float)
    absorption = np.asarray(bundle["absorption"], float)
    speed = np.asarray(bundle["speed"], float)
    absorption_ref, speed_ref = float(absorption.min()), float(np.mean(speed))
    rank = len(fold.bands)
    parts = np.linspace(0.0, 1.0, int(subdivide) + 1)
    new_log_r = np.concatenate([log_r[j] + parts[:-1] * (log_r[j + 1] - log_r[j])
                                for j in range(nint)] + [log_r[-1:]])
    nodes = np.array([0.0, 1 / 3, 2 / 3, 1.0])
    out = np.zeros((rank, nint * int(subdivide), ndeg, npair, 4, nfreq), dtype)
    mixing = fold.interpolation                              # (l, lambda, j)
    for new in range(nint * int(subdivide)):
        old = new // int(subdivide)
        low, high = new_log_r[new], new_log_r[new + 1]
        at = low + nodes * (high - low)
        t_old = at - log_r[old]
        radius = np.exp(at)
        inverse = np.linalg.inv(np.vander(at - low, 4))     # node values -> t^3..t^0
        c = np.asarray(coefficients[:, old], np.complex128)  # (lambda, l, pair, 4, w)
        t = t_old[None, :, None, None, None]
        values = (((c[:, None, :, :, 0] * t + c[:, None, :, :, 1]) * t
                   + c[:, None, :, :, 2]) * t + c[:, None, :, :, 3])
        factor = (np.exp(-(absorption[:, None] - absorption_ref) * radius[None, :])[:, :, None]
                  * np.exp(1j * omega[None, None, :] * radius[None, :, None]
                           * (1.0 / speed[:, None, None] - 1.0 / speed_ref)))
        values = values * factor[:, :, None, None, :]        # (lambda, node, l, pair, w)
        folded = np.einsum("lmj,mnlpw->jnlpw", mixing, values)  # m: wavelength band
        out[:, new] = np.einsum("an,jnlpw->jlpaw", inverse, folded)
    result = {key: bundle[key] for key in ("degree", "acceptance_degree", "pair_lambda",
                                          "pair_mu", "omega", "phase")}
    result.update(coefficients=out, log_r=new_log_r,
                  absorption=np.full(rank, absorption_ref), speed=np.full(rank, speed_ref),
                  band_fold=fold.metadata(), subdivide=int(subdivide))
    return result


def zero_frequency(folded):
    """Contiguous omega=0 copy of a folded bundle (requires omega[0] == 0)."""
    if float(folded["omega"][0]) != 0.0:
        raise ValueError("the folded bundle does not start at omega = 0")
    return dict(folded, coefficients=np.ascontiguousarray(folded["coefficients"][..., :1]),
                omega=np.asarray(folded["omega"][:1]))


def save_folded(folded, directory):
    """Write a folded bundle as ``folded.npy`` plus ``folded.json``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "folded.npy", folded["coefficients"])
    meta = {key: (np.asarray(value).tolist() if isinstance(value, np.ndarray) else value)
            for key, value in folded.items() if key != "coefficients"}
    (directory / "folded.json").write_text(json.dumps(dict(format=FORMAT, **meta)))


def load_folded(directory, *, mmap=True):
    """Read a folded bundle written by :func:`save_folded`."""
    directory = Path(directory)
    meta = json.loads((directory / "folded.json").read_text())
    if meta.pop("format", None) != FORMAT:
        raise ValueError(f"{directory} is not a {FORMAT} pack")
    for key in ("pair_lambda", "pair_mu"):
        meta[key] = np.asarray(meta[key], np.int64)
    for key in ("omega", "log_r", "absorption", "speed"):
        meta[key] = np.asarray(meta[key], float)
    coefficients = np.load(directory / "folded.npy", mmap_mode="r" if mmap else None)
    return dict(meta, coefficients=coefficients)
