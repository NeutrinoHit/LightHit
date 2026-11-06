r"""Isotropic point flash counted at the first entry into the glass sphere.

The receiver is the one of the shower and muon routes: a photon is counted
where it first enters the OM sphere of radius ``R``, with the module
acceptance of its direction there and the weight ``area / (pi R^2)`` per
entering photon. The three collision orders stay separate.

Orders 1 and >=2 use the first-entry directional table
(:class:`lighthit.directional.DirectionalCache` built with a
``receiver_surface``). An isotropic point flash has only the source channel
``l = 0``, so only the coefficients ``t_(0 lambda 0)`` enter, and in the frame
whose polar axis is the flash-to-module line

    R_o(r, omega) = sum_lambda alpha_lambda sqrt(2 lambda + 1)
                    t_(0 lambda 0, o)(r, omega) P_lambda(c) / (4 pi),

with ``c`` the head-on cosine of the module (``+1`` facing the flash). For a
centre-referenced table ``t_(0 lambda 0) = M_lambda / sqrt(2 lambda + 1)``
and this is exactly :meth:`lighthit.cache.ResponseCache.acceptance_spectrum`;
only the table changes with the receiver.

Order 0 is the direct flux into the sphere. A ray leaving the flash at
polar angle ``theta`` to the line of length ``d`` has impact parameter
``b = d sin(theta) = R sin(psi)`` and enters after
``s = d cos(theta) - R cos(psi)``. With ``sin(theta) d(theta) =
(R/d)^2 sin(psi) cos(psi) d(psi) / cos(theta)``

    q0 = N eta (A / pi R^2) / (4 pi) (R/d)^2
         int_0^(pi/2) dpsi sin(psi) cos(psi) / cos(theta)
         int_0^(2 pi) dphi exp(-mu_t s) A(c cos(theta) + sqrt(1-c^2) sin(theta) cos(phi)),

evaluated by Gauss-Legendre in ``psi`` and the trapezoidal rule in ``phi``
(exact for a cubic acceptance). For ``R -> 0`` it is the centre formula
``N eta A A(c) exp(-mu_t d) / (4 pi d^2)``; at a finite radius the entry
paths are shorter by ``2 R / 3`` on average.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.special import eval_legendre

#: quadrature of the direct flux on the sphere
DIRECT_POLAR_NODES = 24
DIRECT_AZIMUTH_NODES = 16


def direct_nodes(distance_m, head_on_cosine, radius_m, acceptance, *,
                 polar_nodes=DIRECT_POLAR_NODES, azimuth_nodes=DIRECT_AZIMUTH_NODES):
    """Entry paths and weights of the direct flux into the sphere.

    Returns ``(paths, weights)`` of shape ``(modules, nodes)``: one photon
    emitted isotropically is counted ``sum(weights * exp(-mu_t * paths))``
    times, before the factor ``area / (pi R^2)`` and the efficiency.
    """
    d = np.atleast_1d(np.asarray(distance_m, float))
    c = np.clip(np.atleast_1d(np.asarray(head_on_cosine, float)), -1.0, 1.0)
    R = np.broadcast_to(np.asarray(radius_m, float), d.shape)
    if d.shape != c.shape or d.ndim != 1:
        raise ValueError("distances and cosines must be matching 1-D arrays")
    if np.any(d <= R):
        raise ValueError("the flash must lie outside every glass sphere")
    x, w = np.polynomial.legendre.leggauss(int(polar_nodes))
    psi = np.pi / 4 * (x + 1)
    wpsi = np.pi / 4 * w
    phi = 2 * np.pi * np.arange(int(azimuth_nodes)) / int(azimuth_nodes)
    wphi = 2 * np.pi / int(azimuth_nodes)
    sin_t = (R[:, None] / d[:, None]) * np.sin(psi)[None, :]
    cos_t = np.sqrt((1 - sin_t) * (1 + sin_t))
    paths = d[:, None] * cos_t - R[:, None] * np.cos(psi)[None, :]
    radial = ((R / d)[:, None] ** 2 * (wpsi * np.sin(psi) * np.cos(psi))[None, :]
              / cos_t / (4 * np.pi))
    perpendicular = np.sqrt((1 - c) * (1 + c))
    argument = (c[:, None, None] * cos_t[:, :, None]
                + perpendicular[:, None, None] * sin_t[:, :, None] * np.cos(phi)[None, None, :])
    values = np.asarray(acceptance(np.clip(argument, -1.0, 1.0).ravel()), float)
    if values.shape != argument.ravel().shape or not np.isfinite(values).all():
        raise ValueError("acceptance must map cosines to finite values")
    angular = values.reshape(argument.shape).sum(axis=2) * wphi
    return paths, radial * angular


def first_entry_contraction(table, radii_m, head_on_cosine, alpha):
    """Orders 1 and >=2 of a unit isotropic flash, ``(frequency, module, 2)``.

    ``table`` is a directional cache; only its ``l = 0, mu = 0`` coefficients
    are read, so any source degree serves.
    """
    radii = np.atleast_1d(np.asarray(radii_m, float))
    c = np.atleast_1d(np.asarray(head_on_cosine, float))
    alpha = np.asarray(alpha, float)
    pairs = table.table
    if len(alpha) < pairs.acceptance_degree + 1:
        raise ValueError("alpha must cover the acceptance degree of the table")
    coefficients = table.coefficients_at(radii)[:, :, 0]
    out = np.zeros((coefficients.shape[0], len(radii), 2), complex)
    for p, (lam, mu) in enumerate(zip(pairs.pair_lambda, pairs.pair_mu)):
        if mu:
            continue
        factor = alpha[lam] * np.sqrt(2 * lam + 1) * eval_legendre(lam, c) / (4 * np.pi)
        out += coefficients[:, :, p, :] * factor[None, :, None]
    return out


def transport_flash_first_entry(kernel, source, *, method):
    """:class:`lighthit.IsotropicFlash` on the glass receiver (``TransportKernel``)."""
    from .readout import readout_bins
    from .sources import IsotropicFlash
    if not isinstance(source, IsotropicFlash):
        raise TypeError("isotropic method requires IsotropicFlash")
    began = perf_counter()
    config = kernel.config
    detector = kernel.detector
    contract = kernel._receiver_contract()
    R = float(contract["radius_m"])
    omega = config.omega_per_ns
    positions = detector.positions_m
    radii = np.linalg.norm(positions - source.position_m[None, :], axis=1)
    if np.any(radii <= R):
        raise ValueError("isotropic point source lies inside an OM sphere")
    cosines = detector.head_on_cosine(source.position_m)
    wavelengths, photon_weight = kernel._flash_nodes(source)
    response = kernel.acceptance()
    alpha = response["alpha"]
    low, high = config.radial_range_m
    inside = (radii >= low) & (radii <= high)
    paths, weights = direct_nodes(radii, cosines, R, detector.angular_acceptance)
    entry = np.broadcast_to(np.asarray(detector.effective_area_m2, float),
                            radii.shape) / (np.pi * R * R)
    spectrum = np.zeros((len(omega), len(positions), 3), complex)
    ballistic = []
    upper = np.zeros(len(positions))
    for wavelength, photons in zip(wavelengths, photon_weight, strict=True):
        table = kernel._directional_cache_for(float(wavelength))
        medium = table.medium
        efficiency = float(detector.spectral_weight(float(wavelength)))
        scale = photons * detector.effective_area_m2 * efficiency
        if np.any(inside):
            value = first_entry_contraction(table, radii[inside], cosines[inside], alpha)
            spectrum[:, inside, 1:] += value * scale[inside][None, :, None]
        node_charge = photons * efficiency * entry[:, None] * weights * np.exp(
            -medium.extinction_per_m * paths)
        arrival = source.time_ns + paths / medium.speed_m_per_ns
        spectrum[:, :, 0] += np.einsum("dq,wdq->wd", node_charge,
                                       np.exp(1j * omega[:, None, None] * arrival[None]))
        ballistic.append((node_charge, arrival))
        acceptance = np.asarray(detector.angular_acceptance(cosines), float)
        upper += (scale * acceptance * np.exp(-medium.absorption_per_m * (radii - R))
                  / (4 * np.pi * (radii - R) ** 2))
    unsafe = (~inside) & (10 * upper >= config.threshold_pe)
    if np.any(unsafe):
        raise ValueError(
            f"{int(unsafe.sum())} OMs outside radial_range_m have an absorption-only "
            "estimate above threshold; prepare a laser table covering the source-to-OM "
            f"distances (up to {float(np.max(radii[unsafe])):g} m)")
    fastest = min(kernel.medium.band(float(w)).group_index for w in wavelengths)
    origins = source.time_ns + radii / (0.299792458 / fastest)
    relative = spectrum * np.exp(-1j * omega[:, None] * origins[None, :])[:, :, None]
    edges = config.relative_time_edges_ns
    components = np.zeros((len(positions), len(edges) - 1, 3))
    fourier_orders = (1, 2) if len(omega) > 1 else ()
    for order in fourier_orders:
        components[:, :, order] = readout_bins(omega, relative[:, :, order], edges,
                                               config.readout_omega_per_ns)
    rows = np.arange(len(positions))[:, None]
    for charge, arrival in ballistic:
        index = np.searchsorted(edges, arrival - origins[:, None], side="right") - 1
        keep = (index >= 0) & (index < len(edges) - 1)
        np.add.at(components[:, :, 0], (np.broadcast_to(rows, index.shape)[keep], index[keep]),
                  charge[keep])
    charge = spectrum[0].real
    active = inside & (charge.sum(axis=1) >= config.threshold_pe)
    if config.threshold_pe == 0:
        active = inside
    return kernel._response(
        spectrum, components, charge, origins, active, method,
        {"elapsed_seconds": perf_counter() - began,
         "wavelength_nm": np.asarray(wavelengths).tolist(),
         "scattered_receiver": "first-entry",
         "receiver_contract": contract,
         "orders_receiver": {"0": "first-entry", "1": "first-entry", ">=2": "first-entry"},
         "detector_angular_model": "exact l=0 contraction of the first-entry directional table",
         "order0": "direct flux into the glass sphere, entry-path extinction and clock",
         "first_order": "finite-L first-entry table spectrum and Fourier bins",
         "fourier_inverted_orders": list(fourier_orders),
         "acceptance_degree": int(response["degree"]),
         "radial_range_m": list(config.radial_range_m),
         "omitted_outside_range": int((~inside).sum())})
