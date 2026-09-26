"""Slow, independent reference for the exactly-one-scattering contribution.

Nothing here is used by :meth:`TransportKernel.transport_prompt`. It exists so
that the fast prompt kernels can be checked against a formulation that shares
no quadrature, no interpolation and no data layout with them.

One directed photon ("pencil") leaves ``x`` in direction ``s0``. A point module
at ``R`` looks along ``n_look``: a photon travelling in direction ``s1`` is
accepted with ``A(s1 . n_look)``. With ``r = |R - x|`` and ``theta`` the angle
between ``s0`` and ``R - x``, the scatter point is parametrised by the
scattering angle ``chi`` in ``(theta, pi)``:

    S(chi)   = r [cos(theta) + sin(theta) tan(chi / 2)]   (total path)
    rho(chi) = r sin(theta) / sin(chi)                    (scatter -> module)
    s1(chi)  = cos(chi) s0 + sin(chi) e_perp,
    e_perp   = (r_hat - cos(theta) s0) / sin(theta),

and the once-scattered fluence read by the module is

    N1 = mu_s / (r sin theta) * int_theta^pi dchi
         exp(-mu_t S) p_HG(cos chi) A(s1 . n_look),

arriving at ``t = t_emit + S / v``. With ``A == 1`` this is exactly the
directed-point density of :func:`lighthit.single.single_scattering_rate`
after the change of variables ``dS = (S - r nu) / rho dl``.

``pencil_order1_bins`` integrates that expression over the actual bin edges
with adaptive quadrature, one bin at a time. ``pencil_order1_monte_carlo``
estimates the same quantity by sampling the first interaction along the ray
and scoring the next-event contribution; it uses neither ``chi`` nor ``S``.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .ballistic import acceptance_from_coefficients
from .single import hg_phase

__all__ = ["pencil_geometry", "pencil_order1_bins", "pencil_order1_monte_carlo",
           "ring_order1_bins"]


def pencil_geometry(position_m, direction, receiver_m, look):
    """Return ``(r, theta, c1, c3, e_perp)`` for one pencil and one module."""
    x = np.asarray(position_m, float)
    s0 = np.asarray(direction, float)
    s0 = s0 / np.linalg.norm(s0)
    n = np.asarray(look, float)
    n = n / np.linalg.norm(n)
    d = np.asarray(receiver_m, float) - x
    r = float(np.linalg.norm(d))
    r_hat = d / r
    cosine = float(r_hat @ s0)
    perp = r_hat - cosine * s0
    sine = float(np.linalg.norm(perp))
    if sine <= 1e-14:
        raise ValueError("pencil points exactly at the module; order 1 is singular")
    e_perp = perp / sine
    theta = float(np.arctan2(sine, cosine))
    return r, theta, float(s0 @ n), float(e_perp @ n), e_perp


def _integrand(chi, r, theta, c1, c3, medium, alpha):
    sine = np.sin(theta)
    path = r * (np.cos(theta) + sine * np.tan(0.5 * chi))
    x = np.cos(chi) * c1 + np.sin(chi) * c3
    acceptance = float(acceptance_from_coefficients(alpha, np.clip([x], -1, 1))[0])
    return (medium.scattering_per_m / (r * sine)
            * np.exp(-medium.extinction_per_m * path)
            * float(hg_phase(np.cos(chi), medium.g)) * acceptance)


def _chi_of_path(path, r, theta):
    sine, cosine = np.sin(theta), np.cos(theta)
    value = (path / r - cosine) / sine
    return 2.0 * np.arctan(value)


def pencil_order1_bins(r, theta, c1, c3, medium, alpha, relative_edges_ns, *,
                       emission_time_ns=0.0, time_origin_ns=0.0,
                       epsrel=1e-10):
    """Adaptive-quadrature first order: total charge and bins over real edges.

    Returns ``(charge, bins)`` per emitted photon and unit effective area.
    ``charge`` is integrated over all times; ``bins`` covers only the
    requested window.
    """
    edges = np.asarray(relative_edges_ns, float) + float(time_origin_ns)
    speed = medium.speed_m_per_ns
    args = (r, theta, c1, c3, medium, alpha)
    forward = min(np.pi, theta + max(0.05, (1 - abs(medium.g)) / max(abs(medium.g), 1e-3) ** .5))
    charge = quad(_integrand, theta, forward, args=args, epsrel=epsrel,
                  epsabs=0, limit=400)[0]
    charge += quad(_integrand, forward, np.pi, args=args, epsrel=epsrel,
                   epsabs=0, limit=400)[0] if forward < np.pi else 0.0
    bins = np.zeros(len(edges) - 1)
    for k in range(len(bins)):
        low_path = speed * (edges[k] - emission_time_ns)
        high_path = speed * (edges[k + 1] - emission_time_ns)
        if high_path <= r:
            continue
        low = theta if low_path <= r else _chi_of_path(low_path, r, theta)
        high = _chi_of_path(high_path, r, theta)
        if high <= low:
            continue
        points = [p for p in (forward,) if low < p < high]
        bins[k] = quad(_integrand, low, high, args=args, epsrel=epsrel,
                       epsabs=0, limit=400, points=points or None)[0]
    return charge, bins


def pencil_order1_monte_carlo(position_m, direction, receiver_m, look, medium,
                              alpha, *, samples=200_000, seed=0):
    """Next-event Monte Carlo of the same first-order fluence.

    The first interaction distance is sampled from ``mu_t exp(-mu_t l)``; the
    photon survives it as a scatter with probability ``mu_s / mu_t`` and the
    module scores ``p(s0.s1) exp(-mu_t rho) A(s1.n) / rho^2``. This is an
    unbiased estimator of ``N1`` that shares no variable with ``chi``.
    Returns ``(mean, standard_error)``.
    """
    rng = np.random.default_rng(seed)
    s0 = np.asarray(direction, float)
    s0 = s0 / np.linalg.norm(s0)
    n = np.asarray(look, float)
    n = n / np.linalg.norm(n)
    mu_t = medium.extinction_per_m
    length = rng.exponential(1.0 / mu_t, size=samples)
    point = np.asarray(position_m, float)[None, :] + length[:, None] * s0[None, :]
    towards = np.asarray(receiver_m, float)[None, :] - point
    rho = np.linalg.norm(towards, axis=1)
    s1 = towards / rho[:, None]
    score = ((medium.scattering_per_m / mu_t) * hg_phase(s1 @ s0, medium.g)
             * np.exp(-mu_t * rho) / rho ** 2
             * acceptance_from_coefficients(alpha, np.clip(s1 @ n, -1, 1)))
    return float(score.mean()), float(score.std(ddof=1) / np.sqrt(samples))


def ring_order1_bins(position_m, axis, cone_cosine, receiver_m, look, medium,
                     alpha, relative_edges_ns, *, emission_time_ns=0.0,
                     time_origin_ns=0.0, phi_nodes=None, epsrel=1e-8):
    """First order of one point Cherenkov cone, averaged over its azimuth.

    The azimuth integral uses an adaptive rule with the closest approach to
    the module direction supplied as a break point. Intended for small
    reference problems only.
    """
    u = np.asarray(axis, float)
    u = u / np.linalg.norm(u)
    helper = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    d = np.asarray(receiver_m, float) - np.asarray(position_m, float)
    lateral = d - (d @ u) * u
    e1 = lateral / np.linalg.norm(lateral) if np.linalg.norm(lateral) > 1e-12 else np.cross(u, helper)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    sine = np.sqrt(1 - cone_cosine ** 2)
    edges = np.asarray(relative_edges_ns, float)

    def pencil(phi):
        s0 = cone_cosine * u + sine * (np.cos(phi) * e1 + np.sin(phi) * e2)
        r, theta, c1, c3, _ = pencil_geometry(position_m, s0, receiver_m, look)
        return pencil_order1_bins(r, theta, c1, c3, medium, alpha, edges,
                                  emission_time_ns=emission_time_ns,
                                  time_origin_ns=time_origin_ns, epsrel=1e-9)

    if phi_nodes is not None:
        nodes = (np.arange(phi_nodes) + 0.5) * 2 * np.pi / phi_nodes - np.pi
        values = [pencil(phi) for phi in nodes]
        charge = float(np.mean([v[0] for v in values]))
        bins = np.mean([v[1] for v in values], axis=0)
        return charge, bins
    charge = quad(lambda phi: pencil(phi)[0], -np.pi, np.pi, points=[0.0],
                  epsrel=epsrel, limit=200)[0] / (2 * np.pi)
    bins = np.array([
        quad(lambda phi, k=k: pencil(phi)[1][k], -np.pi, np.pi, points=[0.0],
             epsrel=epsrel, limit=200)[0] / (2 * np.pi)
        for k in range(len(edges) - 1)])
    return charge, bins
