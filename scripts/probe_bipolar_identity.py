#!/usr/bin/env python3
"""The four identities the directional kernel rests on, measured not asserted.

1. the parity rule ``l + lambda + J`` even, as a property of the folded
   Clebsch-Gordan coefficient rather than a filter applied to it;
2. the bipolar factorisation against a direct numerical integral over the
   direction of ``k``;
3. that integral against a dense lab-frame solve that knows nothing about
   azimuthal blocks;
4. the free tail ratios for ``m != 0`` against a deliberately deeper continued
   fraction, across the whole range of decay rates.

Numbers printed here are the ones quoted in
``docs/research/directional-om-plan.md`` and in @sec-directional-om.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lighthit.angular import (free_tail_ratios_m,                 # noqa: E402
                              tail_ratios_continued_fraction)
from lighthit.cache import CacheGrid                              # noqa: E402
from lighthit.directional import (CouplingTable, DirectionalCache,  # noqa: E402
                                  clebsch_gordan, directional_response)
from lighthit.green import SolverSettings                         # noqa: E402
from lighthit.medium import Medium                                # noqa: E402
from lighthit.experimental import directional_reference as ref    # noqa: E402
from lighthit.experimental.axial_source import (AxialSource,       # noqa: E402
                                                AxisFrame, channel_index)
from lighthit.experimental.event_moments import real_spherical_harmonics  # noqa: E402


def parity_rule(source_degree=8, acceptance_degree=3):
    """The folded nu-sum vanishes exactly where the parity rule says it does."""
    worst_odd = 0.0
    kept_even = 0
    for lam in range(acceptance_degree + 1):
        for l in range(source_degree + 1):
            for J in range(abs(l - lam), l + lam + 1):
                folded = sum((-1.0) ** nu * clebsch_gordan(lam, nu, l, -nu, J, 0)
                             for nu in range(-min(lam, l), min(lam, l) + 1))
                if (l + lam + J) % 2:
                    worst_odd = max(worst_odd, abs(folded))
                elif abs(folded) > 1e-14:
                    kept_even += 1
    table = CouplingTable.build(source_degree, acceptance_degree)
    return {"largest_odd_parity_coefficient": worst_odd,
            "nonzero_even_parity_triples": kept_even,
            "stored_triples": table.triples, "stored_pairs": table.pairs,
            "triples_per_degree": table.triples / (source_degree + 1)}


def factorisation(degree=6, acceptance_degree=3, polar_order=30):
    """Production cache and apply, against the direct k-hat quadrature."""
    medium = Medium(0.02, 0.05, 0.9, 1.35, 450.0, "probe medium")
    settings = SolverSettings(degree, degree, 2.0, 0.1, 6, True)
    omega = np.array([0.0, 0.07])
    radii = np.geomspace(4.0, 40.0, 12)
    cache = DirectionalCache.build(medium, settings, CacheGrid(radii, omega),
                                   acceptance_degree, radial_phase="none")
    rng = np.random.default_rng(17)
    moments = rng.normal(size=(degree + 1) ** 2) * 0.4
    moments[0] = 1.0
    kept, channel_degree = channel_index(degree, degree)
    table = np.zeros((1, len(kept), len(omega)), complex)
    table[0, :, 0] = moments[kept]
    table[0, :, 1] = moments[kept]
    frame = AxisFrame(np.zeros(3), np.array([0.0, 0.0, 1.0]),
                      np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    source = AxialSource(frame, np.zeros(1), np.zeros((1, 2)), table, kept,
                         channel_degree, degree, degree, 0.0, {})
    direction = np.array([0.4, -0.3, 0.866])
    direction /= np.linalg.norm(direction)
    displacement = direction * radii[5]
    look = np.array([0.2, 0.9, -0.39])
    look /= np.linalg.norm(look)
    alpha = np.array([4 * np.pi, 1.3, -0.5, 0.2])[:acceptance_degree + 1]
    produced = directional_response(cache, source, displacement[None, :],
                                    look[None, :], alpha,
                                    source_omega_per_ns=omega)
    out = {}
    for index, frequency in enumerate(omega):
        want = ref.khat_quadrature_response(
            medium, settings, frequency, moments, displacement, look, alpha,
            acceptance_degree=acceptance_degree, polar_order=polar_order)
        out[f"omega={frequency:g}"] = float(
            np.max(np.abs(produced[index, 0] - want)) / np.max(np.abs(want)))
    return out


def dense_cross_check(polar_order=24):
    """Two references that share nothing but the physics."""
    strong = Medium(0.30, 0.06, 0.9, 1.35, 450.0, "strongly attenuating")
    rng = np.random.default_rng(17)
    base = rng.normal(size=25) * 0.4
    base[0] = 1.0
    displacement = np.array([1.7, -0.9, 2.2])
    look = rng.normal(size=3)
    look /= np.linalg.norm(look)
    alpha = np.array([4 * np.pi, 1.3, -0.5, 0.2])
    out = {}
    for degree in (4, 8, 12):
        settings = SolverSettings(4, max(degree, 1), 0.4, 0.05, 6, True)
        moments = np.zeros((degree + 1) ** 2)
        moments[:25] = base
        for omega in (0.0, 0.08):
            blocks = ref.khat_quadrature_response(
                strong, settings, omega, moments, displacement, look, alpha,
                acceptance_degree=3, polar_order=polar_order)
            dense = ref.dense_angular_reference(
                strong, settings, omega, moments, displacement, look, alpha,
                acceptance_degree=3, polar_order=polar_order,
                quadrature_order=40)
            out[f"degree={degree},omega={omega:g}"] = float(
                np.max(np.abs(blocks - dense)) / np.max(np.abs(dense)))
    return out


def free_tail(depth=40000, degree=30):
    medium = Medium(0.02, 0.05, 0.9, 1.35, 450.0, "probe medium")
    k = np.concatenate([[0.0], np.geomspace(0.02, 8.0, 40)])
    out = {}
    for omega in (0.0, 0.15):
        d0 = medium.extinction_per_m - 1j * omega / medium.speed_m_per_ns
        report = {}
        ratios = free_tail_ratios_m(k, d0, degree, 3, report=report)
        row = {"fallback_nodes": report["tail_raise_fallback_nodes"]}
        for m in (1, 2, 3):
            deep = tail_ratios_continued_fraction(k, d0, degree, m, depth=depth)
            window = slice(m + 1, degree)
            relative = (np.abs(ratios[:, m, window] - deep[:, window])
                        / np.maximum(np.abs(deep[:, window]), 1e-300))
            row[f"m={m}"] = float(np.nanmax(relative))
        out[f"omega={omega:g}"] = row
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=".build/review/directional")
    arguments = parser.parse_args()
    results = {"parity": parity_rule(), "factorisation": factorisation(),
               "dense_cross_check": dense_cross_check(), "free_tail": free_tail()}
    print(json.dumps(results, indent=2))
    path = Path(arguments.output)
    path.mkdir(parents=True, exist_ok=True)
    (path / "bipolar-identity.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {path / 'bipolar-identity.json'}")


if __name__ == "__main__":
    main()
