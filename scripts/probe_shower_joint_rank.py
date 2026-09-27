#!/usr/bin/env python3
"""Small source-only test of the shower's angle/time or angle/radius rank.

The angular variable is the emitted photon's cosine to the shower axis, not
the charged track's direction. Two chord nodes and an exact Cherenkov-cone
azimuth integral deposit photon mass into (z, cosine, second-variable) bins. A basis
is fitted on training events only and evaluated on held-out events. No RTE
cache or detector response is used, so these errors are not PE errors.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from lighthit.experimental.axial_source import AxisFrame
from lighthit.experimental.g4_source import SourceContract, load_event

C_M_PER_NS = 0.299792458


def histogram(path, event, z_edges, mu_edges, other_edges, *,
              secondary="time", chunk_size=2048):
    elements = load_event(path, event, SourceContract())
    frame = AxisFrame.of(elements)
    along = elements.direction @ frame.axis
    transverse = np.sqrt(np.maximum(1 - along ** 2, 0.0))
    cone = elements.cone_cosine
    cone_sine = np.sqrt(np.maximum(1 - cone ** 2, 0.0))
    angular_offset = cone_sine * transverse
    nodes, node_weights = np.polynomial.legendre.leggauss(2)
    nodes, node_weights = (nodes + 1) / 2, node_weights / 2
    z_mid = (elements.midpoints_m - frame.centre_m) @ frame.axis
    tau_mid = elements.midpoint_time_ns() - z_mid / C_M_PER_NS
    tau_reference = np.average(tau_mid, weights=elements.photons)
    shape = (len(z_edges) - 1, len(mu_edges) - 1, len(other_edges) - 1)
    out = np.zeros(shape, float)
    angular_bins = np.arange(shape[1])
    for fraction, node_weight in zip(nodes, node_weights, strict=True):
        points = (elements.start_m
                  + (fraction * elements.length_m)[:, None] * elements.direction)
        z = (points - frame.centre_m) @ frame.axis
        time = elements.start_ns + fraction * (elements.end_ns - elements.start_ns)
        if secondary == "time":
            other = time - z / C_M_PER_NS - tau_reference
        elif secondary == "radius":
            offset = points - frame.centre_m - z[:, None] * frame.axis
            other = np.linalg.norm(offset, axis=1)
        else:
            raise ValueError("secondary must be time or radius")
        weights = elements.photons * node_weight
        zi = np.searchsorted(z_edges, z, side="right") - 1
        ti = np.searchsorted(other_edges, other, side="right") - 1
        for begin in range(0, len(elements), chunk_size):
            sl = slice(begin, begin + chunk_size)
            centre = (cone[sl] * along[sl])[:, None]
            spread = angular_offset[sl, None]
            safe = np.maximum(spread, 1e-15)
            low = np.clip((mu_edges[:-1][None, :] - centre) / safe, -1, 1)
            high = np.clip((mu_edges[1:][None, :] - centre) / safe, -1, 1)
            probability = (np.arccos(low) - np.arccos(high)) / np.pi
            degenerate = np.flatnonzero(spread[:, 0] < 1e-15)
            if len(degenerate):
                probability[degenerate] = 0.0
                slot = np.clip(np.searchsorted(mu_edges, centre[degenerate, 0],
                                               side="right") - 1, 0, shape[1] - 1)
                probability[degenerate, slot] = 1.0
            labels = ((zi[sl, None] * shape[1] + angular_bins[None, :])
                      * shape[2] + ti[sl, None])
            out += np.bincount(
                labels.ravel(), weights=(weights[sl, None] * probability).ravel(),
                minlength=out.size).reshape(shape)
    if not np.isclose(out.sum(), elements.photons.sum(), rtol=1e-10):
        raise RuntimeError("joint histogram lost emitted light")
    return out


def rows(histograms):
    values, masses, owner = [], [], []
    for event, histogram in histograms.items():
        layer_mass = histogram.sum(axis=(1, 2))
        active = layer_mass > 0
        values.extend(histogram[active] / layer_mass[active, None, None])
        masses.extend(layer_mass[active] / layer_mass.sum())
        owner.extend([event] * int(active.sum()))
    return np.asarray(values), np.asarray(masses), np.asarray(owner)


def product_and_residual(probability):
    angle = probability.sum(axis=2)
    time = probability.sum(axis=1)
    product = angle[:, :, None] * time[:, None, :]
    return product, probability - product


def event_errors(probability, mass, basis, ranks):
    product, residual = product_and_residual(probability)
    flat = residual.reshape(len(residual), -1)
    total = mass.sum()
    result = {}
    for rank in ranks:
        correction = (flat @ basis[:rank].T) @ basis[:rank]
        estimate = product + correction.reshape(product.shape)
        result[str(rank)] = {
            "joint_l1": float(np.sum(mass[:, None, None]
                                     * np.abs(probability - estimate)) / total),
            "negative_mass": float(np.sum(mass[:, None, None]
                                           * np.maximum(-estimate, 0)) / total),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--test", type=int, nargs="+", default=list(range(5, 10)))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--secondary", choices=("time", "radius"), default="time")
    args = parser.parse_args()
    if set(args.train) & set(args.test):
        parser.error("training and test event sets must be disjoint")
    z_edges = np.r_[-np.inf, np.arange(-6.0, 9.0, 0.5), np.inf]
    mu_edges = np.unique(np.r_[np.linspace(-1, 1, 33),
                                np.linspace(0.55, 0.90, 65)])
    tau_edges = np.r_[-np.inf, np.linspace(-1, 1, 41), 1.5, 2., 3., 5., np.inf]
    radius_edges = np.unique(np.r_[
        0., np.geomspace(0.001, 0.5, 33), 0.6, 0.8, 1., 1.5, 2., 3., np.inf])
    other_edges = tau_edges if args.secondary == "time" else radius_edges
    histograms = {}
    for event in args.train + args.test:
        histograms[event] = histogram(
            args.input, event, z_edges, mu_edges, other_edges,
            secondary=args.secondary)
        print(f"event {event}: {histograms[event].sum():.0f} photons", flush=True)
    training, weight, _ = rows({e: histograms[e] for e in args.train})
    _, residual = product_and_residual(training)
    # Equal event totals; within an event, each layer carries its photon share.
    weighted = residual.reshape(len(residual), -1) * np.sqrt(weight[:, None])
    gram = weighted @ weighted.T
    eigenvalue, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalue)[::-1]
    singular = np.sqrt(np.maximum(eigenvalue[order], 0))
    valid = singular > max(singular[0], 1e-300) * 1e-12
    basis = (vectors[:, order[valid]].T @ weighted) / singular[valid, None]
    ranks = [0, 1, 2, 4, 8, 16]
    if len(basis) < max(ranks):
        ranks = [r for r in ranks if r <= len(basis)]
    report = {"schema": "lighthit/shower-joint-rank-probe/1",
              "input_name": args.input.name, "train_events": args.train,
              "test_events": args.test, "cone_azimuth": "exact arcsine bin masses",
              "secondary": args.secondary,
              "shape": [len(z_edges)-1, len(mu_edges)-1, len(other_edges)-1],
              "basis_singular_values": singular[:20].tolist(),
              "note": "source-only weighted L1; not a detector-response error",
              "held_out": {}}
    for event in args.test:
        probability, mass, _ = rows({event: histograms[event]})
        report["held_out"][str(event)] = event_errors(
            probability, mass, basis, ranks)
        print(f"held-out {event}:", report["held_out"][str(event)], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("written:", args.output, flush=True)


if __name__ == "__main__":
    main()
