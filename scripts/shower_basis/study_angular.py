"""Angular compression of the slice coefficients, judged in photoelectrons.

Trains on G4 events 0-4 and tests on 5-9 (``--kind e100`` or ``e1000``).
For every test event the order >=2 response of the full angular content
(``l <= 32``, ``|m| <= 4``: 277 channels per point) is compared with:

* POD in harmonic space (mean + N-1 centred modes of the per-slice photon
  angular distribution, mass-weighted, fitted on the training events);
* POD of the azimuth-averaged distribution in ``mu`` only (m = 0 channels),
  keeping the m != 0 channels exact -- the representation of
  ``baikal_rte.angular_pod``;
* local positive functions: every emission direction snapped to the
  photon-weighted mean direction of its (cos theta, phi) bin in the axial
  frame;
* truncated azimuthal band M = 0, 1, 2.

Errors are order >=2 charge and L1 of the order >=2 time bins, on the OMs
whose exact order >=2 charge exceeds ``--min-pe``; the source-only L2 error
is printed next to it for contrast.  Output: JSON lines on stdout.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from proto import (apply_ge2, basis_sources, metrics, model, config, origins_of, readout,
                   shower)
from dataclasses import replace


def slice_vectors(source):
    """Axis-point channel vectors at omega = 0 and their masses."""
    axis = np.all(np.abs(source.transverse_m) < 1e-12, axis=1)
    vectors = source.channels[axis, :, 0].real
    return vectors, vectors[:, 0]


def fit_pod(training, n_max, channel_mask):
    rows, masses = [], []
    for vectors, mass in training:
        keep = mass > 1e-9 * mass.max()
        rows.append(vectors[keep][:, channel_mask] / mass[keep, None])
        masses.append(mass[keep] / mass[keep].sum())
    rows = np.concatenate(rows)
    weight = np.concatenate(masses)
    weight /= weight.sum()
    mean = weight @ rows
    centred = (rows - mean) * np.sqrt(weight)[:, None]
    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    # orthonormal basis of span{mean, first n_max-1 centred modes}; the order
    # is kept, so basis[:n] spans {mean, modes 1..n-1} for every n
    stacked = np.vstack([mean, right[:n_max - 1]]).T
    q, r = np.linalg.qr(stacked)
    q *= np.sign(np.diag(r))[None, :]
    return q.T, singular


def project(source, basis, channel_mask):
    """Replace the masked channels of every point and frequency by their projection."""
    channels = source.channels.copy()
    block = channels[:, channel_mask, :]
    coefficient = np.einsum("jc,pcw->pjw", basis, block)
    channels[:, channel_mask, :] = np.einsum("jc,pjw->pcw", basis, coefficient)
    return replace(source, channels=channels)


def source_l2(source, reference):
    return float(np.linalg.norm(source.channels[:, :, 0] - reference.channels[:, :, 0])
                 / np.linalg.norm(reference.channels[:, :, 0]))


def snapped(bins_mu, bins_phi, frame):
    def directions(field):
        local = frame.rotate(field.direction)
        mu = np.clip(local[:, 2], -1, 1)
        phi = np.arctan2(local[:, 1], local[:, 0])
        index = (np.minimum(((mu + 1) / 2 * bins_mu).astype(int), bins_mu - 1) * bins_phi
                 + np.minimum(((phi + np.pi) / (2 * np.pi) * bins_phi).astype(int), bins_phi - 1))
        weight = np.abs(np.asarray(field.photons, float))
        mean = np.zeros((bins_mu * bins_phi, 3))
        np.add.at(mean, index, weight[:, None] * local)
        mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-300)
        world = (mean[index, 0:1] * frame.first + mean[index, 1:2] * frame.second
                 + mean[index, 2:3] * frame.axis)
        return world / np.linalg.norm(world, axis=1, keepdims=True)
    return directions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", default="e100")
    parser.add_argument("--train", default="0,1,2,3,4")
    parser.add_argument("--test", default="5,6,7,8,9")
    parser.add_argument("--ranks", default="1,4,6,8,10,16,32")
    parser.add_argument("--min-pe", type=float, default=0.002)
    parser.add_argument("--only", default="pod,bins,band")
    parser.add_argument("--transverse", default="collapse",
                        help="basis the compressions are applied to (reference: same, uncompressed)")
    args = parser.parse_args()
    k = model().kernel(config())
    degree = k.config.source_degree
    from lighthit.experimental.axial_fast import _layout
    _, _, _, kept, channel_degree = _layout(degree, k.config.azimuthal_degree)
    kept = np.asarray(kept)
    m_of = kept - channel_degree * channel_degree - channel_degree
    all_channels = np.ones(len(kept), bool)
    m0_channels = m_of == 0
    training = []
    for event in [int(e) for e in args.train.split(",")]:
        s0, _, _, _ = basis_sources(k, shower(args.kind, event), omega=np.zeros(1),
                                    transverse=args.transverse)
        training.append(slice_vectors(s0))
    ranks = [int(r) for r in args.ranks.split(",")]
    pods = {"harmonic": fit_pod(training, max(ranks), all_channels),
            "mu_m0": fit_pod(training, min(max(ranks), degree + 1), m0_channels)}
    print(json.dumps({"singular_harmonic": pods["harmonic"][1][:20].tolist(),
                      "singular_mu": pods["mu_m0"][1][:20].tolist()}), flush=True)
    for event in [int(e) for e in args.test.split(",")]:
        source = shower(args.kind, event)
        started = time.time()
        s0, s2, el, sample = basis_sources(k, source, transverse=args.transverse)
        # targets: OMs with a noticeable order >=2 charge, found at omega = 0
        zero = apply_ge2_zero(k, s0, s2, sample)
        targets = np.flatnonzero(zero.real >= args.min_pe)
        origins = origins_of(k, el, sample)[targets]
        spectrum = apply_ge2(k, s0, s2, targets, sample)
        q_ref, b_ref = spectrum[0].real, readout(k, spectrum, origins)
        print(json.dumps({"event": event, "targets": len(targets), "points": len(s0.z_m),
                          "seconds": time.time() - started}), flush=True)

        def report(label, a0, a2, **extra):
            spec = apply_ge2(k, a0, a2, targets, sample)
            out = metrics(spec[0].real, readout(k, spec, origins), q_ref, b_ref)
            bright = np.argsort(-q_ref)[:10]
            out["ge2_rel_max_bright"] = float(np.max(np.abs(spec[0].real - q_ref)[bright]
                                                     / q_ref[bright]))
            print(json.dumps({"event": event, "variant": label, **out, **extra}), flush=True)

        own = fit_pod([slice_vectors(s0)], max(ranks), all_channels)[0]
        for rank in (ranks if "pod" in args.only else []):
            report("pod_in_sample", project(s0, own[:rank], all_channels),
                   project(s2, own[:rank], all_channels), rank=rank,
                   source_l2=source_l2(project(s0, own[:rank], all_channels), s0))
        for name, mask in (() if "pod" not in args.only else
                           (("harmonic", all_channels), ("mu_m0", m0_channels))):
            basis = pods[name][0]
            for rank in ranks:
                if rank > basis.shape[0]:
                    continue
                a0 = project(s0, basis[:rank], mask)
                a2 = project(s2, basis[:rank], mask)
                report(f"pod_{name}", a0, a2, rank=rank, source_l2=source_l2(a0, s0))
        for bins in (((8, 4), (16, 8), (32, 16), (64, 32)) if "bins" in args.only else ()):
            frame = s0.frame
            b0, b2, _, _ = basis_sources(k, source, directions=snapped(*bins, frame),
                                         transverse=args.transverse)
            report("positive_bins", b0, b2, bins=list(bins), source_l2=source_l2(b0, s0))
        for m in ((0, 1, 2) if "band" in args.only else ()):
            a0, a2, _, _ = basis_sources(k, source, M=m, transverse=args.transverse)
            # pad the truncated band back into the M=4 layout for the same kernel
            report("azimuthal_band", pad(a0, s0), pad(a2, s2), M=m,
                   source_l2=source_l2(pad(a0, s0), s0))


def apply_ge2_zero(k, s0, s2, sample):
    targets = np.arange(len(k.detector.positions_m))
    z0 = replace(s0, channels=np.ascontiguousarray(s0.channels[:, :, :1]))
    z2 = replace(s2, channels=np.ascontiguousarray(s2.channels[:, :, :1]))
    store = k.ge2_kernel_store()
    pos = k.detector.positions_m
    looks = np.ascontiguousarray(-k.detector.orientations, float)
    area = np.asarray(k.detector.effective_area_m2, float)
    alpha = np.ascontiguousarray(k.acceptance()["alpha"], float)
    idx = np.arange(len(k.config.omega_per_ns))
    out = np.zeros(len(targets))
    for w, wt, ph in zip(k.wavelength.wavelength_nm, k.wavelength.weight_nm, sample["phase_index"]):
        prepared = store.kernel(k, w, idx, frequencies=slice(0, 1))
        r = prepared.apply(z0, pos, looks, alpha, source_omega_per_ns=k.config.omega_per_ns[:1],
                           receiver_block=4, second_source=z2, first_scale=1 / w ** 2,
                           second_scale=1 / (w ** 2 * ph ** 2), ge2_only=True)
        out += wt * area * float(k.detector.spectral_weight(float(w))) * r[0, :, 1].real
    return out


def pad(small, full):
    """Embed a smaller azimuthal band into the channel layout of ``full``."""
    channels = np.zeros_like(full.channels)
    index = {int(c): i for i, c in enumerate(full.kept)}
    for i, c in enumerate(small.kept):
        channels[:, index[int(c)], :] = small.channels[:, i, :]
    return replace(full, channels=channels)


if __name__ == "__main__":
    main()
