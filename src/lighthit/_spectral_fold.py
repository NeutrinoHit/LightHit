"""Prepared wavelength-integrated directional kernels for Cherenkov sources.

The source spectrum has two wavelength-independent fields.  Folding the
wavelength quadrature into their Green kernels is exact on the stored radial
nodes.  A denser shared radial spline represents their sum between nodes; it
is a separately versioned approximation and never replaces the original
wavelength caches.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
import json
import shutil

import numpy as np
from scipy.interpolate import CubicSpline

from ._directional_numba import PreparedDirectionalKernel
from .directional import CouplingTable


FORMAT = "lighthit/spectral-folded-directional/1"


def _profile(kernel):
    if kernel.config.cache_directory is None:
        raise ValueError("a cache directory is required for spectral folding")
    wavelengths = np.asarray(kernel.wavelength.wavelength_nm, float)
    weights = np.asarray(kernel.wavelength.weight_nm, float)
    sample = kernel.medium.sample(wavelengths)
    detection = np.asarray(kernel.detector.spectral_weight(wavelengths), float)
    if detection.shape != wavelengths.shape or not np.isfinite(detection).all():
        raise ValueError("detector spectral efficiency must match wavelength nodes")
    raw_keys = [kernel._directional_cache_key(float(w)) for w in wavelengths]
    reference_wavelength = float(np.clip(500., wavelengths[0], wavelengths[-1]))
    reference = kernel.medium.band(reference_wavelength)
    payload = {
        "format": FORMAT,
        "raw_keys": raw_keys,
        "wavelength_nm": wavelengths.tolist(),
        "quadrature_weight_nm": weights.tolist(),
        "detector_spectral_efficiency": detection.tolist(),
        "phase_index": np.asarray(sample["phase_index"], float).tolist(),
        "reference_wavelength_nm": reference_wavelength,
        "reference_absorption_per_m": reference.absorption_per_m,
        "reference_speed_m_per_ns": reference.speed_m_per_ns,
        "radial_midpoint_start_m": 100.,
    }
    signature = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    return Path(kernel.config.cache_directory).expanduser().resolve(), payload, signature


def folded_cache_path(kernel):
    directory, _, signature = _profile(kernel)
    return directory / f"directional-folded-{signature}"


def _original_files(directory, keys):
    files = [directory / f"directional-{key}.npz" for key in keys]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "build the directional wavelength caches before spectral folding: "
            + ", ".join(missing))
    return files


def _same_grid(files):
    radii = omega = None
    degree = acceptance_degree = pairs = None
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            current_radii = np.asarray(archive["radii_m"], float)
            current_omega = np.asarray(archive["omega_per_ns"], float)
            current_degree = int(metadata["source_degree"])
            current_acceptance = int(metadata["acceptance_degree"])
            table = CouplingTable.build(current_degree, current_acceptance)
            current_pairs = len(table.pair_lambda)
        if radii is None:
            radii, omega = current_radii, current_omega
            degree, acceptance_degree, pairs = (
                current_degree, current_acceptance, current_pairs)
        elif (not np.array_equal(radii, current_radii)
              or not np.array_equal(omega, current_omega)
              or (degree, acceptance_degree, pairs) !=
              (current_degree, current_acceptance, current_pairs)):
            raise ValueError("directional wavelength caches must share one grid and angular layout")
    return radii, omega, degree, acceptance_degree, pairs


def _prepared_shape(radii, omega, degree, pairs):
    return (len(radii) - 1, degree + 1, pairs, 4, 2, len(omega))


def build_folded_cache(kernel, *, progress=True, frequency_block=32):
    """Prepare two Cherenkov fields and one beta=1 track kernel on disk.

    Original wavelength caches remain untouched.  The final directory is
    published only after all three prepared arrays and metadata are complete.
    """
    began = perf_counter()
    directory, profile, signature = _profile(kernel)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"directional-folded-{signature}"
    if (destination / "metadata.json").is_file():
        if progress:
            print(f"Spectral-folded cache ready: {destination}", flush=True)
        return destination
    if destination.exists():
        raise FileExistsError(f"incomplete folded cache directory: {destination}")
    if not isinstance(frequency_block, int) or frequency_block < 1:
        raise ValueError("frequency_block must be a positive integer")

    files = _original_files(directory, profile["raw_keys"])
    original_radii, omega, degree, acceptance_degree, pairs = _same_grid(files)
    midpoints = np.sqrt(original_radii[:-1] * original_radii[1:])
    radii = np.sort(np.r_[original_radii,
                           midpoints[original_radii[:-1] >= 100.]])
    log_original = np.log(original_radii)
    log_radii = np.log(radii)
    source_shape = (len(omega), len(radii), degree + 1, pairs, 2)
    prepared_shape = _prepared_shape(radii, omega, degree, pairs)
    temporary = Path(mkdtemp(prefix=f".folded-{signature}-", dir=directory))
    try:
        fields = [np.lib.format.open_memmap(
            temporary / f"raw_field{index}.npy", mode="w+", dtype=np.complex128,
            shape=source_shape) for index in (0, 2)]
        for field in fields:
            field[:] = 0

        wavelengths = np.asarray(profile["wavelength_nm"], float)
        weights = np.asarray(profile["quadrature_weight_nm"], float)
        detection = np.asarray(profile["detector_spectral_efficiency"], float)
        phase_index = np.asarray(profile["phase_index"], float)
        for index, path in enumerate(files):
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
                raw = archive["coefficients"]
            medium = metadata["medium"]
            absorption = float(medium["absorption_per_m"])
            velocity = .299792458 / float(medium["group_index"])
            original_scale = np.exp(-absorption * original_radii) / (
                4 * np.pi * original_radii ** 2)
            radial_scale = np.exp(-absorption * radii) / (4 * np.pi * radii ** 2)
            first_weight = weights[index] * detection[index] / wavelengths[index] ** 2
            second_weight = first_weight / phase_index[index] ** 2
            if progress:
                print(f"Fold wavelength {index + 1}/{len(files)}: {wavelengths[index]:.1f} nm",
                      flush=True)
            for begin in range(0, len(omega), frequency_block):
                end = min(begin + frequency_block, len(omega))
                frequency = omega[begin:end]
                values = raw[begin:end] / original_scale[None, :, None, None, None]
                values *= np.exp(-1j * frequency[:, None]
                                 * original_radii[None, :] / velocity)[:, :, None, None, None]
                interpolated = CubicSpline(log_original, values, axis=1)(log_radii)
                interpolated *= radial_scale[None, :, None, None, None]
                interpolated *= np.exp(1j * frequency[:, None]
                                       * radii[None, :] / velocity)[:, :, None, None, None]
                fields[0][begin:end] += first_weight * interpolated
                fields[1][begin:end] += second_weight * interpolated
            del raw
        for field in fields:
            field.flush()

        prepared = {name: np.lib.format.open_memmap(
            temporary / f"{name}.npy", mode="w+", dtype=np.complex128,
            shape=prepared_shape) for name in ("field0", "field2", "track_beta1")}
        reference_absorption = float(profile["reference_absorption_per_m"])
        reference_speed = float(profile["reference_speed_m_per_ns"])
        reference_scale = np.exp(-reference_absorption * radii) / (4 * np.pi * radii ** 2)
        for begin in range(0, len(omega), frequency_block):
            end = min(begin + frequency_block, len(omega))
            frequency = omega[begin:end]
            phase = np.exp(-1j * frequency[:, None] * radii[None, :]
                           / reference_speed)[:, :, None, None, None]
            coefficient_blocks = []
            for name, field in (("field0", fields[0]), ("field2", fields[1])):
                values = field[begin:end] / reference_scale[None, :, None, None, None]
                values *= phase
                cubic = CubicSpline(log_radii, values, axis=1)
                block = np.ascontiguousarray(np.transpose(
                    cubic.c, (1, 3, 4, 0, 5, 2)))
                prepared[name][..., begin:end] = block
                coefficient_blocks.append(block)
            prepared["track_beta1"][..., begin:end] = (
                coefficient_blocks[0] - coefficient_blocks[1])
            if progress:
                print(f"Prepare folded frequencies {end}/{len(omega)}", flush=True)
        for array in prepared.values():
            array.flush()
        del fields, prepared
        (temporary / "raw_field0.npy").unlink()
        (temporary / "raw_field2.npy").unlink()

        metadata = {**profile, "signature": signature,
                    "source_degree": degree,
                    "acceptance_degree": acceptance_degree,
                    "radii_m": radii.tolist(), "omega_per_ns": omega.tolist(),
                    "radial_phase": "flight",
                    "interpolation": "cubic log-radius; extra midpoints after 100 m",
                    "build_seconds": perf_counter() - began,
                    "scattered_orders": [1, 2]}
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        temporary.rename(destination)
        if progress:
            print(f"Spectral-folded cache built in {metadata['build_seconds']:.1f} s: "
                  f"{destination}", flush=True)
        return destination
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


@dataclass(frozen=True)
class FoldedKernels:
    field0: PreparedDirectionalKernel
    field2: PreparedDirectionalKernel
    track_beta1: PreparedDirectionalKernel
    path: Path

    def zero_frequency(self, item):
        prepared = getattr(self, item)
        return replace(prepared,
                       omega_per_ns=prepared.omega_per_ns[:1],
                       coefficients=prepared.coefficients[..., :1])


def load_folded_cache(kernel):
    directory, profile, signature = _profile(kernel)
    destination = directory / f"directional-folded-{signature}"
    metadata_path = destination / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"spectral-folded cache is missing: {destination}; "
            "run kernel.build_folded_directional() after kernel.build()")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("signature") != signature or metadata.get("format") != FORMAT:
        raise ValueError("spectral-folded cache metadata does not match this kernel")
    degree = int(metadata["source_degree"])
    acceptance_degree = int(metadata["acceptance_degree"])
    table = CouplingTable.build(degree, acceptance_degree)
    omega = np.asarray(metadata["omega_per_ns"], float)
    log_r = np.log(np.asarray(metadata["radii_m"], float))
    arrays = {}
    expected = _prepared_shape(np.exp(log_r), omega, degree, len(table.pair_lambda))
    for name in ("field0", "field2", "track_beta1"):
        array = np.load(destination / f"{name}.npy", mmap_mode="r")
        if array.shape != expected or array.dtype != np.complex128:
            raise ValueError(f"invalid spectral-folded {name} coefficients")
        arrays[name] = PreparedDirectionalKernel(
            degree, acceptance_degree,
            np.ascontiguousarray(table.pair_lambda),
            np.ascontiguousarray(table.pair_mu),
            omega, log_r, array,
            float(metadata["reference_absorption_per_m"]),
            float(metadata["reference_speed_m_per_ns"]), "flight")
    return FoldedKernels(arrays["field0"], arrays["field2"],
                         arrays["track_beta1"], destination)


def optional_folded_cache(kernel):
    mode = kernel.config.spectral_folded_cache
    if mode is False:
        return None
    if kernel.config.cache_directory is None and mode == "auto":
        return None
    try:
        return load_folded_cache(kernel)
    except FileNotFoundError:
        if mode is True:
            raise
        return None


__all__ = ["build_folded_cache", "load_folded_cache", "optional_folded_cache",
           "folded_cache_path"]
