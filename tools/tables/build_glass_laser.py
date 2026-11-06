#!/usr/bin/env python3
"""Build the first-entry (glass) table of a monochromatic calibration flash.

The isotropic flash is a pure l = 0 source, so the glass table is the
directional first-entry table of the shower routes restricted to source
degree 0: ``t_(0 lambda 0)(r, omega)`` for lambda up to the acceptance degree,
orders 1 and >=2 kept apart. The receiver numerics (vertex settings and
scattering degree) are taken from a glass data bundle's ``shower-data.json``,
so the flash sees the same glass as the shower and muon routes. Order 0 is
evaluated at run time and needs no table.

The frequency blocks are built in worker processes and checkpointed: repeat
the same command to resume. The output directory receives

* ``directional-<key>.npz``  the table under the key the laser kernel asks for;
* ``laser-data.json``        the kernel settings that select it (read by
  ``lighthit.event.laser_config``);
* ``cache-manifest.json``    the usual cache manifest;
* ``build.json``             wall time, workers and the receiver contract.

Example (repository root)::

    PYTHONPATH=src python tools/tables/build_glass_laser.py \\
        --model DATA/model/bgvd-model.npz --settings DATA/tables/shower-data.json \\
        --output OUT/laser-glass-532 --wavelength 532 --jobs 3
"""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import argparse
import json
import shutil
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _atomic import write_json  # noqa: E402


def receiver_settings(path):
    """Scattering degree and vertex numerics of a glass bundle manifest."""
    settings = json.loads(Path(path).read_text(encoding="utf-8")).get("kernel_settings") or {}
    if settings.get("scattered_receiver") != "first-entry":
        raise SystemExit(f"{path} declares no first-entry receiver")
    return dict(scattered_receiver="first-entry",
                scattering_degree=int(settings["scattering_degree"]),
                source_degree=0,
                receiver_vertex=dict(settings["receiver_vertex"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True,
                        help="shower-data.json of a glass bundle (receiver numerics)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wavelength", type=float, default=532.0)
    parser.add_argument("--frequencies", type=int, default=321,
                        help="table frequencies on [0, 1.2] /ns (laser default 321)")
    parser.add_argument("--bin-ns", type=float, default=5.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--frequency-block", type=int, default=8)
    parser.add_argument("--keep-blocks", action="store_true")
    args = parser.parse_args(argv)

    import lighthit as lh
    from lighthit.cache import CacheGrid
    from lighthit.event import LASER_DATA, LASER_DATA_SCHEMA, laser_config
    from lighthit.receiver import SphereSurface
    from lighthit.receiver_bank import prepare_receiver_bank
    from lighthit.transport import CacheBuildEntry, CacheBuildReport

    began = perf_counter()
    model = lh.load_model(args.model)
    settings = receiver_settings(args.settings)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    omega = np.linspace(0.0, 1.2, args.frequencies)
    config = laser_config(model.detector, cache_directory=output, bin_ns=args.bin_ns,
                          cache_policy="require", omega_per_ns=omega, kernel_settings=settings)
    kernel = lh.TransportKernel(model.medium, model.detector, config)
    wavelength = float(args.wavelength)
    target = kernel._table_path("directional", kernel._directional_cache_key(wavelength))
    contract = kernel._receiver_contract()
    bank = output / ".bank"
    if not target.is_file():
        low, high = config.radial_range_m
        grid = CacheGrid.geometric(low, high, config.radial_nodes, config.omega_per_ns)

        def progress(done, total):
            print(json.dumps(dict(blocks=done, total=total,
                                  seconds=round(perf_counter() - began, 1))), flush=True)

        prepare_receiver_bank([kernel.medium.band(wavelength)], kernel._solver_settings(), grid,
                              SphereSurface(contract["radius_m"]), kernel.acceptance()["degree"],
                              bank, config=config.receiver_vertex, jobs=args.jobs,
                              frequency_block=args.frequency_block, progress=progress)
        (bank / "directional-band-000.npz").replace(target)
        seconds = perf_counter() - began
        write_json(output / "build.json", dict(
            schema="lighthit/laser-glass-build/1", wavelength_nm=wavelength,
            table=target.name, wall_seconds=seconds, jobs=args.jobs,
            frequency_block=args.frequency_block, frequencies=len(omega),
            radial_range_m=list(config.radial_range_m), radial_nodes=config.radial_nodes,
            solver_settings=asdict(kernel._solver_settings()),
            acceptance_degree=kernel.acceptance()["degree"], receiver_contract=contract,
            package_version=lh.__version__))
        if not args.keep_blocks:
            shutil.rmtree(bank)
    write_json(output / LASER_DATA, dict(
        schema=LASER_DATA_SCHEMA, wavelength_nm=[wavelength],
        kernel_settings=settings,
        note="isotropic flash: scattered light at the first entry into the glass sphere"))
    # The table must now be found under the settings the manifest declares.
    check = lh.TransportKernel(model.medium, model.detector,
                               laser_config(model.detector, cache_directory=output,
                                            bin_ns=args.bin_ns, omega_per_ns=omega))
    cache = check._directional_cache_for(wavelength)
    if cache.degree != 0 or not check._receiver_matches(cache.receiver_contract):
        raise SystemExit("the written table does not match the declared laser settings")
    check._write_cache_manifest(
        method="isotropic", kinds=("directional",), wavelengths=[wavelength],
        report=CacheBuildReport((CacheBuildEntry("directional", wavelength, "built", str(target),
                                                 perf_counter() - began),)))
    print(json.dumps(dict(finished=True, table=str(target),
                          seconds=round(perf_counter() - began, 1))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
