"""Small installed-package check requiring no repository or private input."""
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
import argparse

import numpy as np

import lighthit as lh


def run(output=None, *, viewer=False):
    temporary = (TemporaryDirectory(prefix="lighthit-selftest-")
                 if output is None and not viewer else None)
    location = (temporary.name if temporary is not None else
                (mkdtemp(prefix="lighthit-selftest-") if output is None else output))
    root = Path(location).resolve()
    root.mkdir(parents=True, exist_ok=True)
    medium = lh.SpectralMedium(
        [440., 460.], [.03, .03], [.02, .02], [1.34, 1.34], [1.37, 1.37])
    detector = lh.DetectorArray(
        [[10., 0., 0.], [20., 0., 0.]], [[-1., 0., 0.], [-1., 0., 0.]], .05,
        lambda x: np.ones_like(np.asarray(x, float)),
        lambda w: np.full_like(np.asarray(w, float), .2))
    source = lh.IsotropicFlash.monochromatic([0., 0., 0.], 1e6, 450.)
    config = lh.KernelConfig(
        omega_per_ns=np.array([0., .04]),
        relative_time_edges_ns=np.arange(-10., 61., 5.),
        wavelength_nodes=2, scattering_degree=4, source_degree=4,
        k_max_per_m=2., k_panel_per_m=.2, k_order=4,
        radial_range_m=(2., 30.), radial_nodes=8,
        angular_backend="numpy", threshold_pe=0,
        cache_directory=root / "cache")
    kernel = lh.build(medium, detector, source, config=config)
    response = kernel.transport(source)
    assert response.components_pe.shape == (2, 14, 3)
    assert np.isfinite(response.components_pe).all()
    assert np.all(response.charge_pe > 0)
    assert np.all(response.components_pe[:, :, 1] >= 0)
    if viewer:
        payload = lh.viewer_payload(response, source=source, event_id="selftest")
        lh.write_event_viewer(payload, root / "viewer.html")
    print(f"LightHit {lh.__version__}: self-test passed")
    if output is not None or viewer:
        print("output:", root)
    if temporary is not None and not viewer:
        temporary.cleanup()
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--viewer", action="store_true")
    arguments = parser.parse_args(argv)
    run(arguments.output, viewer=arguments.viewer)


if __name__ == "__main__":
    main()
