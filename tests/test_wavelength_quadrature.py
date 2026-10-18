"""The default wavelength quadrature resolves a narrow transparency window.

Scattered orders integrate ``exp(-mu(lambda) r)`` over the detected Cherenkov
spectrum. With a piecewise-linear absorption that has a narrow minimum (as the
Baikal water has near 496 nm) nine Gauss-Legendre nodes miss the window and
lose 6 % of the light at 50 m and 20 % at 100 m; the photon Monte Carlo of
NTSim, which samples the continuous spectrum, showed that deficit in orders 1
and >=2. The medium below is synthetic but has the same features.
"""
import numpy as np
import pytest

import lighthit as lh
from lighthit.prompt import _spectral_tables
from lighthit.shower_basis import Ge2KernelStore

WAVELENGTH = [350., 380., 420., 460., 490., 500., 530., 560., 580., 610.]
ABSORPTION = [.30, .22, .12, .07, .045, .043, .055, .075, .11, .17]


def kernel(**options):
    medium = lh.SpectralMedium(WAVELENGTH, ABSORPTION, np.linspace(.035, .015, 10),
                               np.linspace(1.35, 1.334, 10), np.linspace(1.40, 1.35, 10), g=.9)

    def efficiency(w):
        return np.interp(w, [350., 375., 395., 420., 500., 560., 610.],
                         [.001, .01, .3, .32, .22, .1, .04])
    detector = lh.DetectorArray([[0., 0., 0.]], [[0., 0., 1.]], .15,
                                lambda x: np.ones_like(x), efficiency)
    return lh.TransportKernel(medium, detector,
                              lh.KernelConfig(omega_per_ns=np.array([0.]), **options))


def attenuated_light(k, radius_m):
    """Detected Cherenkov light (beta = 1 field) after ``radius_m`` of attenuation."""
    tables = _spectral_tables(k)
    return np.sum(tables["s0w"] * np.exp(-tables["mut"] * radius_m))


def reference_light(k, radius_m):
    wavelength = np.linspace(350., 610., 200001)
    band = k.medium.sample(wavelength)
    value = (k.detector.spectral_weight(wavelength) / wavelength ** 2
             * np.exp(-(band["absorption_per_m"] + band["scattering_per_m"]) * radius_m))
    return np.trapezoid(value, wavelength)


@pytest.mark.parametrize("radius_m", [0., 20., 50., 100.])
def test_default_quadrature_keeps_attenuated_spectrum_within_one_percent(radius_m):
    k = kernel()
    assert len(k.wavelength.wavelength_nm) == lh.KernelConfig().wavelength_nodes == 32
    np.testing.assert_allclose(attenuated_light(k, radius_m), reference_light(k, radius_m),
                               rtol=1e-2)


def test_nine_nodes_miss_the_transparency_window():
    k = kernel(wavelength_nodes=9)
    assert attenuated_light(k, 100.) < 0.85 * reference_light(k, 100.)


def test_pack_of_another_quadrature_is_refused_by_name(tmp_path):
    k = kernel(cache_directory=tmp_path, cache_policy="require")
    store = Ge2KernelStore()
    store._bundles[(tuple(f"table-{i}" for i in range(9)), int(k.config.source_degree), (0,))] = {}
    with pytest.raises(ValueError, match="holds \\[9\\] wavelength nodes, the kernel uses 32"):
        store.bundle(k, [0])


def test_muon_tables_of_another_quadrature_are_refused_by_name():
    from types import SimpleNamespace
    from lighthit.muon.preview import check_atlas_optics
    atlas = SimpleNamespace(metadata={"tables": {"wavelength_nm": list(np.linspace(355., 605., 9))}})
    with pytest.raises(ValueError, match="hold 9 wavelength nodes, the kernel uses 32"):
        check_atlas_optics(atlas, kernel())
