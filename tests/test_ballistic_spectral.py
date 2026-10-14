"""Independent cone-root comparison, including spectral threshold/endpoints."""
from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip('numba')
from lighthit.sources import SpectralLightElements
from lighthit.experimental.ballistic_spectral import spectral_ballistic_charge
from lighthit.ballistic import ballistic_directional
from lighthit.model import SpectralMedium, WavelengthQuadrature
from lighthit.prompt import _spectral_tables, _ballistic_fields


def setup_kernel(positions, phase=(1.32, 1.35, 1.38)):
    wavelengths = np.array([380., 480., 580.])
    medium = SpectralMedium(wavelengths, np.array([.07, .05, .08]),
                           np.array([.03, .02, .01]), np.array(phase), np.ones(3)*1.4)
    positions = np.asarray(positions, float)
    orientation = np.tile([.3, -.4, np.sqrt(.75)], (len(positions), 1))
    detector = SimpleNamespace(positions_m=positions, orientations=orientation,
                               effective_area_m2=np.linspace(.03, .07, len(positions)),
                               spectral_weight=lambda w: .2+w/10000.)
    return SimpleNamespace(medium=medium, detector=detector,
         wavelength=WavelengthQuadrature(wavelengths, np.array([30., 50., 20.])),
         acceptance=lambda: dict(alpha=np.array([4*np.pi*.6, 4*np.pi*.3/3,
                                                 4*np.pi*.08/5, 4*np.pi*.02/7])))


def source(start, direction, length, beta):
    n = len(beta)
    c0 = np.linspace(1000., 2000., n)
    return SpectralLightElements(np.asarray(start, float), np.asarray(direction, float),
              np.asarray(length, float), c0, -c0/np.asarray(beta)**2, np.asarray(beta, float),
              np.zeros(n), np.ones(n), np.arange(n), np.zeros(n, int), cone_model='spectral')


def reference(kernel, elements):
    tables = _spectral_tables(kernel)
    fields = _ballistic_fields(elements, elements, tables)
    result = np.zeros(len(kernel.detector.positions_m))
    for lam, wavelength in enumerate(tables['wavelength_nm']):
        item = fields(lam)
        if item is None:
            continue
        field, cone = item
        q, _ = ballistic_directional(field, kernel.detector.positions_m,
                    -kernel.detector.orientations, kernel.medium.band(wavelength),
                    cone, kernel.acceptance()['alpha'])
        result += (tables['weight_nm'][lam]*tables['efficiency'][lam]
                   *kernel.detector.effective_area_m2*q)
    return result


def test_random_spectral_thresholds_and_directional_acceptance():
    rng = np.random.default_rng(781)
    elements = source(rng.normal(size=(71, 3))*5, rng.normal(size=(71, 3)),
                      rng.uniform(.1, 20, 71), rng.uniform(.70, 1., 71))
    kernel = setup_kernel(rng.normal(size=(29, 3))*12)
    actual, expected = spectral_ballistic_charge(kernel, elements), reference(kernel, elements)
    assert np.count_nonzero(expected) > 15
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-15)


def test_right_open_root_and_near_axis_floor():
    # Cone cosine .6/sine .8 gives exact cot .75. Start owns root0; end does not.
    elements = source([[0, 0, 0]], [[0, 0, 1]], [2.], [1.])
    kernel = setup_kernel([[4., 0, 3.], [4., 0, 5.], [4., 0, np.nextafter(5., 0)],
                           [0., 0, 1.], [1e-150, 0, 1.], [0., 0, -1.]], phase=(5/3,)*3)
    actual, expected = spectral_ballistic_charge(kernel, elements), reference(kernel, elements)
    assert expected[0] > 0 and expected[1] == 0 and expected[2] > 0
    assert actual[-1] == 0
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-15)


def test_all_below_threshold_and_frozen_rejected():
    elements = source([[0, 0, 0]], [[0, 0, 1]], [2.], [.7])
    kernel = setup_kernel([[1., 0., 1.]])
    np.testing.assert_array_equal(spectral_ballistic_charge(kernel, elements), [0.])
    from dataclasses import replace
    with pytest.raises(ValueError, match='spectral'):
        spectral_ballistic_charge(kernel, replace(elements, cone_model='frozen'))
