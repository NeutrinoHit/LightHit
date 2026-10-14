"""The BLAS operator retains the scalar operator's full source and spectrum."""
import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from lighthit.experimental.axial_source import AxisFrame
from lighthit.experimental.axial_fast import _layout
from lighthit.experimental.point_ge2 import apply_point_spectral
from lighthit.experimental.point_ge2_blas import (
    apply_point_spectral_blas, clear_matrix_cache, matrix_cache_info,
    _coefficient_matrix,
)
from lighthit.readout import inverse_bins
from collections import OrderedDict


def _case(degree=6, m=4, detector_degree=3, strided=False):
    rng = np.random.default_rng(241)
    nlam = 3
    omega = np.linspace(0., 1.2, 11)
    log_r = np.log([3., 8., 20., 50., 100., 300.])
    pairs = [(ell, mu) for ell in range(detector_degree+1) for mu in range(ell+1)]
    shape = (nlam, len(log_r)-1, degree+1, len(pairs), 4, len(omega))
    coefficients = (rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(np.complex64)
    if strided:
        storage = np.zeros(shape[:-1]+(2*shape[-1],), np.complex64)
        storage[..., ::2] = coefficients
        coefficients = storage[..., ::2]
    bundle = dict(degree=degree, acceptance_degree=detector_degree,
        omega=omega, log_r=log_r, pair_lambda=np.array([p[0] for p in pairs]),
        pair_mu=np.array([p[1] for p in pairs]), coefficients=coefficients,
        absorption=np.array([.03, .04, .06]), speed=np.array([.22, .218, .215]),
        phase='flight')
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    frame = AxisFrame(np.array([3., 5., 7.]), rotation[:, 2], rotation[:, 0], rotation[:, 1])
    points = rng.normal(size=(31, 3))*30.
    points[-1] = [2000., 2000., 2000.]  # outside the cache window
    points[-2] = [40., 1., 20.]        # coincident with OM0: below radial window
    receivers = np.array([[40., 1., 20.], [-20., 30., -40.], [1., 1., 90.]])
    looks = rng.normal(size=receivers.shape)
    looks /= np.linalg.norm(looks, axis=1)[:, None]
    times = rng.uniform(-20., 4000., len(points))
    _, _, _, kept, _ = _layout(degree, m)
    angular = rng.normal(size=(nlam, len(points), len(kept)))
    angular[1, 1::2] = 0.             # wavelength-specific Cherenkov thresholds
    angular[:, 3] = 0.
    scales = rng.uniform(.1, 2., size=(nlam, len(receivers)))
    scales[2, 1] = 0.
    alpha = rng.uniform(-1., 1., detector_degree+1)
    return bundle, frame, points, angular, times, receivers, looks, alpha, scales


def _check(reference, candidate):
    error = np.max(np.abs(candidate-reference))/max(np.max(np.abs(reference)), 1e-300)
    assert error < 1e-12


@pytest.mark.parametrize('flight', [False, True])
@pytest.mark.parametrize('degree,m,out', [(4, 0, 2), (6, 3, 3)])
@pytest.mark.parametrize('strided', [False, True])
def test_blas_signed_spectrum_and_bins(flight, degree, m, out, strided):
    args = _case(degree, m, out, strided)
    args[0]['phase'] = 'flight' if flight else 'absolute'
    reference = apply_point_spectral(*args, azimuthal_degree=m)
    actual = apply_point_spectral_blas(*args, azimuthal_degree=m,
        block_size=7, pair_budget_bytes=1024)
    _check(reference, actual)
    omega = args[0]['omega']
    fine = np.linspace(0., 1.2, 51)
    edges = np.linspace(-10., 150., 49)
    rb = inverse_bins(fine, CubicSpline(omega, reference, axis=0)(fine), edges)
    ab = inverse_bins(fine, CubicSpline(omega, actual, axis=0)(fine), edges)
    _check(rb, ab)


def test_mutable_pack_and_readonly_view_observe_mutation():
    clear_matrix_cache()
    args = _case()
    first = apply_point_spectral_blas(*args, block_size=11)
    assert matrix_cache_info()['entries'] == 0
    args[0]['coefficients'] *= 1.3
    second = apply_point_spectral_blas(*args, block_size=11)
    _check(apply_point_spectral(*args), second)
    # A readonly view must not enter the persistent cache while its base is writable.
    view = args[0]['coefficients'].view()
    view.flags.writeable = False
    args[0]['coefficients'] = view
    third = apply_point_spectral_blas(*args, block_size=11)
    assert matrix_cache_info()['entries'] == 0
    view.base[...] *= 2.
    fourth = apply_point_spectral_blas(*args, block_size=11)
    _check(third*2., fourth)


def test_readonly_cache_bounded_and_distinguishes_pack_layouts(monkeypatch):
    import lighthit.experimental.point_ge2_blas as module
    clear_matrix_cache()
    args = _case()
    co = args[0]['coefficients']
    co.flags.writeable = False
    one = _coefficient_matrix(co, 0, 0, OrderedDict())
    assert _coefficient_matrix(co.view(), 0, 0, OrderedDict()) is one
    limit = one.nbytes*2
    monkeypatch.setattr(module, '_MATRIX_CACHE_LIMIT', limit)
    for band in range(3):
        _coefficient_matrix(co, band, 1, OrderedDict())
    assert matrix_cache_info()['bytes'] <= limit
    other = np.array(co*3)
    other.flags.writeable = False
    three = _coefficient_matrix(other, 0, 0, OrderedDict())
    np.testing.assert_array_equal(three[:, 0::2], other[0, 0].reshape(-1, 11).real)
    np.testing.assert_array_equal(three[:, 1::2], other[0, 0].reshape(-1, 11).imag)
    clear_matrix_cache()


def test_radial_clip_and_global_clock_shift():
    args = _case()
    reference = apply_point_spectral(*args, radial_window_m=(20., 100.))
    actual = apply_point_spectral_blas(*args, radial_window_m=(20., 100.), block_size=9)
    _check(reference, actual)
    shifted = list(args)
    shifted[4] = args[4]+1e6
    result = apply_point_spectral_blas(*shifted, radial_window_m=(20., 100.), block_size=9)
    np.testing.assert_allclose(result,
        actual*np.exp(1j*args[0]['omega']*1e6)[:, None], rtol=3e-9,
        atol=3e-9*np.abs(actual).max())


def test_empty_and_dark_source():
    args = list(_case())
    args[3][:] = 0.
    np.testing.assert_array_equal(apply_point_spectral_blas(*args), np.zeros((11, 3)))
    args[2] = args[2][:0]
    args[3] = args[3][:, :0]
    args[4] = args[4][:0]
    np.testing.assert_array_equal(apply_point_spectral_blas(*args), np.zeros((11, 3)))


@pytest.mark.parametrize('keyword,value', [('block_size', 0), ('pair_budget_bytes', 0),
    ('azimuthal_degree', -1), ('radial_window_m', (1., 400.))])
def test_invalid_controls(keyword, value):
    with pytest.raises(ValueError):
        apply_point_spectral_blas(*_case(), **{keyword: value})


def test_exact_knots_and_adjacent_floats_discontinuous_coefficients():
    # Discontinuous random interval coefficients make selecting the wrong
    # interval observable; a normal smooth cubic pack could conceal the error.
    args = list(_case(degree=4, m=2, detector_degree=3))
    args[1] = AxisFrame(np.zeros(3), np.array([0., 0., 1.]),
                       np.array([1., 0., 0.]), np.array([0., 1., 0.]))
    knots = np.exp(args[0]['log_r'])
    radii = np.array([adjacent for knot in knots for adjacent in
                     (np.nextafter(knot, 0.), knot, np.nextafter(knot, np.inf))])
    args[2] = np.column_stack((-radii, np.zeros((len(radii), 2))))
    rng = np.random.default_rng(97)
    args[3] = rng.normal(size=(3, len(radii), args[3].shape[2]))
    args[4] = rng.uniform(0., 100., len(radii))
    args[5] = np.zeros((1, 3))
    args[6] = np.array([[0., 0., 1.]])
    args[8] = np.ones((3, 1))
    for window in [(3., 300.), (float(knots[1]), float(knots[-2]))]:
        expected = apply_point_spectral(*args, azimuthal_degree=2, radial_window_m=window)
        actual = apply_point_spectral_blas(*args, azimuthal_degree=2,
            radial_window_m=window, block_size=4)
        _check(expected, actual)
