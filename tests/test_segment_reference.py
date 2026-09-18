"""The coordinate reference must agree with routes that share nothing with it."""
import numpy as np
import pytest
from scipy.special import roots_legendre

from lighthit import Medium
from lighthit.single import single_spectrum
from lighthit.experimental.cone_segment import ConeSegment, ballistic_spectrum
from lighthit.experimental.segment_reference import (SegmentGeometry,
                                                     ballistic_by_ray_counting,
                                                     directed_first_order,
                                                     phase_function,
                                                     segment_ballistic_field,
                                                     segment_first_order,
                                                     segment_first_order_mc)

MEDIUM = Medium(0.04, 0.05, 0.7, 1.35)
SEGMENT = ConeSegment((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 8.0, 0.99, 1.34, 1.0)
LIT = np.array([10.0, 0.0, 14.0])
UNLIT = np.array([12.0, 0.0, -6.0])


def test_both_kernels_are_normalised():
    x, w = roots_legendre(400)
    for kind, degree in (("hg", None), ("truncated", 24)):
        value = phase_function(x, MEDIUM, kind=kind, degree=degree)
        assert 2 * np.pi * (w * value).sum() == pytest.approx(1.0, rel=1e-12)


def test_truncated_kernel_converges_to_the_full_one():
    x = np.linspace(-1, 1, 51)
    full = phase_function(x, MEDIUM, kind="hg")
    errors = [np.max(np.abs(phase_function(x, MEDIUM, kind="truncated", degree=d) - full))
              for d in (8, 32, 128)]
    assert errors[0] > errors[1] > errors[2]


@pytest.mark.parametrize("omega", [0.0, 0.05, 0.2])
def test_directed_first_order_matches_the_repository_quadrature(omega):
    """One route integrates along the ray, the other over arrival ellipsoids."""
    radius, cosine = 12.0, 0.3
    detector = np.array([0.0, 0.0, radius])
    direction = np.array([np.sqrt(1 - cosine ** 2), 0.0, cosine])
    mine = directed_first_order([omega], np.zeros(3), direction, detector,
                                MEDIUM, kind="hg")[0]
    theirs = single_spectrum(np.array([omega]), radius, cosine, MEDIUM)[0][0]
    assert mine == pytest.approx(theirs, rel=1e-10)


def test_receiver_on_the_emission_ray_is_refused():
    with pytest.raises(ValueError, match="emission ray"):
        directed_first_order([0.0], np.zeros(3), np.array([0.0, 0.0, 1.0]),
                             np.array([0.0, 0.0, 9.0]), MEDIUM)


def test_product_rule_converges_far_from_a_ballistic_root():
    """With the peak far outside the segment the tensor rule is already exact."""
    coarse = segment_first_order([0.0], SEGMENT, UNLIT, MEDIUM, kind="hg",
                                 product_rule=True, longitudinal_order=12,
                                 azimuth_order=24)[0]
    fine = segment_first_order([0.0], SEGMENT, UNLIT, MEDIUM, kind="hg",
                               product_rule=True, longitudinal_order=24,
                               azimuth_order=48)[0]
    assert fine.real == pytest.approx(coarse.real, rel=1e-10)


def test_polar_and_product_rules_agree_where_both_converge():
    """Two discretisations of the same integral, sharing only the integrand."""
    polar = segment_first_order([0.0], SEGMENT, UNLIT, MEDIUM, kind="hg",
                                radial_order=20, polar_order=20, epsrel=1e-9)[0]
    product = segment_first_order([0.0], SEGMENT, UNLIT, MEDIUM, kind="hg",
                                  product_rule=True, longitudinal_order=24,
                                  azimuth_order=48)[0]
    assert polar.real == pytest.approx(product.real, rel=1e-5)


def test_polar_rule_handles_a_root_just_past_the_segment_end():
    """A cone that almost reaches the receiver is the slow case for a tensor rule."""
    short = ConeSegment((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0, 0.99, 1.34, 1.0)
    coarse = segment_first_order([0.0], short, LIT, MEDIUM, kind="hg",
                                 radial_order=16, polar_order=16, epsrel=1e-8)[0]
    fine = segment_first_order([0.0], short, LIT, MEDIUM, kind="hg",
                               radial_order=24, polar_order=24, epsrel=1e-8)[0]
    assert fine.real == pytest.approx(coarse.real, rel=1e-4)


def test_lit_geometry_polar_rule_converges():
    coarse = segment_first_order([0.0], SEGMENT, LIT, MEDIUM, kind="hg",
                                 radial_order=12, polar_order=12, epsrel=1e-8)[0]
    fine = segment_first_order([0.0], SEGMENT, LIT, MEDIUM, kind="hg",
                               radial_order=20, polar_order=20, epsrel=1e-8)[0]
    assert fine.real == pytest.approx(coarse.real, rel=2e-4)


def test_emission_side_and_arrival_side_agree():
    """Two parametrisations with no common variable, to sampling accuracy."""
    quadrature = segment_first_order([0.0], SEGMENT, UNLIT, MEDIUM, kind="hg",
                                     product_rule=True, longitudinal_order=16,
                                     azimuth_order=32)[0]
    sampled, error = segment_first_order_mc([0.0], SEGMENT, UNLIT, MEDIUM,
                                            kind="hg", samples=200_000,
                                            batches=8, seed=3)
    assert sampled[0].real == pytest.approx(quadrature.real, rel=0.05)
    assert error[0].real > 0


def test_ballistic_field_reproduces_the_prototype_at_a_receiver():
    fluence, incoming, time = segment_ballistic_field(LIT[None, :], SEGMENT, MEDIUM)
    small = 1e-4  # small enough that the phase does not wrap
    spectrum = ballistic_spectrum([0.0, small], SEGMENT,
                                  LIT[None, :] - np.asarray(SEGMENT.start_m), MEDIUM)
    assert fluence[0] == pytest.approx(spectrum[0, 0].real, rel=1e-12)
    phase = spectrum[1, 0] / spectrum[0, 0]
    assert np.angle(phase) / small == pytest.approx(time[0], rel=1e-6)
    assert np.linalg.norm(incoming[0]) == pytest.approx(1.0, rel=1e-12)


def test_ray_counting_supports_the_analytic_jacobian():
    analytic = ballistic_spectrum([0.0], SEGMENT,
                                  LIT[None, :] - np.asarray(SEGMENT.start_m),
                                  MEDIUM)[0, 0].real
    counted, error, edge = ballistic_by_ray_counting(
        SEGMENT, LIT, MEDIUM, acceptance_radius_m=0.05, samples=200_000, seed=5)
    assert edge == 0.0
    assert abs(counted - analytic) < 4 * error


def test_segment_geometry_finds_the_ballistic_root():
    geometry = SegmentGeometry.of(SEGMENT)
    root, azimuth = geometry.ballistic_root(LIT)
    assert 0.0 < root < geometry.length_m
    ray = geometry.cone_directions([azimuth])[0]
    point = np.asarray(geometry.start_m, float) + root * np.asarray(geometry.direction)
    towards = LIT - point
    assert np.dot(towards / np.linalg.norm(towards), ray) == pytest.approx(1.0, abs=1e-12)
    assert geometry.ballistic_root(UNLIT)[0] < 0.0
