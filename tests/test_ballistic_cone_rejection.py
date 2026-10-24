"""Cheap cone rejection must retain every independently intersecting ray."""
import numpy as np
import pytest

pytest.importorskip('numba')
from lighthit._ballistic_interval_numba import _cone_can_reach_sphere
from lighthit.receiver import SphereSurface


def test_ray_sphere_intersections_are_never_rejected_across_cone_band():
    rng = np.random.default_rng(57)
    for i in range(3000):
        beta = rng.uniform(.74, 1.)
        phase_min, phase_max = 1.34, 1.38
        low = max(phase_min, np.nextafter(1./beta, np.inf))
        if low >= phase_max:
            continue
        phase = [low, phase_max, rng.uniform(low, phase_max)][i % 3]
        c = 1./(beta*phase); s = np.sqrt(max(0., 1.-c*c))
        direction = np.array([s, 0., c])
        normal = np.array([c, 0., -s])
        length = 10**rng.uniform(-4., 2.)
        radius = 10**rng.uniform(-2., 0.)
        distance = radius*10**rng.uniform(.4, 5.)
        fraction = rng.uniform(-.999, .999)
        emission = np.array([0., 0., rng.uniform(0., length)])
        centre = emission+distance*direction+fraction*radius*normal
        entry = SphereSurface(radius).entry_distance(emission-centre, direction)
        assert np.isfinite(entry)
        assert _cone_can_reach_sphere(centre[2], abs(centre[0]), length, beta,
                                      radius, phase_min, phase_max)


@pytest.mark.parametrize('radius', [.01, .216, 1.])
def test_cone_tangencies_at_step_ends_are_retained(radius):
    beta, nlo, nhi, length, distance = .99, 1.34, 1.38, 2., 100.
    for phase, end, sign in [(nhi, 0., 1.), (nlo, length, -1.)]:
        c = 1./(beta*phase); s = np.sqrt(1.-c*c)
        rho = distance*s+sign*radius*c
        z = end+distance*c-sign*radius*s
        assert _cone_can_reach_sphere(z, rho, length, beta, radius, nlo, nhi)
        rho += sign*radius*.01*c
        z -= sign*radius*.01*s
        assert not _cone_can_reach_sphere(z, rho, length, beta, radius, nlo, nhi)


def test_rejects_backward_sphere_and_below_threshold_source():
    assert not _cone_can_reach_sphere(-.3, 1., 1., 1., .2, 1.34, 1.38)
    assert not _cone_can_reach_sphere(1., 1., 1., .7, .2, 1.34, 1.38)
