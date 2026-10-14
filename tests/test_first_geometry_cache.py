"""Exact position/clock caching must preserve frame, shape and grouping keys."""
import math

import numpy as np
import pytest

numba = pytest.importorskip("numba")
from lighthit import _prompt_numba as kernels


@numba.njit
def _original_geometry(rx, ry, rz, origin, position, clock, speed, tau_step, log_step):
    dx, dy, dz = rx-position[0], ry-position[1], rz-position[2]
    r = math.sqrt(dx*dx + dy*dy + dz*dz)
    t_emit = clock-origin
    tau = clock-origin+r/speed
    radial = int(math.floor(math.log(r)/log_step))+(1 << 20)
    temporal = min(max(int(math.floor(tau/tau_step))+(1 << 20), 0), (1 << 21)-1)
    return np.array([r, dx/r, dy/r, dz/r, t_emit, tau, math.log(r)]), radial, temporal


@numba.njit
def _original_frame(dx, dy, dz, sx, sy, sz, nx, ny, nz):
    r = math.sqrt(dx*dx + dy*dy + dz*dz)
    hx, hy, hz = dx/r, dy/r, dz/r
    cosine = hx*sx + hy*sy + hz*sz
    px, py, pz = hx-cosine*sx, hy-cosine*sy, hz-cosine*sz
    sine = math.sqrt(px*px + py*py + pz*pz)
    c1 = sx*nx + sy*ny + sz*nz
    c3 = (px*nx + py*ny + pz*nz)/sine if sine > 1e-300 else 0.0
    return r, math.atan2(sine, cosine), sine, cosine, c1, c3


def _assert_bits(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual, np.float64).view(np.uint64),
                                  np.asarray(expected, np.float64).view(np.uint64))


def test_exact_geometry_ids_preserve_signed_zero_clocks_and_order():
    positions = np.array([[0., 0., 0.], [-0., 0., 0.], [0., -0., 0.],
                          [0., 0., -0.], [0., 0., 0.], [0., 0., 0.],
                          [0., 0., 0.], [1., 2., 3.], [0., 0., 0.],
                          [1., 2., 3.]])
    clocks = np.array([0., 0., 0., 0., -0., 0., 1e300, -1e300, 1e300, -1e300])
    ids, representatives = kernels._exact_pencil_geometry_ids(positions, clocks)
    np.testing.assert_array_equal(ids, [0, 1, 2, 3, 4, 0, 5, 6, 5, 6])
    np.testing.assert_array_equal(representatives, [0, 1, 2, 3, 4, 6, 7])


def test_geometry_hash_handles_duplicates_and_nonduplicates():
    rng = np.random.default_rng(934)
    distinct = rng.normal(size=(3000, 4))
    points = np.concatenate([distinct, distinct[rng.integers(0, 3000, 2000)]])
    # Arrays match the solver's contiguous float64 input contract.
    positions, clocks = np.ascontiguousarray(points[:, :3]), np.ascontiguousarray(points[:, 3])
    ids, representatives = kernels._exact_pencil_geometry_ids(positions, clocks)
    np.testing.assert_array_equal(representatives, np.arange(3000))
    _assert_bits(points, points[representatives[ids]])
    np.testing.assert_array_equal(ids[:3000], np.arange(3000))


@pytest.mark.parametrize("origin", [0., -0., 1013.21, -1e12, 1e12])
def test_cached_receiver_rows_and_keys_match_original_bitwise(origin):
    rng = np.random.default_rng(573)
    positions = np.ascontiguousarray(rng.normal(size=(50, 3))*100.)
    positions = np.concatenate([positions, positions[:20]])
    clocks = np.r_[rng.normal(size=46)*1e3, 0., -0., 1e300, -1e300]
    clocks = np.r_[clocks, clocks[:20]]
    ids, representatives = kernels._exact_pencil_geometry_ids(positions, clocks)
    rows, keys = np.empty((len(representatives), 7)), np.empty((len(representatives), 2), np.int64)
    for receiver in [[0., -0., 370.], [100., -340., 290.], [1e100, -1e100, 1e100]]:
        kernels._prepare_receiver_geometry(*receiver, origin, positions, clocks,
            representatives, .225, 2., .005, rows, keys)
        for p in range(len(positions)):
            old, radial, temporal = _original_geometry(*receiver, origin, positions[p],
                                                        clocks[p], .225, 2., .005)
            _assert_bits(rows[ids[p]], old)
            np.testing.assert_array_equal(keys[ids[p]], [radial, temporal])
            direction = rng.normal(size=3); direction /= np.linalg.norm(direction)
            look = rng.normal(size=3); look /= np.linalg.norm(look)
            old_frame = _original_frame(*(np.asarray(receiver)-positions[p]), *direction, *look)
            cached_frame = kernels._pencil_frame_from_ray(*rows[ids[p], :4], *direction, *look)
            _assert_bits(cached_frame, old_frame)
            _assert_bits(kernels._pencil_frame(*(np.asarray(receiver)-positions[p]),
                                              *direction, *look), old_frame)


def test_cached_ray_shape_preserves_all_branches_bitwise():
    rng = np.random.default_rng(72)
    n = 20
    positions = rng.normal(size=(n, 3))*100.
    directions = rng.normal(size=(n, 3)); directions /= np.linalg.norm(directions, axis=1)[:, None]
    major = rng.normal(size=(n, 3))
    major -= (major*directions).sum(axis=1)[:, None]*directions
    major /= np.linalg.norm(major, axis=1)[:, None]
    mat = rng.normal(size=(n, 3, 3)); covariance = mat @ mat.transpose(0, 2, 1)
    covariance = covariance[:, (0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]
    var_major, var_minor = np.full(n, .003), np.full(n, .0002)
    prepared = kernels._prepare_pencil_shape(directions, major, covariance)
    tangent, weighted = kernels._prepare_gauss_inverse(kernels._GAUSS_TAU, kernels._GAUSS_W)
    for p in range(n):
        for distance in [1e-4, 1., 40., 400.]:
            for theta in [0., 1e-10, .002, .03, .5, 1.5]:
                receiver = positions[p]+distance*(math.cos(theta)*directions[p]+math.sin(theta)*major[p])
                # Reuse precisely the actual displacement/r arithmetic, also
                # when rounding makes the constructed receiver angle differ.
                dx = receiver-positions[p]; r = math.sqrt(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2])
                ray = dx/r
                old = kernels._pencil_shape_factor_prepared(p, r, theta, math.sin(theta),
                    *receiver, positions, major, var_major, var_minor, prepared, 3., tangent, weighted)
                new = kernels._pencil_shape_factor_from_ray(p, r, theta, math.sin(theta),
                    *ray, major, var_major, var_minor, prepared, 3., tangent, weighted)
                assert old[2:] == new[2:]
                _assert_bits(old[:2], new[:2])


def test_zero_radius_retains_original_exception():
    positions, clocks = np.zeros((1, 3)), np.zeros(1)
    with pytest.raises(ZeroDivisionError):
        kernels._pencil_frame(0., 0., 0., 1., 0., 0., 0., 0., 1.)
    with pytest.raises(ZeroDivisionError):
        kernels._prepare_receiver_geometry(0., 0., 0., 0., positions, clocks,
            np.zeros(1, np.int64), .225, 2., .005, np.empty((1, 7)), np.empty((1, 2), np.int64))
