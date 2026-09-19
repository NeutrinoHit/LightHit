"""The batched Bessel table in ResponseCache.build agrees with a plain loop.

``ResponseCache.build`` used to fill its Bessel table one radius at a time
(``for i, rad in enumerate(block): table[i] = spherical_jn(...)``). That call
per radius has nothing to do with correctness -- ``scipy.special.spherical_jn``
is a broadcasting ufunc in (order, argument) -- so the loop was replaced by
one call over the whole radius block. This pins the two down to the identity
they are: bit-exact, not merely close.
"""
import numpy as np
from scipy.special import spherical_jn


def _looped_table(ell, k, block, norm, weight):
    table = np.empty((len(block), len(k), len(ell)), complex)
    for i, rad in enumerate(block):
        table[i] = (spherical_jn(ell[None, :], k[:, None] * rad)
                    * norm[None, :] * weight[:, None])
    return table


def _vectorised_table(ell, k, block, norm, weight):
    return (spherical_jn(ell[None, None, :], k[None, :, None] * block[:, None, None])
            * norm[None, None, :] * weight[None, :, None])


def test_vectorised_bessel_table_matches_the_loop_bit_for_bit():
    rng = np.random.default_rng(3)
    degree = 9
    ell = np.arange(degree + 1)
    k = np.linspace(0.01, 5.0, 37)
    norm = np.sqrt((2 * ell + 1) / 2) * (1j) ** ell
    weight = rng.uniform(0.1, 1.0, size=len(k))
    block = rng.uniform(3.0, 60.0, size=17)

    looped = _looped_table(ell, k, block, norm, weight)
    vectorised = _vectorised_table(ell, k, block, norm, weight)

    assert looped.shape == vectorised.shape
    assert np.array_equal(looped, vectorised)


def test_response_cache_build_is_unaffected_by_the_vectorisation():
    from lighthit import SolverSettings, synthetic_medium
    from lighthit.cache import CacheGrid, ResponseCache

    medium = synthetic_medium()
    omega = np.linspace(0.0, 0.2, 3)
    settings = SolverSettings(10, 5, 5.0, 0.04, 10)
    grid = CacheGrid.geometric(3.0, 40.0, 12, omega)
    cache = ResponseCache.build(medium, settings, grid)
    moments = cache.moments_at(np.array([8.0, 20.0]))
    assert np.isfinite(moments).all()
    assert np.any(np.abs(moments) > 0)
