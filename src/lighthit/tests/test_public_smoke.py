import numpy as np

import lighthit as lh


def test_installed_geometry_and_numeric_source_pose():
    detector = lh.DetectorArray(
        [[-1., 0., 0.], [1., 0., 0.]], [[0., 0., 1.]] * 2, .05,
        lambda x: np.ones_like(np.asarray(x, float)),
        lambda w: np.ones_like(np.asarray(w, float)),
        identifiers={"cluster": [1, 1]})
    summary = lh.describe_geometry(detector)
    np.testing.assert_array_equal(summary.centroid_m, [0., 0., 0.])
    shower = lh.SyntheticShower.gaussian(
        lh.SourcePose([4., 5., 6.], [0., 1., 0.], 7.), elements=12, seed=3)
    np.testing.assert_allclose(shower.centroid_m, [4., 5., 6.], atol=1e-12)
    np.testing.assert_allclose(shower.principal_axis, [0., 1., 0.], atol=1e-12)
    assert shower.elements.start_ns.min() == 7.
