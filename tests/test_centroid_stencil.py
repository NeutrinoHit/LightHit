"""Optional local-centroid stencils preserve joint source moments."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("numba")
from lighthit._shower_basis import compile_axial_slices
from lighthit.experimental.axial_source import AxisFrame
from lighthit.experimental.g4_source import LightElements, SourceContract


FRAME = AxisFrame(np.zeros(3), np.array([0., 0., 1.]),
                  np.array([1., 0., 0.]), np.array([0., 1., 0.]))


def field(seed=5, count=36):
    rng = np.random.default_rng(seed)
    positions = rng.normal(size=(count, 3))*[.08, .12, 1.2] + [.5, -.3, 0.]
    directions = rng.normal(size=(count, 3))*.2 + [0., 0., 1.]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    length = rng.uniform(.01, .04, count)
    times = np.linspace(0., 4., count)
    return LightElements(positions, directions, length, rng.uniform(.5, 2., count),
                         rng.uniform(.7, .8, count), times, times+.2,
                         np.arange(count), np.arange(count), SourceContract(), {})


def moments(source):
    points = source.frame.rotate(source.points_m()-source.frame.centre_m)
    x, y = points[:, 0], points[:, 1]
    monomials = np.stack([np.ones(len(x)), x, y, x*x, y*y])
    return np.einsum("kp,pcw->kcw", monomials, source.channels)


@pytest.mark.parametrize("axial_order", [1, 3])
@pytest.mark.parametrize("time_deposit", ["linear", "nearest"])
def test_centred_stencil_preserves_every_field_harmonic_frequency_moment(axial_order, time_deposit):
    first = field()
    second = replace(first, photons=first.photons*np.linspace(.2, 1.8, len(first)))
    options = dict(azimuthal_degree=3, cell_m=.3, axial_order=axial_order,
                   stencil_threshold_m=0., frame=FRAME, element_order=2,
                   time_deposit=time_deposit)
    old = compile_axial_slices([first, second], 6, [0., .17, .8], **options)
    new = compile_axial_slices([first, second], 6, [0., .17, .8],
                               stencil_center="centroid", **options)
    assert new[0].summary["stencil_slices"] > 0
    for axis_source, centered_source in zip(old, new):
        np.testing.assert_allclose(moments(centered_source), moments(axis_source),
                                   rtol=3e-12, atol=3e-13)
    # The two coefficient fields must use exactly the same geometry.
    np.testing.assert_array_equal(new[0].points_m(), new[1].points_m())


def test_default_is_bitwise_identical_to_explicit_axis_mode():
    options = dict(azimuthal_degree=2, cell_m=.4, frame=FRAME,
                   stencil_threshold_m=.03)
    implicit = compile_axial_slices(field(), 4, [0., .3], **options)
    explicit = compile_axial_slices(field(), 4, [0., .3], stencil_center="axis", **options)
    np.testing.assert_array_equal(implicit.points_m(), explicit.points_m())
    np.testing.assert_array_equal(implicit.channels, explicit.channels)
    assert implicit.summary == explicit.summary


def test_locally_thin_offset_track_uses_one_point_per_slice():
    track = field(count=80)
    positions = np.column_stack([np.full(80, .5), np.full(80, -.25), np.linspace(0., 4., 80)])
    track = replace(track, start_m=positions, direction=np.tile([0., 0., 1.], (80, 1)),
                    length_m=np.full(80, .01))
    options = dict(azimuthal_degree=2, cell_m=.25, frame=FRAME,
                   stencil_threshold_m=.03)
    old = compile_axial_slices(track, 4, [0., .3], **options)
    new = compile_axial_slices(track, 4, [0., .3], stencil_center="centroid", **options)
    assert new.summary["stencil_slices"] == 0
    assert old.summary["stencil_slices"] > 0
    assert len(new.z_m) == new.summary["slices"]
    assert len(old.z_m)-len(new.z_m) == 4*old.summary["stencil_slices"]
    np.testing.assert_allclose(moments(new), moments(old), rtol=2e-12, atol=1e-12)


def test_signed_cubic_moments_cannot_hide_a_wide_support():
    # At slice z=0, +.5625*1 and -.0625*9 at x=1 cancel all C0
    # transverse moments. The x=0 point leaves positive net C0 mass, so a
    # variance-only criterion would call the slice thin even though its
    # angular and temporal transverse coefficients do not cancel.
    item = field(count=3)
    item = replace(item, start_m=np.array([[0., 0., -.005], [1., 0., .495], [1., 0., 1.495]]),
                   direction=np.tile([0., 0., 1.], (3, 1)), length_m=np.full(3, .01),
                   photons=np.array([1., 1., 9.]), cone_cosine=np.array([.7, .75, .8]))
    new = compile_axial_slices(item, 4, [0., .7], cell_m=1., frame=FRAME,
                               stencil_threshold_m=.03, stencil_center="centroid", element_order=1)
    assert np.count_nonzero(np.isclose(new.z_m, 0., rtol=0., atol=1e-13)) == 5
    assert np.isfinite(new.channels).all()


def test_zero_mass_signed_slice_has_finite_centroid_and_nonzero_channels():
    item = field(count=3)
    item = replace(item, start_m=np.array([[0., 0., -.005], [1., 0., .495], [1., 0., 1.495]]),
                   direction=np.tile([0., 0., 1.], (3, 1)), length_m=np.full(3, .01),
                   photons=np.array([0., 1., 9.]), cone_cosine=np.array([.7, .75, .8]))
    new = compile_axial_slices(item, 4, [0., .7], cell_m=1., frame=FRAME,
                               stencil_threshold_m=.03, stencil_center="centroid", element_order=1)
    assert np.isfinite(new.points_m()).all()
    assert np.isfinite(new.channels).all()
    selected = np.isclose(new.z_m, 0., rtol=0., atol=1e-13)
    assert np.any(new.channels[selected, 1:] != 0.)


def test_nearly_cancelled_slice_stays_inside_physical_transverse_support():
    # At z=0, +.5625 at x=0 and -.0625*(9-eps) at x=1
    # leave tiny mass. The signed centroid diverges as eps -> 0.
    item = field(count=3)
    item = replace(item,start_m=np.array([[0.,0.,-.005],[0.,0.,.495],[1.,0.,1.495]]),
        direction=np.tile([0.,0.,1.],(3,1)),length_m=np.full(3,.01),
        photons=np.array([0.,1.,9.-1e-5]),cone_cosine=np.array([.7,.75,.8]))
    options=dict(cell_m=1.,frame=FRAME,stencil_threshold_m=.03,element_order=1,
        azimuthal_degree=2)
    stable=compile_axial_slices(item,4,[0.,.7],stencil_center='centroid',**options)
    axis=compile_axial_slices(item,4,[0.,.7],**options)
    selected=np.isclose(stable.z_m,0.,atol=1e-13,rtol=0.)
    points=stable.points_m()[selected]
    assert len(points)==5
    assert points[:,0].min()>=-.2 and points[:,0].max()<=1.2
    np.testing.assert_allclose(moments(stable),moments(axis),rtol=3e-12,atol=3e-13)


def test_centroid_charge_and_spectrum_have_identical_points_and_subselection():
    options = dict(azimuthal_degree=2, cell_m=.3, frame=FRAME,
                   stencil_threshold_m=.03, stencil_center="centroid")
    zero = compile_axial_slices(field(), 4, [0.], **options)
    full = compile_axial_slices(field(), 4, [0., .3], **options)
    np.testing.assert_array_equal(zero.points_m(), full.points_m())
    selected = np.arange(len(full.z_m)) % 3 == 1
    part = compile_axial_slices(field(), 4, [0., .3], point_selection=selected, **options)
    np.testing.assert_array_equal(part.points_m(), full.points_m()[selected])
    np.testing.assert_array_equal(part.channels, full.channels[selected])


def test_invalid_stencil_center_is_rejected():
    with pytest.raises(ValueError, match="stencil_center"):
        compile_axial_slices(field(), 4, [0.], stencil_center="arbitrary")


def subset(item, mask):
    names = ("start_m", "direction", "length_m", "photons", "cone_cosine",
             "start_ns", "end_ns", "row_index", "uid")
    return replace(item, **{name: getattr(item, name)[mask] for name in names})


def test_fixed_c0_geometry_is_identical_across_spectral_weights_and_thresholds():
    reference = field(count=40)
    # Remove geometrically extreme and early/late rows in the second band;
    # their zero physical weights must not remove reference spatial support.
    keep = np.arange(len(reference)) % 3 == 1
    first = replace(reference, photons=reference.photons*np.linspace(.1, 2., len(reference)),
                    cone_cosine=reference.cone_cosine*.95)
    second = subset(replace(reference, photons=reference.photons*np.linspace(1.8, .2, len(reference)),
                             cone_cosine=reference.cone_cosine*.98), keep)
    options = dict(azimuthal_degree=2, cell_m=.3, stencil_threshold_m=.03,
                   stencil_center="centroid", geometry_reference=reference)
    a = compile_axial_slices(first, 4, [0., .3], **options)
    b = compile_axial_slices(second, 4, [0., .3], **options)
    b0 = compile_axial_slices(second, 4, [0.], **options)
    np.testing.assert_array_equal(a.points_m(), b.points_m())
    np.testing.assert_array_equal(b0.points_m(), b.points_m())
    assert a.reference_ns == b.reference_ns == b0.reference_ns
    assert a.summary["slices"] == b.summary["slices"]
    assert a.summary["stencil_slices"] == b.summary["stencil_slices"]
    for physical, compiled in ((first, a), (second, b)):
        assert compiled.channels[:, 0, 0].real.sum()*np.sqrt(4*np.pi) == pytest.approx(
            physical.photons.sum(), rel=3e-13)


def test_fixed_geometry_matches_explicit_zero_padding_without_extra_emission():
    reference = field(count=40)
    keep = np.arange(len(reference)) % 3 == 1
    physical = replace(reference, photons=reference.photons*np.linspace(.2, 1.8, len(reference)),
                       cone_cosine=reference.cone_cosine*.95)
    padded = replace(physical, photons=np.where(keep, physical.photons, 0.))
    options = dict(azimuthal_degree=3, cell_m=.3, stencil_threshold_m=0.,
                   stencil_center="centroid", frame=FRAME)
    # Existing two-field compilation supplies the C0 geometry independently;
    # the new reference-only path must reproduce every physical coefficient.
    # The old two-field API shares one cone across its weight fields. The
    # reference-only API below must ignore the C0 reference's original cone.
    geometry_carrier = replace(reference, cone_cosine=padded.cone_cosine)
    expected = compile_axial_slices([geometry_carrier, padded], 6, [0., .17, .8], **options)[1]
    actual = compile_axial_slices(subset(physical, keep), 6, [0., .17, .8],
                                  geometry_reference=reference, **options)
    np.testing.assert_array_equal(actual.points_m(), expected.points_m())
    np.testing.assert_allclose(actual.channels, expected.channels, rtol=3e-12, atol=3e-13)
    # Different stencil centres must still reproduce every joint moment of
    # the same padded physical source, not just its integrated photon count.
    axis = compile_axial_slices(padded, 6, [0., .17, .8], azimuthal_degree=3,
                                cell_m=.3, frame=FRAME, stencil_threshold_m=0.)
    np.testing.assert_allclose(moments(actual), moments(axis), rtol=3e-12, atol=3e-13)


def test_equal_geometry_reference_preserves_centroid_result_bitwise():
    reference = field()
    options = dict(azimuthal_degree=2, cell_m=.3, stencil_threshold_m=.03,
                   stencil_center="centroid", frame=FRAME)
    old = compile_axial_slices(reference, 4, [0., .3], **options)
    shared = compile_axial_slices(reference, 4, [0., .3], geometry_reference=reference, **options)
    np.testing.assert_array_equal(old.points_m(), shared.points_m())
    np.testing.assert_array_equal(old.channels, shared.channels)


def test_geometry_reference_accepts_only_direction_normalization_roundoff():
    reference = field()
    normalized = replace(reference, direction=reference.direction/np.linalg.norm(reference.direction, axis=1)[:, None])
    options = dict(stencil_center="centroid", geometry_reference=reference)
    a = compile_axial_slices(reference, 4, [0.], **options)
    b = compile_axial_slices(normalized, 4, [0.], **options)
    np.testing.assert_array_equal(a.points_m(), b.points_m())
    np.testing.assert_array_equal(a.channels, b.channels)
    changed = reference.direction.copy()
    changed[0, 0] += 1e-8
    with pytest.raises(ValueError, match="direction differs"):
        compile_axial_slices(replace(reference, direction=changed), 4, [0.], **options)


@pytest.mark.parametrize("name", ["start_m", "length_m", "start_ns", "end_ns", "uid"])
def test_geometry_reference_rejects_changed_chord_or_clock(name):
    reference = field()
    values = getattr(reference, name).copy()
    values.flat[0] += 1
    with pytest.raises(ValueError, match=f"{name} differs"):
        compile_axial_slices(replace(reference, **{name: values}), 4, [0.],
                              stencil_center="centroid", geometry_reference=reference)


def test_geometry_reference_requires_unique_matching_and_supported_rows():
    reference = field()
    options = dict(stencil_center="centroid", geometry_reference=reference)
    changed = reference.row_index.copy()
    changed[0] = 10000
    with pytest.raises(ValueError, match="absent"):
        compile_axial_slices(replace(reference, row_index=changed), 4, [0.], **options)
    changed[0] = changed[1]
    with pytest.raises(ValueError, match="unique"):
        compile_axial_slices(replace(reference, row_index=changed), 4, [0.], **options)
    with pytest.raises(ValueError, match="geometry_reference must have unique"):
        compile_axial_slices(reference, 4, [0.], stencil_center="centroid",
                              geometry_reference=replace(reference, row_index=changed))
    weights = reference.photons.copy()
    weights[0] = 0.
    with pytest.raises(ValueError, match="cover every emitting"):
        compile_axial_slices(reference, 4, [0.], stencil_center="centroid",
                              geometry_reference=replace(reference, photons=weights))


def test_reference_support_stays_wide_when_a_band_drops_its_wide_rows():
    reference = field(count=4)
    reference = replace(reference, start_m=np.array([[0., 0., 0.], [.01, 0., .01], [1., 0., .02], [1., 0., .03]]),
                        direction=np.tile([0., 0., 1.], (4, 1)), length_m=np.full(4, .001),
                        photons=np.ones(4))
    narrow = subset(reference, np.array([True, True, False, False]))
    options = dict(azimuthal_degree=0, frame=FRAME, cell_m=1.,
                   stencil_center="centroid", stencil_threshold_m=.03)
    independent = compile_axial_slices(narrow, 2, [0.], **options)
    common = compile_axial_slices(narrow, 2, [0.], geometry_reference=reference, **options)
    all_rows = compile_axial_slices(reference, 2, [0.], geometry_reference=reference, **options)
    assert independent.summary["stencil_slices"] == 0
    assert common.summary["stencil_slices"] > 0
    np.testing.assert_array_equal(common.points_m(), all_rows.points_m())


@pytest.mark.parametrize("options", [{}, {"stencil_center": "centroid", "transverse": "collapse"}])
def test_geometry_reference_requires_centroid_stencil_mode(options):
    with pytest.raises(ValueError, match="requires stencil_center"):
        compile_axial_slices(field(), 4, [0.], geometry_reference=field(), **options)
