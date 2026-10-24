"""Complete-event composition, independent selection and Fourier contracts.

Transport primitives are replaced by known responses so these tests isolate
composition bugs rather than duplicate their numerical physics tests.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from lighthit import KernelConfig
from lighthit.experimental import joint_muon
from lighthit.model import DetectorArray, SpectralMedium
from lighthit.transport import TransportKernel


@pytest.fixture
def composed_case(monkeypatch):
    medium = SpectralMedium(np.array([350., 610.]), np.full(2, .05), np.full(2, .015),
        np.full(2, 1.34), np.full(2, 1.37), g=.9)
    detector = DetectorArray(np.array([[0., 0., 20.], [10., 0., 20.], [20., 0., 20.]]),
        np.tile([0., 0., 1.], (3, 1)), 1., lambda cosine: 1.+.2*cosine,
        lambda wavelength: np.ones_like(wavelength),
        identifiers={'cluster_id': np.ones(3, int), 'string_id': np.arange(3)})
    omega = np.array([0., .1, .2, .3])
    kernel = TransportKernel(medium, detector, KernelConfig(omega_per_ns=omega,
        readout_omega_per_ns=np.linspace(0., .3, 7), relative_time_edges_ns=np.array([0., 5., 10.]),
        threshold_pe=.01, cache_policy='require'))
    source = SimpleNamespace(start_m=np.zeros((2, 3)), direction=np.tile([0., 0., 1.], (2, 1)),
        length_m=np.ones(2), start_ns=np.zeros(2))
    kernel._source_elements = lambda value: value
    origins = np.array([100., 101., 102.])
    q0 = np.array([.002, .0001, .015])
    b0 = np.column_stack((q0, np.zeros(3)))
    preview = SimpleNamespace(charge_components_pe=np.array([[0., .005, .002], [.999, .00001, .00001], [0., .005, .002]]),
        charge_pe=np.array([0., .999, 0.]), unsafe_mask=np.zeros(3, bool),
        angular_scattered_pe=np.array([[.004, .001], [.000004, .000006], [.003, .001]]))
    calls = dict(ballistic=0, primary=[], secondary=[])
    def ballistic(*args, **kwargs):
        calls['ballistic'] += 1
        return SimpleNamespace(charge_pe=q0.copy(), bins_pe=b0.copy())
    monkeypatch.setattr('lighthit.ballistic_interval.ballistic_interval', ballistic)
    monkeypatch.setattr('lighthit._shower_basis.earliest_element_arrival', lambda *args: origins.copy())
    def primary(local, elements, *stores, **options):
        ids = (local.detector.positions_m[:, 0]/10).astype(int)
        calls['primary'].append(ids.copy())
        assert elements is source
        assert options['skip_ballistic'] is True
        assert local.config.threshold_pe == 0.
        np.testing.assert_array_equal(options['time_origin_ns'], origins[ids])
        charge = np.tile([999., .004, .002], (len(ids), 1))
        bins = np.zeros((len(ids), 2, 3)); bins[:, 0, :] = charge
        spectrum = .002*np.exp(1j*omega[:, None]*origins[ids][None, :])
        return SimpleNamespace(charge_orders_pe=charge, bins_orders_pe=bins,
            spectrum_ge2_pe=spectrum, metadata={'primary_only': True})
    monkeypatch.setattr('lighthit.experimental.universal_primary.transport_universal_primary', primary)
    def secondary(original_kernel, elements, ancestry, ids, **options):
        assert original_kernel is kernel
        assert elements is source
        calls['secondary'].append(ids.copy())
        np.testing.assert_array_equal(options['time_origin_ns'], origins[ids])
        q1 = np.full(len(ids), .005); q2 = np.full(len(ids), .001)
        b1 = np.column_stack((q1, np.zeros(len(ids))))
        b2 = np.column_stack((q2, np.zeros(len(ids))))
        spectrum = .001*np.exp(1j*omega[:, None]*origins[ids][None, :])
        return q1, b1, q2, b2, {'all_secondary_families': True}, spectrum
    monkeypatch.setattr(joint_muon, 'secondary_joint_transport', secondary)
    stores = dict(primary_first_store=object(), primary_ge2_store=SimpleNamespace(omega_per_ns=omega.copy()),
                  ge2_store=object(), joint_first_atlas=object())
    return kernel, source, preview, stores, origins, q0, b0, calls


def test_all_orders_conserved_without_primary_ballistic_double_count(composed_case):
    kernel, source, preview, stores, origins, q0, b0, calls = composed_case
    before = preview.charge_components_pe.copy()
    response = joint_muon.transport_joint_muon(kernel, source, {}, preview=preview, **stores)
    assert calls['ballistic'] == 1
    selected = np.concatenate(calls['primary'])
    np.testing.assert_array_equal(selected, [0, 2])
    np.testing.assert_array_equal(selected, np.concatenate(calls['secondary']))
    assert len(np.unique(selected)) == len(selected)
    np.testing.assert_allclose(response.charge_components_pe[selected],
        np.column_stack((q0[selected], np.full(2, .009), np.full(2, .003))), rtol=0, atol=3e-18)
    np.testing.assert_array_equal(response.components_pe[:, :, 0], b0)
    np.testing.assert_allclose(response.components_pe[selected].sum(1), response.charge_components_pe[selected])
    np.testing.assert_allclose(response.spectrum_ge2_pe[:, selected],
        .003*np.exp(1j*kernel.config.omega_per_ns[:, None]*origins[selected][None, :]), atol=1e-18)
    np.testing.assert_array_equal(response.active, [True, False, True])
    # The misleading preview direct charge on OM1 must be replaced globally.
    assert response.charge_components_pe[1, 0] == q0[1]
    np.testing.assert_array_equal(preview.charge_components_pe, before)


def test_glass_direct_charge_selects_globally_but_bins_only_evaluated_oms(composed_case, monkeypatch):
    from dataclasses import replace
    from lighthit.receiver import SphereSurface
    kernel, source, preview, stores, origins, q0, b0, calls = composed_case
    kernel.detector = replace(kernel.detector, receiver_surface=SphereSurface(.216))
    binned = []

    class Prepared:
        def charge(self):
            return SimpleNamespace(charge_pe=q0.copy(), bins_pe=None,
                                   metadata={'reference_point': 'first-entry'})

        def bins(self, ids, clock, edges):
            binned.extend(ids.tolist())
            np.testing.assert_array_equal(clock, origins[ids])
            return SimpleNamespace(charge_pe=q0[ids], bins_pe=b0[ids])

    monkeypatch.setattr('lighthit.ballistic_interval._prepare_ballistic_interval',
                        lambda *args: Prepared())
    response = joint_muon.transport_joint_muon(kernel, source, {}, preview=preview, **stores)
    assert calls['ballistic'] == 0
    assert binned == [0, 2]
    np.testing.assert_array_equal(response.charge_components_pe[:, 0], q0)
    np.testing.assert_array_equal(response.components_pe[[0, 2], :, 0], b0[[0, 2]])
    np.testing.assert_array_equal(response.components_pe[1, :, 0], 0.)
    assert response.metadata['receiver_reference_by_order'] == {
        '0': 'first-entry', '1': 'centre', '>=2': 'centre'}


def test_uncomputed_scatter_is_labelled_and_keeps_zero_frequency_charge(composed_case):
    kernel, source, preview, stores, origins, q0, b0, calls = composed_case
    response = joint_muon.transport_joint_muon(kernel, source, {}, preview=preview, **stores)
    assert response.metadata['fine_charge_mask'] == [True, False, True]
    assert response.metadata['inactive_order1_charge_may_be_coarse']
    assert response.metadata['inactive_ge2_charge_may_be_coarse']
    np.testing.assert_array_equal(response.charge_components_pe[1, 1:],
                                  preview.angular_scattered_pe[1])
    assert 'inactive_isotropic_acceptance_factor' not in response.metadata
    assert response.spectrum_ge2_pe[0, 1] == response.charge_components_pe[1, 2]
    np.testing.assert_array_equal(response.spectrum_ge2_pe[1:, 1], 0.)
    np.testing.assert_array_equal(response.components_pe[1, :, 1:], 0.)


def test_unselected_response_requires_directional_moments(composed_case):
    kernel, source, preview, stores, *_ = composed_case
    preview.angular_scattered_pe = None
    with pytest.raises(ValueError, match='arrival-direction moments'):
        joint_muon.transport_joint_muon(kernel, source, {}, preview=preview, **stores)


@pytest.mark.parametrize('grid', [np.array([0., .1]), np.array([0., .11, .2, .3])])
def test_primary_and_secondary_frequency_grids_cannot_silently_mix(composed_case, grid):
    kernel, source, preview, stores, origins, q0, b0, calls = composed_case
    stores['primary_ge2_store'] = SimpleNamespace(omega_per_ns=grid)
    with pytest.raises(ValueError, match='frequency grids'):
        joint_muon.transport_joint_muon(kernel, source, {}, preview=preview, **stores)
    assert calls['ballistic'] == 0
    assert not calls['primary'] and not calls['secondary']


def test_threshold_is_applied_after_primary_and_all_secondary_charges(composed_case):
    kernel, source, preview, stores, origins, q0, b0, calls = composed_case
    response = joint_muon.transport_joint_muon(kernel, source, {}, preview=preview, **stores)
    assert .004 < kernel.config.threshold_pe and .005 < kernel.config.threshold_pe
    assert response.active[0]
    assert response.charge_components_pe[0].sum() > kernel.config.threshold_pe


def test_primary_cannot_skip_ballistic_during_its_own_receiver_selection():
    from lighthit.experimental.universal_primary import transport_universal_primary
    kernel = SimpleNamespace(config=SimpleNamespace(threshold_pe=.01))
    with pytest.raises(ValueError, match='externally selected'):
        transport_universal_primary(kernel, None, None, None, skip_ballistic=True)
