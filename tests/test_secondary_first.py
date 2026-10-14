"""Numerical parity of the isolated secondary adapter and original callback."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import lighthit as lh
from lighthit import prompt
from lighthit.experimental.muon_components import MuonFineStrategy
from lighthit.experimental.muon_packets import SourcePacket
from lighthit.experimental.secondary_first import evaluate_secondary_first
from test_prompt import config, spectral_medium


def case(tmp_path):
    detector = lh.DetectorArray(np.array([[11., 4., 10.], [-9., 5., 12.], [700., 0., 10.]]),
        np.array([[0., 0., 1.], [1., .5, .3], [.1, -.4, 1.]]), .05,
        lambda c: 1.+.2*c+.1*c**2+.03*c**3,
        lambda w: .17+.11*(np.asarray(w)-350.)/260.)
    kernel = lh.TransportKernel(spectral_medium(), detector,
        config(tmp_path, wavelength_nodes=9, relative_time_edges_ns=np.arange(-20., 201., 10.)))
    phase = kernel.medium.sample(kernel.wavelength.wavelength_nm)['phase_index']
    beta = np.array([1., 1., .5*(1/phase.min()+1/phase.max()), .4, .92, 1.])
    direction = np.array([[0., 0., 1.], [1., 0., .4], [-.3, 1., .2],
                          [0., .4, 1.], [.2, -.7, 1.], [0., 1., .2]])
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    start = np.array([[1., 2., 0.], [0., 0., 1.], [.2, 0., 1.],
                      [-1., 1., 0.], [0., .5, 2.], [.1, .2, 1.]])
    times = np.array([0., 1., 2., 3., 4., 1e6])
    c0 = np.array([2e7, 1e7, 8e6, 5e6, 6e6, 7e6])
    source = lh.SpectralLightElements(start_m=start, direction=direction,
        length_m=np.full(6, .15), coefficient0=c0, coefficient2=-c0/beta**2,
        beta=beta, start_ns=times, end_ns=times+.15/.299792458,
        row_index=np.arange(6), uid=np.array([0, 1, 1, 1, 2, 3]), cone_model='spectral')
    packets = [SourcePacket('primary', np.array([0]), 0),
               SourcePacket('shower', np.array([3, 2, 1]), 1),
               SourcePacket('general', np.array([4]), 2),
               SourcePacket('late', np.array([5]), 3)]
    return kernel, source, packets


def test_matches_original_secondary_callback_all_spectral_bands(tmp_path, monkeypatch):
    kernel, source, packets = case(tmp_path)
    # Leave the original callback's denominator unchanged while omitting its
    # separately fitted primary response from the comparison.
    monkeypatch.setattr('lighthit.experimental.muon_components.fit_track_integrals',
        lambda primary, **kwargs: SimpleNamespace(elements=SimpleNamespace(beta=np.empty(0))))
    tables = prompt._spectral_tables(kernel)
    strategy = MuonFineStrategy(source, packets, phase_indices=tables['phase_index'])
    settings = lh.PromptConfig(element_method='grouped', screen=False, batch_charge_tolerance_pe=1e-6)
    ids = np.array([1, 0, 2]); origins = np.array([.7, -.3, 0.])
    positions = kernel.detector.positions_m[ids]
    looks = -kernel.detector.orientations[ids]
    areas = kernel.detector.effective_area_m2[ids]
    coef, ipow, jpow, degree = prompt.acceptance_monomials(kernel.acceptance()['alpha'])
    common = source.subset(source.beta*tables['phase_index'].max()>1.)
    oldq = np.zeros(3); oldb = np.zeros((3, len(kernel.config.relative_time_edges_ns)-1))
    for band, phase in enumerate(tables['phase_index']):
        part = common.subset(common.beta*phase>1.)
        part = replace(part, cone_model='frozen', reference_phase_index=float(phase))
        single = {key: value[band:band+1] for key, value in tables.items()}
        q, b, _ = strategy('grouped', kernel, source, part, prompt._element_segments(part),
            positions, looks, areas, origins, np.arange(3), single,
            coef, ipow, jpow, degree, kernel.medium.g, kernel.config.relative_time_edges_ns,
            replace(settings, batch_charge_tolerance_pe=1e-6/len(tables['wavelength_nm'])), np.zeros(3))
        oldq += q; oldb += b
    q, b, metadata = evaluate_secondary_first(kernel, source, packets, ids, origins, config=settings)
    np.testing.assert_allclose(q, oldq, rtol=3e-14, atol=1e-16)
    np.testing.assert_allclose(b, oldb, rtol=3e-14, atol=1e-16)
    assert q[0] > 0. and q[1] > 0. and q[0] != q[1]
    assert q[2] == 0. and not np.any(b[2])
    assert len(metadata['wavelengths']) == 9
    assert [row['packet'] for row in metadata['packets']] == [1, 2, 3]
    counts = [row['secondary_elements'] for row in metadata['wavelengths']]
    assert min(counts) == 3 and max(counts) == 4
    for band, row in enumerate(metadata['wavelengths']):
        valid = source.beta*tables['phase_index'][band]>1.
        shower = valid & (source.uid==1)
        expected = 1e-6/9*source.coefficient0[shower].sum()/source.coefficient0[valid].sum()
        actual = next(item for item in metadata['packets'][0]['wavelengths']
                      if item['wavelength_nm']==row['wavelength_nm'])
        assert actual['batch_charge_tolerance_pe'] == pytest.approx(expected, rel=2e-15)
    # An out-of-window source still has stationary charge; it supplies no bins.
    lateq, lateb, _ = evaluate_secondary_first(kernel, source, [packets[-1]], ids, origins, config=settings)
    assert np.all(lateq[:2]>0.)
    np.testing.assert_array_equal(lateb, 0.)


@pytest.mark.parametrize('indices', [[0, 0], [-1], [3], [0.5]])
def test_invalid_receivers_raise_before_indexing(tmp_path, indices):
    kernel, source, packets = case(tmp_path)
    with pytest.raises(ValueError, match='receiver indices'):
        evaluate_secondary_first(kernel, source, packets, indices, np.zeros(len(indices)))
