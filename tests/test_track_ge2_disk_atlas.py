"""Bounded-memory and charge-only contracts for the event-independent atlas."""
import json
import hashlib
import numpy as np
import pytest
from lighthit.experimental.track_ge2_primitive import (DiskTrackGe2Atlas,
    build_track_ge2_orientation_row,orientation_basis_looks,look_polynomials,track_ge2_axial_edges)


def make_atlas(tmp_path):
    omega=np.array([0.,.1]);rows=[]
    looks=orientation_basis_looks()
    for i,rho in enumerate([4.,5.,6.,7.,8.]):
        edges=np.linspace(-10,10,21)
        def evaluator(u):
            angular=look_polynomials(looks)@np.arange(1,11)
            return np.broadcast_to(angular[:,None,None],(10,len(u),2)).astype(complex)
        row=build_track_ge2_orientation_row(evaluator,edges,omega,order=8)
        np.savez(tmp_path/f'full{i}.npz',edges_m=edges,integrated_coefficients=row.integrated_coefficients,prefix=row.prefix)
        np.savez(tmp_path/f'charge{i}.npz',edges_m=edges,
            integrated_coefficients=row.integrated_coefficients.reshape(20,9,10,2)[...,0],
            prefix=row.prefix.reshape(21,10,2)[...,0])
        rows.append(dict(impact_m=rho,file=f'full{i}.npz',charge_file=f'charge{i}.npz',normalization=1.,sha256=hashlib.sha256((tmp_path/f'full{i}.npz').read_bytes()).hexdigest(),
            charge_sha256=hashlib.sha256((tmp_path/f'charge{i}.npz').read_bytes()).hexdigest()))
    manifest=dict(schema='lighthit/track-ge2-disk-atlas/1',status='complete',validation=[dict(low_m=4.,high_m=8.,passed=True)],metadata={'clock_ns_per_m':1/.299792458},
                  omega_per_ns=omega.tolist(),front_ns_per_m=0.,rows=rows)
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return DiskTrackGe2Atlas(tmp_path)


def test_charge_query_does_not_load_frequency_rows(tmp_path):
    atlas=make_atlas(tmp_path)
    look=np.array([.6,0,.8]);q=atlas.segment_charge(5.5,look,3.,-2.)
    assert atlas.full_row_loads==0
    assert not atlas._cache
    full=atlas.segment(5.5,look,3.,-2.)
    assert q==pytest.approx(full[0].real,rel=1e-14)
    assert atlas.full_row_loads==3
    assert atlas.segment_charge(5.5,look,3.,-2.,closest_time_ns=50.)==q


def test_lru_and_no_extrapolation(tmp_path):
    atlas=make_atlas(tmp_path);look=np.array([0.,0.,1.])
    atlas.segment(4.2,look,3.,-2.);atlas.segment(7.8,look,3.,-2.)
    assert len(atlas._cache)==3
    assert atlas.full_row_loads==5
    with pytest.raises(ValueError,match='fallback'):atlas.segment_charge(3.9,look,3.,-2.)


def test_scaled_axial_grid_is_local_sphere():
    for rho in [3.01,25.,100.,299.]:
        edges=track_ge2_axial_edges(rho)
        assert np.max(np.diff(edges))<=4.+1e-12
        assert np.all(np.diff(edges)>0)
        np.testing.assert_allclose(rho*rho+edges[[0,-1]]**2,300.**2,rtol=2e-15)
    with pytest.raises(ValueError):track_ge2_axial_edges(300.)


def test_batch_charge_matches_scalar_without_full_loads(tmp_path):
    atlas=make_atlas(tmp_path)
    rho=np.array([4.2,5.5,7.8]);look=np.tile([.6,0,.8],(3,1))
    a=np.array([3.,2.,7.]);b=np.array([-2.,-1.,0.])
    result=atlas.segment_charge_many(rho,look,a,b)
    reference=np.array([atlas.segment_charge(r,l,x,y) for r,l,x,y in zip(rho,look,a,b)])
    np.testing.assert_allclose(result,reference,rtol=5e-14,atol=1e-12)
    assert atlas.full_row_loads==0


def test_incomplete_manifest_and_corrupted_row_rejected(tmp_path):
    atlas=make_atlas(tmp_path)
    manifest=json.loads((tmp_path/'manifest.json').read_text());manifest['status']='building'
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='incomplete'):DiskTrackGe2Atlas(tmp_path)
    with (tmp_path/'full0.npz').open('ab') as stream:stream.write(b'corruption')
    with pytest.raises(ValueError,match='checksum'):atlas.segment(4.2,np.array([0.,0.,1.]),3.,-2.)


def test_fused_orientation_contraction_matches_generic_spectral_apply():
    from lighthit.experimental.track_ge2_primitive import cone_point_evaluator
    rng=np.random.default_rng(123)
    pairs=[(l,m) for l in range(4) for m in range(l+1)]
    shape=(2,3,4,10,4,3)
    coefficients=(rng.normal(size=shape)+1j*rng.normal(size=shape))*1e-3
    bundle=dict(degree=3,acceptance_degree=3,omega=np.array([0.,.2,.8]),
        pair_lambda=np.array([p[0] for p in pairs]),pair_mu=np.array([p[1] for p in pairs]),
        log_r=np.log([3.,10.,100.,300.]),coefficients=coefficients,
        absorption=np.array([.05,.03]),speed=np.array([.22,.225]),phase='flight')
    looks=np.array([[0.,0.,1.],[.6,0.,.8],[-.5,np.sqrt(.5),.5]])
    kwargs=dict(impact_m=7.,look=looks,alpha=np.array([4.,2.,.3,-.04]),wavelength_nm=[400.,500.],
        wavelength_weight_nm=[100.,100.],phase_index=[1.35,1.33],efficiency=[.2,.1],beta=1.)
    a=cone_point_evaluator(bundle,**kwargs)
    b=cone_point_evaluator(bundle,backend='fused-looks',**kwargs)
    u=np.array([-50.,-3.,0.,10.,250.])
    np.testing.assert_allclose(a(u),b(u),rtol=2e-13,atol=1e-22)


def test_weak_backward_region_falls_back_without_disabling_charge(tmp_path):
    atlas=make_atlas(tmp_path)
    for item in atlas.manifest['validation']:
        item.update(front_passed=True,charge_passed=True,backward_passed=False)
    look=np.array([0.,0.,1.])
    assert atlas.query_supported(5.5,2.,-3.)
    assert not atlas.query_supported(5.5,-1.,-3.)
    assert atlas.query_supported(5.5,-1.,-3.,charge_only=True)
    with pytest.raises(ValueError,match='region failed validation'):
        atlas.segment(5.5,look,-1.,-3.)
    assert np.isfinite(atlas.segment_charge(5.5,look,-1.,-3.))
