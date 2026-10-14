"""Lazy universal cache IO is independent of detector/event geometry."""
from pathlib import Path
import json,hashlib
import numpy as np
import pytest
from lighthit.experimental.track_first_primitive import TrackFirstAtlasStore,orientation_nodes


def store_files(folder):
    radii=np.array([2.,3.,4.]);x=np.array([-2.,-1.,0.,2.]);t=np.array([-1.,0.,1.,3.])
    _,powers,_=orientation_nodes()
    ql=[];qr=[];records=[]
    for radius in radii:
        charge=np.zeros((4,10));charge[:,0]=(x-x[0])*np.exp(-.1*radius)
        right=charge[-1]-charge
        cdf=charge[:,None,:]*np.array([0.,.1,.5,.8])[None,:,None]
        cdf_right=right[:,None,:]*np.array([0.,.1,.5,.8])[None,:,None]
        path=folder/f'row{radius}.npz'
        np.savez_compressed(path,cdf_left=cdf,cdf_right=cdf_right)
        records.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        ql.append(charge);qr.append(right)
    path=folder/'charges.npz';np.savez_compressed(path,charge_left=np.array(ql),charge_right=np.array(qr))
    manifest=dict(status='complete',physics=dict(beta=1.,speed_m_per_ns=.3),radii_m=radii.tolist(),
        x_edges=x.tolist(),time_over_rho_edges=t.tolist(),powers=powers.tolist(),rows=records,
        charge_file=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_charge_queries_do_not_load_any_cdf_and_match_full_response(tmp_path):
    store_files(tmp_path);store=TrackFirstAtlasStore(tmp_path)
    look=np.array([.3,.4,np.sqrt(.75)])
    for rho in np.linspace(2.,4.,50):
        q=store.query_charge(rho,look,-rho,1.2*rho)
        np.testing.assert_allclose(q,np.exp(-.1*rho)*2.2)
    assert store.time_row_loads==0 and len(store._rows)==0
    rho=2.5;q=store.query_charge(rho,look,-rho,1.2*rho)
    full,bins=store.query(rho,look,-rho,1.2*rho,np.array([-.5,.5,2.])*rho)
    np.testing.assert_allclose(q,full,rtol=1e-14)
    assert store.time_row_loads==2 and len(store._rows)==2
    store.query(3.5,look,-3.5,3.5,[-1.,1.,5.])
    assert store.time_row_loads==3 and len(store._rows)==2
    assert list(store._rows)==[1,2]


def test_charge_query_survives_missing_time_rows_but_time_query_reports_error(tmp_path):
    manifest=store_files(tmp_path);store=TrackFirstAtlasStore(tmp_path)
    (tmp_path/manifest['rows'][0]['file']).unlink()
    assert store.query_charge(2.5,[0.,0.,1.],-1.,1.)>0
    with pytest.raises(FileNotFoundError):store.query(2.5,[0.,0.,1.],-1.,1.,[-1.,1.])


def test_changed_payload_and_incomplete_manifest_are_rejected(tmp_path):
    manifest=store_files(tmp_path)
    path=tmp_path/manifest['rows'][0]['file'];path.write_bytes(path.read_bytes()+b'changed')
    store=TrackFirstAtlasStore(tmp_path)
    with pytest.raises(ValueError,match='checksum'):
        store.query(2.5,[0.,0.,1.],-1.,1.,[-1.,1.])
    manifest['status']='building';(tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='incomplete'):TrackFirstAtlasStore(tmp_path)


def test_causal_zero_extension_precedes_stored_time_axis_without_late_extrapolation(tmp_path):
    manifest=store_files(tmp_path)
    manifest['physics']['causal_zero_before_tau0']=True
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    store=TrackFirstAtlasStore(tmp_path)
    q,b=store.query(2.5,[0.,0.,1.],-1.,1.,[-100.,-10.,-3.,-.5,1.])
    assert b[0]==0 and b[1]==0
    with pytest.raises(ValueError,match='time edges'):
        store.query(2.5,[0.,0.,1.],-1.,1.,[-100.,100.])


def test_optical_hash_ignores_geometry_but_detects_qe_and_area_changes():
    from types import SimpleNamespace
    from copy import copy
    from lighthit.experimental.track_first_primitive import track_first_optical_contract
    def sample(w):
        return {key:np.full_like(w,value) for key,value in dict(absorption_per_m=.05,
            scattering_per_m=.015,phase_index=1.34,group_index=1.37).items()}
    def efficiency(w):return np.ones_like(w)
    efficiency.wavelength_knots_nm=np.array([350.,450.,610.])
    detector=SimpleNamespace(spectral_efficiency=efficiency,spectral_weight=lambda w:1.,
        effective_area_m2=np.array([1.]),positions_m=np.array([[0.,0.,0.]]),orientations=np.array([[0.,0.,1.]]))
    kernel=SimpleNamespace(medium=SimpleNamespace(g=.9,sample=sample,wavelength_nm=np.array([350.,450.,610.]),
        wavelength_range_nm=(350.,610.)),detector=detector,config=SimpleNamespace(wavelength_range_nm=(350.,610.)),
        acceptance=lambda:{'alpha':np.array([4*np.pi])})
    _,original=track_first_optical_contract(kernel)
    other=copy(kernel);other.detector=copy(detector)
    other.detector.positions_m=np.array([[500.,100.,-30.]])
    other.detector.orientations=np.array([[1.,0.,0.]])
    assert track_first_optical_contract(other)[1]==original
    other.detector.effective_area_m2=np.array([2.])
    assert track_first_optical_contract(other)[1]!=original
    other.detector.effective_area_m2=np.array([1.]);other.detector.spectral_weight=lambda w:2.
    assert track_first_optical_contract(other)[1]!=original


def test_vector_charge_queries_match_scalar_and_never_load_time_rows(tmp_path):
    store_files(tmp_path);store=TrackFirstAtlasStore(tmp_path)
    rng=np.random.default_rng(821)
    rho=rng.uniform(2.,4.,500);look=rng.normal(size=(500,3));look/=np.linalg.norm(look,axis=1)[:,None]
    low=rng.uniform(-2.,.2,500)*rho;high=rng.uniform(.3,2.,500)*rho
    density=rng.uniform(0.,2.,500)
    actual=store.query_charge_many(rho,look,low,high,coefficient0_per_m=density)
    expected=np.array([store.query_charge(r,n,a,b,coefficient0_per_m=c) for r,n,a,b,c in zip(rho,look,low,high,density)])
    np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=1e-15)
    assert store.time_row_loads==0
    with pytest.raises(ValueError,match='radius outside'):
        store.query_charge_many(np.array([2.5,5.]),look[:2],[-1.,-1.],[1.,1.])
    with pytest.raises(ValueError,match='source interval'):
        store.query_charge_many(np.array([2.5,3.]),look[:2],[-100.,-1.],[1.,1.])
    assert store.query_charge_many(np.array([]),np.empty((0,3)),[],[]).shape==(0,)


def test_edge_times_radius_roundoff_is_accepted_by_all_three_queries(tmp_path):
    manifest=store_files(tmp_path)
    manifest['x_edges']=[-8.5,-1.,0.,8.5]
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    store=TrackFirstAtlasStore(tmp_path)
    rho=np.random.default_rng(4281).uniform(2.,4.,500)
    low=-8.5*rho;high=8.5*rho
    # Reproduce the actual multiply/divide failure, not an invented epsilon.
    assert np.any(low/rho < -8.5) and np.any(high/rho > 8.5)
    looks=np.tile([0.,0.,1.],(len(rho),1))
    batch=store.query_charge_many(rho,looks,low,high)
    expected=[]
    for r,a,b in zip(rho,low,high):
        q=store.query_charge(r,[0.,0.,1.],a,b)
        full,bins=store.query(r,[0.,0.,1.],a,b,[-1.,1.])
        np.testing.assert_allclose(q,full,rtol=1e-14)
        expected.append(q)
    np.testing.assert_allclose(batch,expected,rtol=1e-14)
    # A genuine 1e-8 excursion in normalized source coordinate still fails.
    for a,b in [((-8.5-1e-8)*rho[0],high[0]),(low[0],(8.5+1e-8)*rho[0])]:
        with pytest.raises(ValueError,match='source interval'):
            store.query_charge(rho[0],[0.,0.,1.],a,b)
        with pytest.raises(ValueError,match='source interval'):
            store.query(rho[0],[0.,0.,1.],a,b,[-1.,1.])
        with pytest.raises(ValueError,match='source interval'):
            store.query_charge_many(rho[:1],looks[:1],[a],[b])
