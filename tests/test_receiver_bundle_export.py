"""Runtime dependency closure, relocation and numerical/MC check separation."""
from pathlib import Path
from types import SimpleNamespace
import json,os,sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/tables'));sys.path.insert(0,str(ROOT/'experiments'))
import export_receiver_bundle as export
# The checker pins one thread for its own command line; importing it must not
# change the thread settings of the test session after Numba has started.
_THREADS={name:os.environ.get(name) for name in ('NUMBA_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}
import check_receiver_bundle as checker
for _name,_value in _THREADS.items():
    if _value is None:os.environ.pop(_name,None)
    else:os.environ[_name]=_value


def _write(path,value=b'data'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)) if isinstance(value,dict) else path.write_bytes(value)
    return path


def _source(root):
    model=_write(root/'model.npz');table=root/'original';table.mkdir()
    first=root/'q1-root'
    for name in ['q1-reference','q1-adaptive','geometry','geometry-primary','q1-bands']:
        _write(first/name/'manifest.json',{'schema':'fixture'});_write(first/name/'row.npy')
    for key in ['a','b']:_write(table/f'rte-cache/directional-{key}.npz')
    _write(table/'full/array.npy',b'physical array');_write(table/'pack/charge.npy',b'charge')
    _write(table/'pack/pack.json',dict(schema='fixture',bundles=[dict(file='../full/array.npy',key=[['a','b'],32,[0,1]],meta={'omega':[0,.1]}),dict(file='charge.npy',key=[['a','b'],32,[0]],meta={'omega':[0]})]))
    for name in ['folded.json','folded.npy']:_write(table/'fold'/name)
    def atlas(path):
        _write(path/'manifest.json',{'status':'complete','rows':[{'file':'rows/row.npz','charge_file':'rows/charge.npz'}]})
        _write(path/'rows/row.npz');_write(path/'rows/charge.npz')
    for name in ['q1','ge2','anchor1','anchor0999']:atlas(table/name)
    for name in ['beta1.npz','beta2.npz','preview.npz','proposal.npz','cube.npz']:_write(table/name)
    _write(table/'time.json',dict(schema='lighthit/first-time-beta/1',validation_passed=True,nodes=[dict(beta=b,directory=d,manifest_sha256=export.digest(table/d/'manifest.json')) for b,d in [(1.,'anchor1'),(.999,'anchor0999')]]))
    common=dict(rte_cache='rte-cache',ge2_pack='pack',ge2_band_fold='fold',kernel_settings=dict(scattered_receiver='first-entry',first_entry_tables='../q1-root'))
    shower=_write(table/'shower-data.json',common)
    muon=_write(table/'muon-data.json',dict(common,primary_first_atlas='q1',primary_ge2_atlas='ge2',primary_first_beta='beta1.npz',primary_ge2_beta='beta2.npz',preview_atlas='preview.npz',preview_proposal_atlas='proposal.npz',primary_first_time_beta='time.json',primary_first_cubature='cube.npz'))
    return model,shower,muon


def test_export_collects_external_pack_array_and_preserves_primary_manifest_hashes(tmp_path):
    source=tmp_path/'source';source.mkdir();args=_source(source);target=tmp_path/'output'
    task=export.Export(source,target);task.plan(*args)
    assert not target.exists()
    assert task.files['tables/ge2-pack/array-000.npy']==(source/'original/full/array.npy').resolve()
    assert all('..' not in b['file'] for b in task.documents['tables/ge2-pack/pack.json']['bundles'])
    time=task.documents['tables/muon/first-time-beta.json']
    for node in time['nodes']:
        copied=task.files['tables/muon/'+node['directory']+'/manifest.json']
        assert export.digest(copied)==node['manifest_sha256']
    assert task.documents['tables/shower-data.pending.json']['kernel_settings']['first_entry_tables']=='first-entry'


def test_export_failure_does_not_publish_and_resume_retains_completed_copies(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();args=_source(source);target=tmp_path/'output'
    task=export.Export(source,target);provenance=task.plan(*args)
    def fail(root):raise ValueError('component check failed')
    monkeypatch.setattr(export,'publish',fail)
    with pytest.raises(ValueError,match='component'):task.run(provenance)
    assert not (target/'bundle.json').exists()
    stamp=(target/'tables/ge2-pack/array-000.npy').stat().st_mtime_ns
    with pytest.raises(ValueError,match='component'):task.run(provenance)
    assert (target/'tables/ge2-pack/array-000.npy').stat().st_mtime_ns==stamp
    assert (target/'tables/ge2-pack/array-000.npy').read_bytes()==b'physical array'


def test_source_path_cannot_escape_explicit_export_boundary(tmp_path):
    source=tmp_path/'source';source.mkdir();args=_source(source)
    _write(tmp_path/'outside.npy',b'private')
    pack=source/'original/pack/pack.json';m=json.loads(pack.read_text());m['bundles'][0]['file']='../../../outside.npy';pack.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='escapes'):export.Export(source,tmp_path/'output').plan(*args)


def test_numerical_copy_check_is_separate_from_retained_mc_tail_flags(tmp_path):
    charge=np.array([[.002282829,.005067886,.012557503]]);bins=charge[:,None,:]
    response=SimpleNamespace(charge_components_pe=charge,components_pe=bins,active=np.array([False]),time_origin_ns=np.array([0.]))
    expected=dict(charge_components_pe=charge.copy(),components_pe=bins.copy(),active=response.active,time_origin_ns=response.time_origin_ns)
    assert checker.compare(response,expected,rtol=1e-6,atol=1e-8)['passed']
    source=SimpleNamespace(start_m=np.zeros((1,3)),length_m=np.ones(1),direction=np.array([[0.,0.,1.]]))
    model=SimpleNamespace(detector=SimpleNamespace(positions_m=np.array([[125.,0.,0.]])))
    mc=[.0019037795,.0049737968,.0112011941,.0180787704];sigma=[.000234,.0003303,.00040903,.00034031]
    row=dict(distance_m=[100.,150.],oms=1)
    for i,key in enumerate(['0','1','>=2','total']):row[key]=dict(mc_pe=mc[i],mc_sigma_pe=sigma[i],mc_share=mc[i]/4.55078)
    actual=checker.mc_groups(source,model,charge,dict(metrics=dict(by_distance=[row])))[0]['orders']
    assert not actual['total']['passed'] and not actual['>=2']['passed'] and actual['1']['passed']
    assert actual['total']['significant']
    broken=dict(expected,components_pe=np.zeros_like(bins))
    assert not checker.compare(response,broken,rtol=1e-6,atol=1e-8)['passed']
