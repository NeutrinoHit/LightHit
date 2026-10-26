"""Portable cache preparation, interruption handling and publication gates."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/tables'))
import build_receiver_bundle as bundle
from test_model_file import model as example_model
from lighthit.receiver import SphereSurface


@pytest.fixture
def model_path(tmp_path):
    m=example_model();m=replace(m,detector=replace(m.detector,receiver_surface=SphereSurface(.216)))
    p=tmp_path/'model.npz';m.save(p);return p


def test_plan_is_read_only_and_keeps_all_fast_components(model_path,tmp_path,capsys):
    root=tmp_path/'output'
    assert bundle.main(['--model',str(model_path),'--output',str(root),'--jobs','3','--plan'])==0
    plan=json.loads(capsys.readouterr().out);assert not root.exists()
    names=[s['name'] for s in plan['stages']]
    assert names.index('q1-reference')<names.index('q1-adaptive')
    assert names.index('fast-packs')<names.index('primary-ge2')
    assert {'time-beta1','time-beta0999','glass-cubature','surface-preview','time-validation'}<=set(names)
    cmd=next(s['commands'][0] for s in plan['stages'] if s['name']=='fast-packs')
    assert cmd[cmd.index('--manifest-name')+1]=='shower-data.pending.json'
    for stage in plan['stages']:
        for command in stage['commands']:
            assert Path(command[1]).is_file()
            assert '--model' in command
    axes=bundle._grids(.216,1000.)
    assert len(axes)==308 and axes[0]>.216 and axes[-1]==1000.
    assert len(bundle.TIME_RADII)==25 and 5.75 in bundle.TIME_RADII and 6.5 in bundle.TIME_RADII


def test_real_subprocess_resume_and_failed_stage_do_not_publish(tmp_path):
    root=tmp_path/'out';a=root/'a.txt';b=root/'b.txt';ready=root/'ready'
    first=bundle.Stage('a',(a,),[[sys.executable,'-c',f"from pathlib import Path;Path({str(a)!r}).write_text('a')"]])
    failed=bundle.Stage('b',(b,),[[sys.executable,'-c','raise SystemExit(3)']])
    engine=bundle.Builder(root,{'physics':'same'})
    with pytest.raises(subprocess.CalledProcessError):engine.run([first,failed],finalize=lambda:ready.touch())
    assert not ready.exists()
    state=json.loads((root/'.prepare/build.json').read_text());assert set(state['completed'])=={'a'}
    stamp=a.stat().st_mtime_ns
    second=bundle.Stage('b',(b,),action=lambda:b.write_text('b'))
    engine.run([first,second],finalize=lambda:ready.touch())
    assert ready.exists() and a.stat().st_mtime_ns==stamp
    assert (root/'.prepare/logs/b.log').exists()


def test_resume_refuses_changed_model_and_preserves_existing_files(tmp_path):
    root=tmp_path/'out';p=root/'data';stage=bundle.Stage('data',(p,),action=lambda:p.write_text('science'))
    bundle.Builder(root,{'model':'one'}).run([stage])
    with pytest.raises(ValueError,match='changed'):
        bundle.Builder(root,{'model':'two'}).run([stage])
    assert p.read_text()=='science'
    p.write_text('changed')
    with pytest.raises(ValueError,match='changed|truncated'):
        bundle.Builder(root,{'model':'one'}).run([stage])


def test_worker_count_can_change_without_rebuilding_completed_science(tmp_path):
    root=tmp_path/'out';p=root/'science';calls=[]
    def run(command,log):calls.append(command);p.write_text('result')
    stage=lambda jobs:bundle.Stage('same',(p,),[['program','--jobs',str(jobs)]])
    engine=bundle.Builder(root,{},runner=run);engine.run([stage(1)]);engine.run([stage(3)])
    assert len(calls)==1


def test_component_validator_gates_checkpoint_and_final_publication(tmp_path):
    root=tmp_path/'out';p=root/'unvalidated';ready=root/'ready'
    def reject():raise ValueError('physical check failed')
    stage=bundle.Stage('check',(p,),action=lambda:p.write_text('candidate'),validate=reject)
    with pytest.raises(ValueError,match='physical check'):bundle.Builder(root,{}).run([stage],finalize=lambda:ready.touch())
    assert not ready.exists() and not json.loads((root/'.prepare/build.json').read_text())['completed']


def test_output_lock_covers_finalization_and_is_released_after_failure(tmp_path):
    root=tmp_path/'out';p=root/'data'
    stage=bundle.Stage('data',(p,),action=lambda:p.write_text('x'))
    def finalize():
        with pytest.raises(RuntimeError,match='another builder'):
            with bundle.build_lock(root/'.prepare'):pass
        raise RuntimeError('interrupted final validation')
    with pytest.raises(RuntimeError,match='interrupted'):bundle.Builder(root,{}).run([stage],finalize=finalize)
    with bundle.build_lock(root/'.prepare'):pass


def test_inventory_verifies_portable_content_and_rejects_escaping_paths(tmp_path):
    root=tmp_path/'one';root.mkdir();(root/'data').write_text('abc')
    inv=bundle.inventory(root,[root/'data']);moved=tmp_path/'two';shutil.move(root,moved)
    bundle.verify_inventory(moved,inv)
    with pytest.raises(ValueError,match='escapes'):bundle.verify_inventory(moved,{'../outside':{'size':1,'sha256':''}})
    outside=tmp_path/'outside';outside.write_text('secret');(moved/'link').symlink_to(outside)
    with pytest.raises(ValueError,match='escapes'):bundle.inventory(moved,[moved/'link'])


def test_original_bundle_is_not_overwritten(tmp_path):
    root=tmp_path/'existing';root.mkdir();p=root/'model.npz';p.write_bytes(b'original')
    with pytest.raises(FileExistsError):bundle.Builder(root,{}).run([])
    assert p.read_bytes()==b'original'


def _candidate(root,model_path):
    from lighthit.receiver_data import encode_settings
    root.mkdir();(root/'model').mkdir();shutil.copyfile(model_path,root/'model/bgvd-model.npz')
    tables=root/'tables';muon=tables/'muon';muon.mkdir(parents=True)
    for p in ['rte-cache','ge2-pack','ge2-band-fold','first-entry','muon/q1-atlas','muon/ge2-atlas']:(tables/p).mkdir()
    for p in ['first-beta/first-beta-charge.npz','ge2-beta/beta-charge.npz','surface-preview/surface-preview.npz','preview-proposal.npz','first-time-beta.json','surface-rule.npz']:
        p=muon/p;p.parent.mkdir(exist_ok=True);p.write_text('fixture')
    settings={'scattered_receiver':'first-entry','first_entry_tables':tables/'first-entry'}
    bundle.atomic_json(tables/'shower-data.pending.json',dict(rte_cache='rte-cache',ge2_pack='ge2-pack',ge2_band_fold='ge2-band-fold',kernel_settings=encode_settings(settings,tables)))


def test_real_manifests_publish_after_component_checks_and_work_after_move(tmp_path,model_path,monkeypatch):
    import lighthit.shower,lighthit.muon,lighthit.receiver_prompt
    root=tmp_path/'cache';_candidate(root,model_path);calls=[]
    class Engine:
        def __init__(self,kernel,data):self.kernel=kernel
        def tables(self):calls.append('tables')
    monkeypatch.setattr(lighthit.shower,'ShowerTransport',Engine)
    monkeypatch.setattr(lighthit.muon,'MuonTransport',Engine)
    monkeypatch.setattr(lighthit.receiver_prompt,'FirstEntryPrompt',lambda k:calls.append('geometry'))
    bundle.publish(root);assert calls==['tables','geometry','tables']
    bundle.atomic_json(root/'bundle.json',dict(schema=bundle.RECIPE,complete=True,files=bundle.inventory(root,[root/'model',root/'tables'])))
    moved=tmp_path/'relocated';shutil.move(root,moved);bundle.check(moved)
    data=lighthit.muon.MuonData.load(moved/'tables/muon-data.json')
    assert all(Path(p).is_relative_to(moved) for p in [data.rte_cache,data.preview_atlas,data.primary_first_time_beta,data.primary_first_cubature,data.kernel_settings['first_entry_tables']])


def test_publication_failure_leaves_no_final_runtime_manifests(tmp_path,model_path,monkeypatch):
    import lighthit.shower,lighthit.muon,lighthit.receiver_prompt
    root=tmp_path/'cache';_candidate(root,model_path)
    class Good:
        def __init__(self,kernel,data):self.kernel=kernel
        def tables(self):pass
    class Bad(Good):
        def tables(self):raise ValueError('optics mismatch')
    monkeypatch.setattr(lighthit.shower,'ShowerTransport',Good);monkeypatch.setattr(lighthit.muon,'MuonTransport',Bad)
    monkeypatch.setattr(lighthit.receiver_prompt,'FirstEntryPrompt',lambda k:None)
    with pytest.raises(ValueError,match='optics mismatch'):bundle.publish(root)
    assert not (root/'tables/shower-data.json').exists() and not (root/'tables/muon-data.json').exists()


def test_zero_support_row_check_is_finite_without_relaxing_nonzero_checks(monkeypatch):
    import build_track_first_atlas
    import lighthit.experimental.track_first_primitive as primitive
    monkeypatch.setattr(primitive,'build_track_first_primitive',lambda *a,**k:SimpleNamespace(charge_cells=np.array([0.]),bins_cells=np.zeros((1,162)),metadata={'build_cpu_s':0.}))
    result=build_track_first_atlas.check_geometry(None,lambda *args:(0.,np.zeros(162)),300.,1.)
    assert all(r['bins_l1_relative']==0 and r['charge_relative_error']==0 for r in result)
    with pytest.raises(ValueError,match='outside'):
        build_track_first_atlas.check_geometry(None,lambda *args:(0.,np.ones(162)),300.,1.)


def test_model_based_optics_loader_does_not_import_private_provider(tmp_path,model_path,monkeypatch):
    import lighthit as lh
    import build_track_ge2_atlas
    from lighthit.shower_basis import Ge2KernelStore
    k=lh.load_model(model_path).kernel(lh.KernelConfig(omega_per_ns=np.linspace(0.,1.2,65)))
    keys=tuple('band-'+str(j) for j in range(32));cache=tmp_path/'rte';cache.mkdir()
    for key,w in zip(keys,k.wavelength.wavelength_nm):
        np.savez_compressed(cache/('directional-'+key+'.npz'),metadata=json.dumps({'medium':dict(wavelength_nm=float(w),**{name:float(value) for name,value in k.medium.sample(float(w)).items()},g=k.medium.g)}))
    pack={'omega':k.config.omega_per_ns,'degree':32,'acceptance_degree':3,'log_r':np.log([3.,300.])}
    monkeypatch.setattr(Ge2KernelStore,'load',lambda *a,**k:SimpleNamespace(_bundles={(keys,32,tuple(range(65))):pack}))
    monkeypatch.setitem(sys.modules,'bgvd_model',None)
    kernel,_,_=build_track_ge2_atlas.load_optics(tmp_path/'pack',cache,model_path)
    np.testing.assert_array_equal(kernel.medium.absorption_per_m,k.medium.absorption_per_m)


def test_checkpoint_failure_never_exposes_partial_arrays(tmp_path,monkeypatch):
    import _atomic
    old=tmp_path/'old.npz';fresh=tmp_path/'fresh.npz'
    _atomic.savez(old,value=np.array([3.,4.]))
    def fail(path,**arrays):
        Path(path).write_bytes(b'partial')
        raise OSError('interrupted write')
    monkeypatch.setattr(_atomic.np,'savez_compressed',fail)
    for path in [old,fresh]:
        with pytest.raises(OSError,match='interrupted'):_atomic.savez(path,value=np.ones(1))
    assert not fresh.exists()
    with np.load(old) as data:np.testing.assert_array_equal(data['value'],[3.,4.])


def test_interrupted_temporary_files_are_not_runtime_products(tmp_path):
    root=tmp_path/'cache';root.mkdir();(root/'preview-proposal.npz').write_bytes(b'complete')
    partial=root/'table-orphan';partial.mkdir();(partial/'checkpoint.npz').write_bytes(b'incomplete')
    (root/'other.tmp.npz').write_bytes(b'incomplete')
    assert set(bundle.inventory(root,[root]))=={'preview-proposal.npz'}


def test_real_bootstrap_needs_no_preexisting_cache(model_path,tmp_path):
    root=tmp_path/'new';stages=bundle.preparation(model_path,root,2)
    bundle.Builder(root,{'model':bundle.digest(model_path)}).run(stages[:1])
    assert bundle.digest(root/'model/bgvd-model.npz')==bundle.digest(model_path)
    assert len(np.load(root/'.prepare/radii.npy'))==308
    assert len(np.load(root/'.prepare/primary-radii.npy'))==328
    np.testing.assert_array_equal(np.load(root/'.prepare/primary-radii.npy')[:308],np.load(root/'.prepare/radii.npy'))
    assert json.loads((root/'.prepare/fold-template.json').read_text())['band_fold']['bands']==[10,16,21]
    assert not (root/'bundle.json').exists()
