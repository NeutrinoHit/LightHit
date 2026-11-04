"""Independent radius subprocesses join only checked, compatible checkpoints."""
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/tables'))
from build_track_first_atlas import _parallel_rows


@pytest.fixture
def child(tmp_path):
    script=tmp_path/'radius.py'
    script.write_text('''import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--radii',nargs='+',type=float);p.add_argument('--build-radii',type=float);a,_=p.parse_known_args()
r=a.build_radii
if (Path(__file__).parent/'fail').exists() and r==5:raise SystemExit(7)
a.output.mkdir(parents=True,exist_ok=True);name='row-'+str(r)+'.npz';path=a.output/name
np.savez(path,values=np.array([r,r*r]))
sha=hashlib.sha256(path.read_bytes()).hexdigest()
if (Path(__file__).parent/'corrupt').exists():sha='bad'
m=dict(optics_grid_hash='contract',radii_m=a.radii,rows=[dict(rho_m=r,file=name,sha256=sha)],row_checks={str(r):[dict(passed=True)]})
(a.output/'manifest.json').write_text(json.dumps(m))
''')
    return script


def state(tmp_path,workers=2):
    out=tmp_path/'atlas';out.mkdir()
    args=SimpleNamespace(output=out,radius_workers=workers,workers=1,model=None,first_entry_tables=None,
        beta=1.,cells=96,time_step=.1,max_time_ns=900.,check_rows=True)
    manifest=dict(rows=[],row_checks={});records={};path=out/'manifest.json'
    return args,manifest,records,path


def test_real_radius_subprocesses_join_in_radius_order_and_preserve_completed_rows(tmp_path,child):
    args,m,records,path=state(tmp_path)
    radii=np.array([3.,5.])
    _parallel_rows(args,radii,radii,'contract',m,records,path,script=child)
    assert [r['rho_m'] for r in json.loads(path.read_text())['rows']]==[3.,5.]
    for r,record in records.items():
        with np.load(args.output/record['file']) as d:np.testing.assert_array_equal(d['values'],[r,r*r])
    stamps={r:(args.output/v['file']).stat().st_mtime_ns for r,v in records.items()}
    args.radius_workers=1
    _parallel_rows(args,radii,radii,'contract',m,records,path,script=child)
    assert stamps=={r:(args.output/v['file']).stat().st_mtime_ns for r,v in records.items()}


def test_failed_radius_keeps_other_committed_rows_for_resume(tmp_path,child):
    args,m,records,path=state(tmp_path,1);radii=np.array([3.,5.])
    marker=tmp_path/'fail';marker.touch()
    with pytest.raises(RuntimeError,match='radius 5.0 build failed'):
        _parallel_rows(args,radii,radii,'contract',m,records,path,script=child)
    assert set(records)=={3.} and len(json.loads(path.read_text())['rows'])==1
    original=(args.output/records[3.]['file']).read_bytes();marker.unlink()
    _parallel_rows(args,radii,radii,'contract',m,records,path,script=child)
    assert set(records)=={3.,5.} and (args.output/records[3.]['file']).read_bytes()==original


def test_corrupt_radius_payload_is_not_joined(tmp_path,child):
    args,m,records,path=state(tmp_path);radii=np.array([3.,5.]);(tmp_path/'corrupt').touch()
    with pytest.raises(ValueError,match='checksum'):
        _parallel_rows(args,radii,radii,'contract',m,records,path,script=child)
    assert not records and not list(args.output.glob('row-*.npz'))


def test_stored_rows_retain_identical_npy_members_and_load_without_inflation(tmp_path):
    import zipfile
    from build_track_first_atlas import _write_row
    row=dict(cdf_left=np.arange(240.,dtype=float).reshape(4,6,10),
        cdf_right=np.linspace(0.,1.,240).reshape(4,6,10),charge_left=np.ones((4,10)),charge_right=np.zeros((4,10)))
    compressed=tmp_path/'compressed.npz';stored=tmp_path/'stored.npz'
    _write_row(compressed,row,'compressed');_write_row(stored,row,'stored')
    with zipfile.ZipFile(compressed) as a,zipfile.ZipFile(stored) as b:
        assert a.namelist()==b.namelist()
        for name in a.namelist():assert a.read(name)==b.read(name)
        assert all(i.compress_type==zipfile.ZIP_STORED for i in b.infolist())
    with np.load(stored,allow_pickle=False) as data:
        for key,value in row.items():np.testing.assert_array_equal(data[key],value)
