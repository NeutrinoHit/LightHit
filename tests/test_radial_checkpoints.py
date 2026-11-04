"""Preparation progress preserves scientific arrays across interruption."""
import json
import numpy as np
import pytest
from lighthit._table_checkpoint import JoinedArrays,radial_blocks
from lighthit.receiver_geometry import ReceiverGeometry,ReceiverGeometryConfig


def test_radial_jobs_do_not_depend_on_worker_count():
    for count in [2,3,4,5,308,328]:
        blocks=radial_blocks(count)
        np.testing.assert_array_equal(np.concatenate(blocks),np.arange(count))
        assert all(2<=len(b)<=3 for b in blocks)


def test_committed_rows_survive_and_uncommitted_rows_are_replaced(tmp_path):
    shapes={'a':(5,3),'b':(2,5,2)}
    # A spectral array's radial coordinate is axis 1.
    store=JoinedArrays(tmp_path,{'b':shapes['b']},{'physics':1},axis=1)
    values={'b':np.arange(8).reshape(2,2,2)}
    store.merge(0,[0,1],values,metadata={'known':True})
    store.arrays['b'][:,2:]=123.;store.arrays['b'].flush()
    again=JoinedArrays(tmp_path,{'b':shapes['b']},{'physics':1},axis=1)
    assert again.contains(0,[0,1]) and not again.contains(1,[2,3,4])
    np.testing.assert_array_equal(again.arrays['b'][:,:2],values['b'])
    with pytest.raises(ValueError,match='inputs changed'):JoinedArrays(tmp_path,{'b':shapes['b']},{'physics':2},axis=1)
    again.arrays['b'][0,0,0]=999;again.arrays['b'].flush()
    with pytest.raises(ValueError,match='checksum'):JoinedArrays(tmp_path,{'b':shapes['b']},{'physics':1},axis=1)


def test_geometry_resume_after_deleted_job_is_bitwise_identical(tmp_path):
    options=dict(radii_unit=np.array([3.,4.,6.,9.,12.]),g=.9,max_excess_unit=20.,
        config=ReceiverGeometryConfig(hit_nodes=3,miss_nodes=5,path_grid_nodes=17,path_nodes=16,path_order=4,scatter_order=6))
    direct=ReceiverGeometry.build(**options)
    def interrupt(done,total):raise RuntimeError('interrupted after joined checkpoint')
    with pytest.raises(RuntimeError,match='interrupted'):
        ReceiverGeometry.build_parallel(**options,output=tmp_path,jobs=1,keep_blocks=False,progress=interrupt)
    assert not (tmp_path/'manifest.json').exists()
    assert not list((tmp_path/'blocks').glob('*/manifest.json'))
    checkpoint=json.loads((tmp_path/'build-progress.json').read_text());assert len(checkpoint['completed'])==1
    actual=ReceiverGeometry.build_parallel(**options,output=tmp_path,jobs=2,keep_blocks=False)
    for name in ('hit_cdf','hit_total','hit_entry','miss_cdf','miss_total'):
        np.testing.assert_array_equal(getattr(actual,name),getattr(direct,name))
    assert not list((tmp_path/'blocks').glob('*/manifest.json'))


def test_preparation_refuses_insufficient_disk_before_allocating_arrays(tmp_path,monkeypatch):
    import lighthit._table_checkpoint as checkpoint
    monkeypatch.setattr(checkpoint.shutil,'disk_usage',lambda p:type('Space',(),{'free':1})())
    with pytest.raises(OSError,match='bytes available'):JoinedArrays(tmp_path,{'a':(100,100)},{})
    assert not (tmp_path/'a.npy').exists()


def test_changed_geometry_inputs_preserve_existing_complete_atlas(tmp_path):
    cfg=ReceiverGeometryConfig(hit_nodes=3,miss_nodes=3,path_grid_nodes=9,path_nodes=8,path_order=3,scatter_order=4)
    options=dict(radii_unit=np.array([3.,4.]),g=.9,max_excess_unit=10.,config=cfg,output=tmp_path,jobs=1)
    ReceiverGeometry.build_parallel(**options)
    old=(tmp_path/'manifest.json').read_bytes()
    with pytest.raises(ValueError,match='inputs changed'):
        ReceiverGeometry.build_parallel(**dict(options,g=.8),overwrite=True)
    assert (tmp_path/'manifest.json').read_bytes()==old


def test_completed_spectral_substage_can_resume_after_quadratic_addition(tmp_path):
    from lighthit.receiver_first_spectral import SpectralFirstEntry,SpectralFirstEntryConfig
    config=ReceiverGeometryConfig(hit_nodes=3,miss_nodes=5,path_grid_nodes=65,path_nodes=16,path_order=4,scatter_order=6)
    geometry=ReceiverGeometry.build(radii_unit=np.array([3.,4.]),g=.9,max_excess_unit=5.,config=config)
    geometry.save(tmp_path/'geometry')
    tables=dict(mut=np.array([.1]),mus=np.array([.02]),speed=np.array([.22]),s0w=np.array([1.]),s2w=np.array([.5]))
    args=dict(geometry_directory=tmp_path/'geometry',radius_m=.216,tables=tables,output=tmp_path/'spectral',jobs=1,
              config=SpectralFirstEntryConfig(time_nodes=33),keep_blocks=False)
    atlas=SpectralFirstEntry.build_parallel(**args);atlas.prepare_quadratic(geometry,tmp_path/'spectral')
    stamp=(tmp_path/'spectral/hit_cdf.npy').stat().st_mtime_ns
    resumed=SpectralFirstEntry.build_parallel(**args,overwrite=True)
    assert resumed.metadata['quadratic_time']
    assert (tmp_path/'spectral/hit_cdf.npy').stat().st_mtime_ns==stamp
