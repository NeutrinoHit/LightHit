"""A joined receiver bank equals direct construction at block boundaries."""
from dataclasses import replace
import numpy as np
from lighthit.cache import CacheGrid
from lighthit.directional import DirectionalCache
from lighthit.green import SolverSettings
from lighthit.medium import synthetic_medium
from lighthit.receiver import SphereSurface
from lighthit.receiver_vertex import ReceiverVertexConfig
from lighthit.receiver_bank import prepare_receiver_bank


def test_parallel_frequency_blocks_join_without_losing_receiver_contract(tmp_path):
    media=[synthetic_medium(),replace(synthetic_medium(),wavelength_nm=500.,group_index=1.4)]
    grid=CacheGrid(np.array([3.,4.]),np.array([0.,.03,.1]))
    settings=SolverSettings(4,2,2.,.2,4);surface=SphereSurface(.216)
    config=ReceiverVertexConfig(angular_pairs=8,radial_nodes=32,return_angle_nodes=8,return_path_nodes=16)
    metadata=prepare_receiver_bank(media,settings,grid,surface,0,tmp_path,config=config,jobs=2,frequency_block=2)
    for medium,entry in zip(media,metadata['bands']):
        direct=DirectionalCache.build(medium,settings,grid,0,receiver_surface=surface,
            receiver_config=config,angular_backend='numba',radial_phase='flight')
        joined=DirectionalCache.load(tmp_path/entry['file'])
        np.testing.assert_array_equal(joined.grid.omega_per_ns,grid.omega_per_ns)
        np.testing.assert_allclose(joined.coefficients,direct.coefficients,rtol=1e-13,atol=1e-20)
        assert joined.receiver_contract==direct.receiver_contract
    (tmp_path/'manifest.json').unlink()
    # A matching completed frequency block can be resumed unchanged.
    block=tmp_path/'blocks/band-000/omega-0000.npz';stamp=block.stat().st_mtime_ns
    prepare_receiver_bank(media,settings,grid,surface,0,tmp_path,config=config,jobs=1,frequency_block=2)
    assert block.stat().st_mtime_ns==stamp


def test_extending_acceptance_reuses_exact_lower_degrees():
    medium=synthetic_medium();grid=CacheGrid(np.array([3.,4.]),np.array([0.,.1]))
    settings=SolverSettings(4,2,2.,.2,4);surface=SphereSurface(.216)
    config=ReceiverVertexConfig(angular_pairs=8,radial_nodes=32,return_angle_nodes=8,return_path_nodes=16)
    kwargs=dict(receiver_surface=surface,receiver_config=config,angular_backend='numba',radial_phase='flight')
    low=DirectionalCache.build(medium,settings,grid,0,**kwargs)
    direct=DirectionalCache.build(medium,settings,grid,1,**kwargs)
    reused=DirectionalCache.build(medium,settings,grid,1,reuse_receiver_cache=low,**kwargs)
    np.testing.assert_allclose(reused.coefficients,direct.coefficients,rtol=2e-13,atol=1e-18)
    assert reused.timings_s['reused_acceptance_degree']==0
