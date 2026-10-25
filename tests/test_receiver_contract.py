"""Centre and first-entry banks cannot alias through packs or directory scans."""
from dataclasses import replace
import numpy as np
import pytest
import lighthit as lh
from lighthit.receiver import SphereSurface
from lighthit.receiver_vertex import ReceiverVertexConfig
from test_prompt import config,detector,spectral_medium


def _kernel(directory,reference='centre',radius=.216):
    cfg=config(directory,wavelength_nodes=1,omega_per_ns=np.array([0.,.1]),
        scattering_degree=4,source_degree=4,radial_range_m=(3.,5.),radial_nodes=4,
        scattered_receiver=reference,
        receiver_vertex=ReceiverVertexConfig(angular_pairs=8,radial_nodes=32,return_angle_nodes=8,return_path_nodes=16))
    return lh.TransportKernel(spectral_medium(),replace(detector(count=2),receiver_surface=SphereSurface(radius)),cfg)


def test_boundary_radius_and_numerics_change_cache_keys(tmp_path):
    centre=_kernel(tmp_path);surface=_kernel(tmp_path,'first-entry');larger=_kernel(tmp_path,'first-entry',.22)
    assert len({k._directional_cache_key(450.) for k in [centre,surface,larger]})==3
    altered=lh.TransportKernel(surface.medium,surface.detector,
        replace(surface.config,receiver_vertex=replace(surface.config.receiver_vertex,return_angle_nodes=16)))
    assert altered._directional_cache_key(450.)!=surface._directional_cache_key(450.)


def test_old_preview_can_propose_selection_but_cannot_supply_surface_output(tmp_path):
    from types import SimpleNamespace
    from lighthit.prompt import _spectral_tables
    from lighthit.muon.preview import check_atlas_optics
    kernel=_kernel(tmp_path,'first-entry')
    atlas=SimpleNamespace(metadata=dict(tables=_spectral_tables(kernel),g=kernel.medium.g))
    with pytest.raises(ValueError,match='receiver boundary'):check_atlas_optics(atlas,kernel)
    check_atlas_optics(atlas,kernel,selection_only=True)
    atlas.metadata['angular_receiver_contract']=kernel._receiver_contract()
    check_atlas_optics(atlas,kernel)
    with pytest.raises(ValueError,match='receiver boundary'):check_atlas_optics(atlas,_kernel(tmp_path,'first-entry',.22))


def test_directory_scan_never_crosses_receiver_boundaries(tmp_path):
    centre=_kernel(tmp_path);surface=_kernel(tmp_path,'first-entry')
    cache=surface._directional_cache_for(450.)
    assert surface._receiver_matches(cache.receiver_contract)
    assert centre._stored_table('directional',450.) is None
    centre._directional_cache_for(450.)
    another=_kernel(tmp_path,'first-entry')
    assert another._stored_table('directional',450.)==surface._table_path('directional',surface._directional_cache_key(450.))
    assert _kernel(tmp_path,'first-entry',.22)._stored_table('directional',450.) is None


def test_renamed_centre_table_and_old_pack_are_rejected(tmp_path):
    centre=_kernel(tmp_path);surface=_kernel(tmp_path,'first-entry')
    cache=centre._directional_cache_for(450.)
    cache.save(surface._table_path('directional',surface._directional_cache_key(450.)))
    with pytest.raises(ValueError,match='receiver boundary'):surface._directional_cache_for(450.)
    from lighthit.shower_basis import Ge2KernelStore
    store=Ge2KernelStore();indices=np.array([0])
    store._bundles[store._key(surface,indices)]={'coefficients':np.zeros(1)}
    with pytest.raises(ValueError,match='receiver boundary'):store.bundle(surface,indices)
    with pytest.raises(ValueError,match='receiver boundary'):
        store.kernel(surface,float(surface.wavelength.wavelength_nm[0]),np.array([0,1]),frequencies=slice(0,1))
