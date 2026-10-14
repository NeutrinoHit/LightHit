"""Stationary signed beta modes preserve the base and reject invalid domains."""
import hashlib,json
import numpy as np
import pytest
from test_track_ge2_disk_atlas import make_atlas
from lighthit.experimental.ge2_charge_beta_atlas import Ge2ChargeBetaAtlas


def model_file(tmp_path,base):
    base.segment_charge_many([5.],[ [0.,0.,1.] ],[3.],[-2.])
    eo,po,edges,coefs,prefix,lognorm=base._packed_charge
    path=tmp_path/'beta.npz'
    metadata=dict(beta_min=.99,synthetic_validation_passed=True,
        base_manifest_sha256=hashlib.sha256((base.directory/'manifest.json').read_bytes()).hexdigest())
    np.savez(path,schema='lighthit/ge2-charge-beta/1',metadata=json.dumps(metadata),
        nodes_m=base.impact_nodes_m,edge_offsets=eo,panel_offsets=po,edges_m=edges,
        coefficients=np.stack([2*coefs,-3*coefs]),prefix=np.stack([2*prefix,-3*prefix]),lognorm=lognorm)
    return path


def test_signed_modes_integrate_as_polynomial_and_preserve_beta_one(tmp_path):
    base=make_atlas(tmp_path);model=Ge2ChargeBetaAtlas(model_file(tmp_path,base),base)
    rho=np.array([4.2,5.5,7.8]);looks=np.tile([0.,0.,1.],(3,1));a=np.array([3.,2.,1.]);b=-a
    beta=np.array([1.,.999,.99]);density=np.array([1.,2.,3.]);delta=1/beta-1
    expected=base.segment_charge_many(rho,looks,a,b,coefficient0_per_m=density)*(1+2*delta-3*delta**2)
    np.testing.assert_allclose(model.segment_charge_many(rho,looks,a,b,beta,coefficient0_per_m=density),expected,rtol=1e-13)
    assert base.full_row_loads==0
    assert not model.supports(5.,.98,3.,-2.)
    with pytest.raises(ValueError,match='beta outside'):
        model.segment_charge_many([5.],[[0.,0.,1.]],[3.],[-2.],[.98])
    with pytest.raises(ValueError,match='geometry'):
        model.segment_charge_many([3.],[[0.,0.,1.]],[3.],[-2.],[1.])


def test_modified_base_manifest_rejected(tmp_path):
    base=make_atlas(tmp_path);path=model_file(tmp_path,base)
    manifest=base.directory/'manifest.json';manifest.write_text(manifest.read_text()+' ')
    with pytest.raises(ValueError,match='another base'):
        Ge2ChargeBetaAtlas(path,base)
