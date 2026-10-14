"""Q1 shape interpolation preserves an analytic spectrum/geometry family."""
import hashlib,json
from types import SimpleNamespace
import numpy as np
import pytest
from lighthit.experimental.first_charge_beta_atlas import FirstChargeBetaAtlas
from lighthit.experimental.track_first_primitive import orientation_nodes


def analytic_model(tmp_path):
    (tmp_path/'manifest.json').write_text('synthetic optical test')
    radii=np.array([3.,5.,8.,12.]);x=np.array([-8.5,-4.,0.,1.,2.,4.,8.5])
    _,powers,_=orientation_nodes();left=np.zeros((4,7,10));right=np.zeros_like(left)
    amplitude=np.exp(-.1*radii-.002*radii**2)
    left[:,:,0]=amplitude[:,None]*(x-x[0]);right[:,:,0]=amplitude[:,None]*(x[-1]-x)
    base=SimpleNamespace(directory=tmp_path,radii_m=radii,x_edges=x,powers=powers,
        charge_left=left,charge_right=right,validate_kernel=lambda k:True)
    metadata=dict(base_manifest_sha256=hashlib.sha256((tmp_path/'manifest.json').read_bytes()).hexdigest(),
        beta_min=.999,minimum_span_over_rho=.05,yield_slope=.8,synthetic_validation_passed=True)
    path=tmp_path/'beta.npz';np.savez(path,schema='lighthit/first-charge-beta/1',metadata=json.dumps(metadata),
        left=.2*left,right=.2*right)
    return FirstChargeBetaAtlas(path,base),amplitude


def test_log_radial_interpolation_and_beta_shape_match_analytic_family(tmp_path):
    model,_=analytic_model(tmp_path)
    rho=np.array([4.,7.,10.]);beta=np.array([1.,.9999,.999]);lo=-rho;hi=2*rho;density=np.array([1.,2.,3.])
    look=np.array([[0.,0.,1.],[.6,0.,.8],[0.,0.,-1.]])
    s=1/beta**2-1
    expected=density*np.exp(-.1*rho-.002*rho**2)*3*(1-.8*s)*(1+.2*s)
    np.testing.assert_allclose(model.query_charge_many(rho,look,lo,hi,beta,coefficient0_per_m=density),expected,rtol=1e-12)
    assert model.supports(4.,.999,-4.,8.)
    assert not model.supports(4.,.998,-4.,8.)
    assert not model.supports(4.,1.,0.,.01)
    with pytest.raises(ValueError,match='narrower'):
        model.query_charge_many([4.],[[0.,0.,1.]],[0.],[.01],[1.])
    with pytest.raises(ValueError,match='beta outside'):
        model.query_charge_many([4.],[[0.,0.,1.]],[0.],[1.],[.99])


def test_changed_base_rejected(tmp_path):
    model,_=analytic_model(tmp_path)
    (tmp_path/'manifest.json').write_text('different optical test')
    with pytest.raises(ValueError,match='another base'):
        FirstChargeBetaAtlas(tmp_path/'beta.npz',model.base)
