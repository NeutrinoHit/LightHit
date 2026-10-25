"""Constant panel densities agree with the general cubic CDF integrator."""
import numpy as np
import pytest
from lighthit._receiver_geometry_lookup import fold_linear_moments
from lighthit._receiver_geometry_fold import fold_coefficients


@pytest.mark.parametrize('mu',[0.,.03,.3])
def test_linear_density_fold_matches_general_integrator(mu):
    rng=np.random.default_rng(421)
    path=3.+np.r_[0.,np.geomspace(1e-6,300.,70)]
    moments=np.vstack((np.zeros((1,10)),np.cumsum(rng.normal(size=(len(path)-1,10)),axis=0)))
    coefficients=np.zeros((4,len(path)-1,10))
    coefficients[2]=np.diff(moments,axis=0)/np.diff(path)[:,None];coefficients[3]=moments[:-1]
    edges=np.r_[-10.,path/.22,np.linspace(0.,1500.,117)]
    before=fold_coefficients(path,coefficients,mu,.02,.22,edges,0.)
    after=fold_linear_moments(path,moments,mu,.02,.22,edges)
    for x,y in zip(before,after):np.testing.assert_allclose(x,y,rtol=2e-12,atol=3e-16)
