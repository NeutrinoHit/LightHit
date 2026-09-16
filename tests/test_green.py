from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import quad
from lighthit import Medium, synthetic_medium, SolverSettings, PointGreenSolver


FAST = SolverSettings(16, 72, 3.0, .05, 8)


def test_rotation_and_source_linearity_and_time_shift():
    m=synthetic_medium(); solver=PointGreenSolver(m, FAST)
    omega=np.array([0,.03,.1]); r=np.array([8.,3.,6.]); s=np.array([0.,0.,1.])
    Q=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    a=solver.solve(omega,r,direction=s)
    b=solver.solve(omega,Q@r,direction=Q@s,photons=7,emission_time_ns=9)
    np.testing.assert_allclose(b.components,7*a.components*np.exp(1j*omega*9)[:,None,None],rtol=1e-12,atol=1e-14)
    assert np.all(a.components[:,:,0]==0)


def test_batch_observations_match_separate_calls():
    solver=PointGreenSolver(synthetic_medium(),FAST)
    r=np.array([[8,0,6],[10,0,0.]])
    batch=solver.solve([0,.06],r)
    for d in range(2):
        single=solver.solve([0,.06],r[d])
        np.testing.assert_allclose(batch.components[:,d],single.components[:,0],rtol=1e-12,atol=1e-14)


def test_ballistic_isotropic_charge_and_time_bins():
    m=replace(synthetic_medium(),scattering_per_m=0)
    solver=PointGreenSolver(m,FAST)
    r=10.; front=r/m.speed_m_per_ns
    result=solver.solve([0,.1],[r,0,0],direction=None)
    expected=np.exp(-m.absorption_per_m*r)/(4*np.pi*r*r)
    np.testing.assert_allclose(result.charge_per_m2,[expected],rtol=1e-14)
    profile=result.readout([0,front-.1,front+.1,front+5],0)
    np.testing.assert_allclose(profile.values_per_m2,[[0,expected,0]],rtol=1e-14)


def test_isotropic_real_space_against_infinite_sine_integral():
    # Analytic isotropic spectral resolvent, inverted independently with QUADPACK.
    m=replace(synthetic_medium(),g=0.0); r=20.
    solver=PointGreenSolver(m,SolverSettings(0,0,8,.04,10))
    for omega in [0.0,.05]:
        d0=m.extinction_per_m-1j*omega/m.speed_m_per_ns
        def kernel(k):
            A=(np.arctan(k/d0)/k) if k else 1/d0
            return k*m.scattering_per_m**2*A**3/(1-m.scattering_per_m*A)/(2*np.pi**2*r)
        expected=quad(lambda k:kernel(k).real,0,np.inf,weight='sin',wvar=r,epsabs=1e-12,limlst=100)[0]
        expected+=1j*quad(lambda k:kernel(k).imag,0,np.inf,weight='sin',wvar=r,epsabs=1e-12,limlst=100)[0]
        got=solver.solve([omega],[r,0,0],direction=None).components[0,0,2]
        np.testing.assert_allclose(got,expected,rtol=2e-5,atol=1e-12)


@pytest.mark.parametrize("displacement,direction", [([0,0,0],[0,0,1]),([0,0,10],[0,0,1]),([1,0,0],[0,0,2])])
def test_singular_or_invalid_geometry(displacement,direction):
    with pytest.raises(ValueError):
        PointGreenSolver(synthetic_medium(),FAST).solve([0.0],displacement,direction=direction)


def test_zero_photon_result():
    result=PointGreenSolver(synthetic_medium(),FAST).solve([0,.1],[10,0,0],photons=0)
    np.testing.assert_array_equal(result.components,0)
    np.testing.assert_array_equal(result.readout([0,50,100]).components,0)


@pytest.mark.parametrize("kwargs", [dict(absorption_per_m=0),dict(scattering_per_m=-1),dict(g=1),dict(group_index=0)])
def test_invalid_medium(kwargs):
    with pytest.raises(ValueError):replace(synthetic_medium(),**kwargs)


@pytest.mark.parametrize("omega", [[],[.1,0],[0,0],[-.1,0],[0,np.nan]])
def test_invalid_frequencies(omega):
    with pytest.raises(ValueError):PointGreenSolver(synthetic_medium(),FAST).solve(omega,[10,0,0])
