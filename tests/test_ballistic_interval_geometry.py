"""Independent spectral support and arrival-bin checks for compiled core."""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

pytest.importorskip('numba')
from lighthit._ballistic_interval_numba import interval_kernel, _stationary_cuts, _arrival


def calculate(*, phase=(1.34, 1.4), group=(1.37, 1.44), wavelengths=(350., 610.),
              lengths=(20.,), starts=(0.,), edges=(-100., 2000.), receivers=((3., 0., 15.),),
              times=None, order=32, beta=1.):
    lengths, starts = np.asarray(lengths), np.asarray(starts)
    n = len(lengths)
    start = np.zeros((n, 3)); start[:, 2] = starts
    direction = np.tile([0., 0., 1.], (n, 1))
    t0 = starts/.299792458 if times is None else np.asarray(times)[0]
    t1 = (starts+lengths)/.299792458 if times is None else np.asarray(times)[1]
    coeff = lengths * 1e8
    efficiency = np.zeros((len(wavelengths)-1, 4)); efficiency[:, 0] = 1.
    receivers = np.asarray(receivers)
    return interval_kernel(receivers, np.tile([0., 0., 1.], (len(receivers), 1)),
        np.ones(len(receivers)), start, direction, lengths, coeff, -coeff/beta**2,
        np.full(n,beta), t0, t1, np.asarray(wavelengths), np.asarray(phase), np.asarray(group),
        np.full(len(phase), .02), efficiency, np.array([4*np.pi]), *leggauss(order),
        np.zeros(len(receivers)), np.asarray(edges), True, np.zeros(len(receivers)), 0.)


def reference(phase, group, wavelengths, edges, z=15., rho=3., length=20.):
    def weight(lam):
        n = np.interp(lam, wavelengths, phase)
        sine = np.sqrt(1.-1./n**2)
        a = z-rho/np.sqrt(n*n-1.)
        if not 0 <= a < length:
            return 0.
        return 1e8*(1.-1./n**2)/lam**2*np.exp(-.02*rho/sine)/(2*np.pi*rho*sine)
    def arrival(lam):
        n = np.interp(lam, wavelengths, phase)
        ng = np.interp(lam, wavelengths, group)
        a = z-rho/np.sqrt(n*n-1.)
        return a/.299792458+rho/np.sqrt(1.-1./n**2)*ng/.299792458
    # Independent dense bracketing with Brent's root solver identifies all jumps.
    from scipy.optimize import brentq
    grid = np.unique(np.r_[np.linspace(wavelengths[0], wavelengths[-1], 10001), wavelengths])
    cuts = list(wavelengths)
    for target in edges:
        vals = np.array([arrival(lam)-target for lam in grid])
        for j in np.flatnonzero(vals[:-1]*vals[1:] < 0):
            cuts.append(brentq(lambda lam: arrival(lam)-target, grid[j], grid[j+1]))
    for boundary in (0., length):
        def root(lam):
            n = np.interp(lam, wavelengths, phase)
            return z-rho/np.sqrt(n*n-1.)-boundary
        vals = np.array([root(lam) for lam in grid])
        for j in np.flatnonzero(vals[:-1]*vals[1:] < 0):
            cuts.append(brentq(root, grid[j], grid[j+1]))
    cuts = np.unique(cuts)
    charge = 0.; bins = np.zeros(len(edges)-1)
    for a,b in zip(cuts[:-1], cuts[1:]):
        value = quad(weight, a, b, epsabs=1e-11, epsrel=1e-11)[0]
        charge += value
        slot = np.searchsorted(edges, arrival((a+b)/2), side='right')-1
        if 0 <= slot < len(bins): bins[slot] += value
    return charge, bins


@pytest.mark.parametrize('phase,group', [((1.34,1.4),(1.37,1.44)),
                                        ((1.34,1.34),(1.3,1.6))])
def test_bins_against_independent_quadrature(phase, group):
    edges = np.arange(40., 85., .37)
    q,bins,_ = calculate(phase=phase, group=group, edges=edges)
    qr,br = reference(phase, group, (350.,610.), edges)
    np.testing.assert_allclose(q[0],qr,rtol=2e-11)
    np.testing.assert_allclose(bins[0],br,rtol=2e-10,atol=1e-10)


def test_nonmonotone_phase_support_and_split_track_invariance():
    phase, group, waves = (1.1,1.5,1.1), (1.2,1.55,1.25), (350.,480.,610.)
    receivers = ((3.,0.,5.),)
    edges = np.arange(10.,45.,.23)
    args = dict(phase=phase,group=group,wavelengths=waves,receivers=receivers,edges=edges)
    q,b,_=calculate(lengths=(4.,),starts=(0.,),**args)
    qr,br=reference(phase,group,waves,edges,z=5.,length=4.)
    np.testing.assert_allclose(q[0],qr,rtol=1e-11)
    np.testing.assert_allclose(b[0],br,atol=1e-10,rtol=1e-10)
    qs,bs,_=calculate(lengths=(.7,1.1,2.2),starts=(0.,.7,1.8),**args)
    np.testing.assert_allclose(q,qs,rtol=1e-12)
    np.testing.assert_allclose(b,bs,atol=1e-10,rtol=1e-10)


def test_internal_time_extremum_bins():
    # tau=1/c, n_g constant => extremum at n=n_g.
    phase, group = (1.2,1.6), (1.4,1.4)
    cuts = _stationary_cuts(350.,610.,350.,260.,1.2,.4,1.4,0.,1.,1/.299792458)
    np.testing.assert_allclose(cuts, [350.,480.,610.],atol=1e-10)
    minimum = _arrival(480.,350.,260.,1.2,.4,1.4,0.,1.,15.,3.,1/.299792458,0.)
    edges=np.r_[minimum-1.,minimum+.0001,minimum+.01,minimum+.1,minimum+10.]
    q,b,_=calculate(phase=phase,group=group,edges=edges)
    qr,br=reference(phase,group,(350.,610.),edges)
    np.testing.assert_allclose(q[0],qr,rtol=1e-11)
    np.testing.assert_allclose(b[0],br,rtol=2e-9,atol=1e-9)
    assert b[0,0] > 0.
    np.testing.assert_allclose(b.sum(),q[0],rtol=1e-12)


def test_on_axis_singularity_is_explicit():
    with pytest.raises(ValueError,match='singular'):
        calculate(receivers=((0.,0.,10.),))


def test_rotated_short_track_directional_acceptance():
    # z>L must not overwrite the original Cartesian displacement.
    axis=np.array([1.,2.,3.]); axis/=np.linalg.norm(axis)
    tangent=np.cross(axis,[0.,1.,0.]); tangent/=np.linalg.norm(tangent)
    z,rho,length=5.,3.,4.
    receiver=z*axis+rho*tangent
    look=np.array([.2,.3,.7]); look/=np.linalg.norm(look)
    phase=np.array([1.2,1.5]); waves=np.array([350.,610.])
    # A(cos)=1+0.4 cos, always positive and nonisotropic.
    alpha=np.array([4*np.pi,4*np.pi*.4/3])
    args=(receiver[None,:],look[None,:],np.ones(1),np.zeros((1,3)),axis[None,:],
          np.array([length]),np.array([length*1e8]),np.array([-length*1e8]),np.ones(1),
          np.zeros(1),np.array([length/.299792458]),waves,phase,np.array([1.4,1.5]),
          np.array([.02,.02]),np.array([[1.,0.,0.,0.]]),alpha,*leggauss(32),
          np.zeros(1),np.array([-100.,1000.]),True,np.zeros(1),0.)
    q,b,_=interval_kernel(*args)
    def integrand(lam):
        n=np.interp(lam,waves,phase); sine=np.sqrt(1-1/n**2)
        distance=rho/sine
        acceptance=1+.4*np.dot(sine*tangent+axis/n,look)
        return 1e8*(1-1/n**2)/lam**2*np.exp(-.02*distance)/(2*np.pi*rho*sine)*acceptance
    cutoff=waves[0]+(waves[1]-waves[0])*(np.hypot(z,rho)/z-phase[0])/(phase[1]-phase[0])
    expected=quad(integrand,max(waves[0],cutoff),waves[1],epsabs=1e-11,epsrel=1e-12)[0]
    np.testing.assert_allclose(q[0],expected,rtol=1e-11)
    np.testing.assert_allclose(b.sum(),expected,rtol=1e-11)
    args=list(args); args[-3]=False
    qc,bc,_=interval_kernel(*args)
    np.testing.assert_allclose(qc,q,rtol=1e-12)
    assert bc.shape==(1,0)


def test_below_cherenkov_threshold_has_no_signal():
    q,b,diag=calculate(beta=.5)
    assert q.sum()==0. and b.sum()==0. and diag.sum()==0


def test_medium_above_threshold_but_no_geometric_root():
    q,b,diag=calculate(receivers=((50.,0.,1.),))
    assert q.sum()==0. and b.sum()==0. and diag.sum()==0


@pytest.mark.parametrize('reverse', [False, True])
def test_monotone_with_plateaus_and_extra_optical_subdivisions(reverse):
    # Cone support spans part of the steep central interval. QE subpanels
    # must not affect interval ownership, including phase-index plateaus.
    waves=np.array([350.,400.,480.,520.,610.])
    phase=np.array([1.1,1.1,1.5,1.5,1.6])
    if reverse: phase=phase[::-1].copy()
    group=np.array([1.25,1.3,1.45,1.55,1.65])
    fine=np.unique(np.r_[waves,np.linspace(350.,610.,365)])
    edges=np.arange(10.,45.,.31)
    args=dict(receivers=((3.,0.,5.),),lengths=(4.,),starts=(0.,),edges=edges)
    q,b,_=calculate(phase=phase,group=group,wavelengths=waves,**args)
    qf,bf,_=calculate(phase=np.interp(fine,waves,phase),group=np.interp(fine,waves,group),
                      wavelengths=fine,**args)
    np.testing.assert_allclose(qf,q,rtol=1e-12)
    np.testing.assert_allclose(bf,b,rtol=1e-10,atol=1e-10)
