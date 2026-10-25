"""Local positive deposition, conserved mass and CDF variation bounds."""
import numpy as np
from lighthit.receiver_time import quadratic_rule,deposit_cdf
from lighthit._receiver_first_lookup import _scalar_edge


def test_quadratic_cdf_error_is_bounded_by_its_real_extrema():
    for nodes,maximum in [(17,4.),(65,12.),(257,12.)]:
        s,a,b,tv,valid=quadratic_rule(nodes,maximum);assert valid
        du=maximum/(nodes-1)
        for j in range(nodes-1):
            x=np.linspace(0.,1.,501)
            error=(np.arcsinh(s[j]+(s[j+1]-s[j])*x)-j*du)/du-(a[j]*x+b[j]*x*x)
            assert abs(np.diff(error)).sum()<=tv[j]+1e-12
            assert a[j]>=0 and a[j]+2*b[j]>=0


def test_deposition_conserves_charge_and_respects_clock_convolved_error():
    nodes=65;maximum=8.;u=np.linspace(0.,maximum,nodes);s,a,b,tv,valid=quadratic_rule(nodes,maximum)
    cdf=1-np.exp(-2*u);cdf/=cdf[-1]
    du=maximum/(nodes-1);d=np.diff(cdf)/du
    primitive=np.r_[0.,np.cumsum(cdf[:-1]*np.diff(s)+d*(du*s[1:]-np.diff(np.cosh(u))))]
    for spread in [0.,1e-10,.8,15.]:
        edges=np.arange(-5.,601.,.7);bins=np.zeros(len(edges)-1)
        deposit_cdf(cdf,.1,spread,4.,.15,edges,s,a,b,bins)
        values=np.array([_scalar_edge(t,.1,spread,4.,.15,maximum,cdf,primitive) for t in edges])
        assert np.min(bins)>-1e-14
        assert abs(bins.sum()-1)<2e-12
        assert abs(bins-np.diff(values)).sum()<=np.sum(abs(np.diff(cdf))*tv)+2e-11
