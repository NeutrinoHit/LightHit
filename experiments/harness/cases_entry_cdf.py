"""Independent validation of an eight-moment first-entry arc CDF."""
import math
import numpy as np
from scipy.integrate import quad
from .cases import case
from .entry_cdf import make_basis


def validate(poly, relative_density_tolerance=2e-6/64):
    prepare,cdf,metadata=make_basis(poly,relative_density_tolerance)
    rng=np.random.default_rng(261024);rows=[];fallbacks=0;density_error=0.;bin_error=0.
    for ratio in [.001,.005,.02,.1,.5]:
      for cosine in [.2,.7,.98]:
        r=1/ratio;sine=math.sqrt(1-cosine*cosine);rho=r*sine
        if rho<=1:continue
        for h in [-.8,.1,.95]:
            root=math.sqrt(1-h*h);k0=math.sqrt(r*r-h*h);tangent=math.sqrt(r*r-1)
            delta=root*root/(k0+tangent);v=1/math.sqrt(2*(k0+tangent)*sine*rho);a2=(v*root)**2
            look=rng.normal(size=3);look/=np.linalg.norm(look);axial,across,side=look;mu=.03
            coefficient,scale,bound=prepare(cosine,sine,axial,across,side,mu,tangent,delta,a2,poly)
            if scale<0:
                fallbacks+=1
                continue
            def density(u):
                y=math.sqrt(max(0.,1-u*u));cp=1-2*a2*u*u;sp=2*math.sqrt(a2*u*u*(1-a2*u*u))
                shift=delta*y*y;relative_path=shift-y*math.sqrt(delta*(2*tangent+shift))
                mean=cosine*axial+sine*cp*across;sidepart=sine*sp*side
                value=sum(max(0.,np.polynomial.polynomial.polyval(mean+sign*sidepart,poly)) for sign in [-1,1])
                return value*math.exp(-mu*relative_path)/math.sqrt(1-a2*u*u)
            # Factor out the common extinction before adaptive quadrature;
            # otherwise a faint integral can falsely pass epsabs at low order.
            for upper in [1e-9,.01,.3,.8,.999999,1.]:
                reference=quad(lambda x:density(upper*x),0.,1.,epsabs=1e-14,epsrel=2e-12)[0]*upper
                actual=cdf(upper,coefficient,scale)/scale
                error=abs(actual/reference-1)
                if error>bound+1e-11:raise AssertionError((ratio,cosine,h,upper,error,bound))
                rows.append(dict(radius_over_r=ratio,cosine=cosine,h=h,upper=upper,
                    relative_error=float(error),density_bound=float(bound)))
            nodes=np.linspace(0.,1.,129)
            values=np.array([cdf(u,coefficient,scale)/scale for u in nodes])
            if np.any(np.diff(values)<-1e-14):raise AssertionError('CDF is not monotone')
            for u in nodes:
                actual=np.polynomial.polynomial.polyval(math.sqrt(max(0.,1-u*u)),coefficient)
                error=abs(actual/density(u)-1);density_error=max(density_error,error)
                if error>bound+1e-11:raise AssertionError(('density',error,bound))
            edges=np.array([0.,1e-9,.1,.4,.8,.999999,1.])
            reference=np.array([quad(density,a,b,epsabs=1e-14,epsrel=2e-12)[0] for a,b in zip(edges[:-1],edges[1:])])
            actual=np.diff([cdf(u,coefficient,scale)/scale for u in edges])
            error=float(abs(actual-reference).sum()/reference.sum());bin_error=max(bin_error,error)
            if error>bound+1e-11:raise AssertionError(('bins',error,bound))
    return dict(headline=dict(cdf_checks=len(rows),fallback_geometries=fallbacks,
        maximum_cdf_relative_error=max(r['relative_error'] for r in rows),
        maximum_density_relative_error=float(density_error),maximum_bin_l1_relative_error=bin_error,
        moment_bytes=metadata['full_moment_bytes']),basis=metadata,rows=rows,
        production_route_changed=False)


@case('entry-cdf-moments','arc-moments',kind='metrics',
      notes='Independent arc integrals and pointwise density checks; no event route is enabled')
def arc_moments(ctx):
    kernel=ctx.kernel();alpha=kernel.acceptance()['alpha'];ell=np.arange(len(alpha))
    poly=np.polynomial.legendre.leg2poly((2*ell+1)*alpha/(4*np.pi))
    return validate(poly)
