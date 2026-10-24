"""Geometry-only first-entry rows, independently checked on a visible cap."""
import time

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator

from .cases import case


def _cap_charge(r, theta, g, mu):
    rho, z = r*np.sin(theta), r*np.cos(theta)
    upper = z-np.sqrt(1-rho*rho) if z > 0 and rho < 1 else np.inf
    x, w = np.polynomial.legendre.leggauss(48)
    q, qw = (x+1)/2, w/2
    phi, pw = np.pi*(x+1), np.pi*w

    def integrand(ell):
        axial = z-ell
        distance = np.hypot(rho, axial)
        root = np.sqrt(distance**2-1+q*q)
        c = root/distance
        s = np.sqrt(np.maximum(0., 1-c*c))
        dz = c[:, None]*axial/distance+s[:, None]*rho/distance*np.cos(phi)
        phase = (1-g*g)/(4*np.pi*(1+g*g-2*g*dz)**1.5)
        entry = (distance**2-1)/(root+q)
        jac = q/(distance*root)
        return np.sum(np.exp(-mu*(ell+entry))*qw*jac*(phase@pw))/np.pi

    return quad(integrand, 0., upper, epsabs=1e-10, epsrel=2e-8)[0]


@case('first-entry-receiver', 'geometry-rows', kind='metrics',
      notes='Three geometry rows against independent first-flight cap integration; no event route or table interpolation')
def geometry_rows(ctx):
    from lighthit.receiver import SphereSurface
    from lighthit.receiver_first import first_entry_profile
    r, g, mu = 6., .9, .12
    rows = []
    x, w = np.polynomial.legendre.leggauss(8)
    for theta in (0., .7, 2.8):
        start = time.perf_counter()
        profile = first_entry_profile(SphereSurface(1.), distance_m=r,
                                      direction_cosine=np.cos(theta), g=g,
                                      max_path_m=r+200)
        build_seconds = time.perf_counter()-start
        p = profile.path_m
        half = np.diff(p)/2
        nodes = (p[:-1]+p[1:])[:, None]/2+half[:, None]*x
        derivative = PchipInterpolator(p, profile.cumulative_m_inv[:, 0]).derivative()
        value = np.sum(half[:, None]*w*np.exp(-mu*nodes)*derivative(nodes))
        reference = _cap_charge(r, theta, g, mu)
        rows.append(dict(theta_rad=theta, attenuated_m_inv=float(value),
                         cap_reference_m_inv=float(reference), relative_error=float(value/reference-1),
                         minimum_mass_increment=float(np.diff(profile.cumulative_m_inv[:, 0]).min()),
                         path_nodes=len(p), build_seconds=build_seconds))
    return dict(headline=dict(rows=len(rows),
                             maximum_relative_error=max(abs(v['relative_error']) for v in rows),
                             geometry_build_seconds=sum(v['build_seconds'] for v in rows)),
                rows=rows, g=g, extinction_times_radius=mu,
                event_route_changed=False)


@case('direct-first-entry', 'axial-sphere', kind='metrics',
      notes='Direct glass entry against an analytic axial spectral integral; default routes unchanged')
def direct_axial(ctx):
    import lighthit as lh
    from lighthit.ballistic_interval import first_entry_ballistic
    medium = lh.SpectralMedium([350., 610.], [.7, .7], [0., 0.],
                              [1.35, 1.35], [1.4, 1.4], g=.9)
    detector = lh.DetectorArray([[0., 0., .25]], [[0., 0., -1.]], .05,
        lambda x: np.ones_like(x), lambda w: np.full_like(w, .3),
        receiver_surface=lh.SphereSurface(.2))
    kernel = lh.TransportKernel(medium, detector, lh.KernelConfig(
        omega_per_ns=np.array([0.]), acceptance_degree=0))
    L = .02
    source = lh.SpectralLightElements(np.zeros((1, 3)), np.array([[0., 0., 1.]]),
        np.array([L]), np.array([1e8*L]), np.array([-1e8*L]), np.ones(1),
        np.zeros(1), np.array([L/.299792458]), np.array([0]), np.array([0]), cone_model='spectral')
    x, w = np.polynomial.legendre.leggauss(96)
    along = .25-(x+1)*L/2
    c = 1/1.35
    entry = c*along-np.sqrt(.2**2-(1-c*c)*along**2)
    expected = (1e8*(1-c*c)*.3*(1/350-1/610)*.05/(np.pi*.2**2)
                *np.dot(w, np.exp(-.7*entry))*L/2)
    result = first_entry_ballistic(kernel, source, time_origin_ns=[0.],
                                   relative_edges_ns=[-5., 0., 5., 10.])
    q = float(result.charge_pe[0])
    relative_error = q/expected-1
    if abs(relative_error) > 2e-8:
        raise ValueError('direct axial glass-entry control failed')
    return dict(headline=dict(charge_pe=q, relative_error=relative_error,
                bins_minus_charge_pe=float(result.bins_pe.sum()-q)),
                reference_pe=expected, ray_pairs=result.metadata['ray_pairs'],
                default_event_route_changed=False)


@case('direct-first-entry', 'path-moment-window', kind='metrics',
      notes='Dispersive finite windows and tilted acceptance against independent wavelength/source/azimuth entry rays')
def path_moment_window(ctx):
    import lighthit as lh
    from lighthit.ballistic_interval import first_entry_ballistic
    from lighthit._surface_ballistic_rays import evaluate as rays
    from lighthit._surface_ballistic import PreparedBallistic
    from lighthit.ballistic_interval import BallisticIntervalConfig
    medium = lh.SpectralMedium([350., 610.], [.07, .04], [0., 0.],
                              [1.38, 1.32], [1.43, 1.36], g=.9)
    beta, L = .9, .05
    z = 2/np.sqrt((beta*1.35)**2-1)+L/2
    look = np.array([.6, .3, np.sqrt(.55)])
    detector = lh.DetectorArray([[2., 0., z], [2., 0., z-.12]], [-look, -look], .05,
        lambda x: 1+.3*x+.1*x*x+.02*x*x*x, lambda w: np.full_like(w, .3),
        receiver_surface=lh.SphereSurface(.126))
    kernel = lh.TransportKernel(medium, detector, lh.KernelConfig(
        omega_per_ns=np.array([0.]), acceptance_degree=3))
    source = lh.SpectralLightElements(np.zeros((1, 3)), np.array([[0., 0., 1.]]),
        np.array([L]), np.array([1e8*L]), np.array([-1e8*L/beta**2]), np.array([beta]),
        np.zeros(1), np.array([L/(beta*.299792458)]), np.array([0]), np.array([0]), cone_model='spectral')
    p = PreparedBallistic(kernel, source, BallisticIntervalConfig())
    t, d = p.tables, kernel.detector
    def reference(order):
        q, _ = rays(*p.pairs, d.positions_m, -d.orientations, d.effective_area_m2,
            p.radii, source.start_m, source.direction, source.length_m, source.coefficient0,
            source.coefficient2, source.beta, source.start_ns, source.end_ns,
            t['panel_edges'], t['phase_nodes'], t['group_nodes'], t['extinction_nodes'],
            t['efficiency_coefficients'], kernel.acceptance()['alpha'],
            *np.polynomial.legendre.leggauss(order), np.zeros(2), np.array([-5., 40.]))
        return q
    low, high = reference(32), reference(64)
    value = first_entry_ballistic(kernel, source)
    tolerance = 1e-10+2e-6*abs(high)
    error = abs(value.charge_pe-high)
    if np.any(abs(high-low) > tolerance/4) or np.any(error > tolerance):
        raise ValueError('path-moment window control failed')
    return dict(headline=dict(receivers=2,
                maximum_tolerance_fraction=float((error/tolerance).max()),
                reference_refinement_fraction=float((abs(high-low)/tolerance).max())),
                charge_pe=value.charge_pe.tolist(), reference_pe=high.tolist(),
                charge_control=value.metadata['charge_control'],
                default_event_route_changed=False)
