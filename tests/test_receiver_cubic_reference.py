"""All four receiver harmonics against direct glass-entry path integrals."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from lighthit.cache import CacheGrid
from lighthit.directional import DirectionalCache
from lighthit.green import SolverSettings
from lighthit.medium import Medium
from lighthit.receiver import SphereSurface
from lighthit.receiver_first import first_entry_profile
from lighthit.receiver_vertex import ReceiverVertexConfig


def test_isotropic_source_with_cubic_receiver_matches_full_hg_paths():
    medium=Medium(.08,.02,.3,1.36);R=.216;r=3.
    omega=np.array([0.,.05]);grid=CacheGrid(np.array([r,4.]),omega)
    cache=DirectionalCache.build(medium,SolverSettings(16,2,12.,.15,6),grid,3,
        receiver_surface=SphereSurface(R),receiver_config=ReceiverVertexConfig(angular_pairs=16,radial_nodes=64),
        angular_backend='numba')
    x,w=np.polynomial.legendre.leggauss(16);gx,gw=np.polynomial.legendre.leggauss(8)
    cap=np.arcsin(R/r);cuts=[0.,cap,2*cap,.3,1.,np.pi];reference=np.zeros((len(omega),4),complex)
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        for theta,weight in zip((lo+hi)/2+(hi-lo)/2*x,(hi-lo)/2*w):
            profile=first_entry_profile(SphereSurface(R),distance_m=r,direction_cosine=np.cos(theta),g=medium.g,
                max_path_m=r+30/medium.extinction_per_m,path_nodes=128)
            density=PchipInterpolator(profile.path_m,profile.cumulative_m_inv,axis=0).derivative()
            half=np.diff(profile.path_m)/2;path=(profile.path_m[1:]+profile.path_m[:-1])[:,None]/2+half[:,None]*gx
            weighted=half[:,None]*gw*np.exp(-(medium.extinction_per_m-1j*omega[:,None,None]/medium.speed_m_per_ns)*path)
            moments=np.einsum('wtp,tpm->wm',weighted,density(path))
            s,c=np.sin(theta),np.cos(theta)
            first=s*moments[:,1]+c*moments[:,2]
            second=s*s*moments[:,3]+2*s*c*moments[:,4]+c*c*moments[:,5]
            third=s**3*moments[:,6]+3*s*s*c*moments[:,7]+3*s*c*c*moments[:,8]+c**3*moments[:,9]
            reference+=weight*s/2*medium.scattering_per_m*np.stack(
                (moments[:,0],first,(3*second-moments[:,0])/2,(5*third-3*first)/2),axis=-1)
    for degree in range(4):
        pair=np.flatnonzero((cache.table.pair_lambda==degree)&(cache.table.pair_mu==0))[0]
        actual=cache.coefficients[:,0,0,pair,0]/np.sqrt(2*degree+1)
        assert np.max(abs(actual-reference[:,degree]))/abs(reference[0,0])<.003
