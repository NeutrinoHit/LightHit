#!/usr/bin/env python3
"""Independent stationary scalar fluence from absorption endpoints.

For homogeneous absorbing HG media, p_abs(x)=mua*F(x). Conditional on n
scatterings before absorption, the n+1 free lengths are independent Exp(mut).
Conditioning n>=2 gives n=1+Geometric(mua/mut), with fluence mass
(mus/mut)^2/mua. No detector hits or time-bin transport are sampled.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from numba import njit


@njit(cache=True)
def sample_endpoints(count,mua,mus,g,seed):
    np.random.seed(seed)
    xyz=np.zeros((count,3));collisions=np.zeros(count,np.int32)
    mut=mua+mus
    for i in range(count):
        n=1+np.random.geometric(mua/mut)
        collisions[i]=n
        sx,sy,sz=0.,0.,1.
        x,y,z=0.,0.,0.
        for j in range(n+1):
            length=-np.log(1.-np.random.random())/mut
            x+=length*sx;y+=length*sy;z+=length*sz
            if j==n:break
            u=np.random.random()
            cosine=2*u-1 if abs(g)<1e-12 else (1+g*g-((1-g*g)/(1-g+2*g*u))**2)/(2*g)
            sine=np.sqrt(max(0.,1-cosine*cosine));phi=2*np.pi*np.random.random()
            if abs(sz)<.9:ex,ey,ez=-sy,sx,0.
            else:ex,ey,ez=0.,-sz,sy
            norm=np.sqrt(ex*ex+ey*ey+ez*ez);ex/=norm;ey/=norm;ez/=norm
            bx,by,bz=sy*ez-sz*ey,sz*ex-sx*ez,sx*ey-sy*ex
            cp,sp=np.cos(phi),np.sin(phi)
            sx,sy,sz=(cosine*sx+sine*(cp*ex+sp*bx),
                      cosine*sy+sine*(cp*ey+sp*by),
                      cosine*sz+sine*(cp*ez+sp*bz))
        xyz[i,0]=x;xyz[i,1]=y;xyz[i,2]=z
    return xyz,collisions


def exact_stationary_moments(mua,mus,g):
    mut=mua+mus;d1=mua+mus*(1-g);d2=mua+mus*(1-g*g)
    zz=np.diag([0.,0.,1.]);unit=np.eye(3)
    S=unit/3+mua/d2*(zz-unit/3)
    X=2/(mua*d1)*S;mean=np.array([0.,0.,1/d1])
    w0=mua/mut;w1=mua*mus/mut**2;w2=(mus/mut)**2
    m0=np.array([0.,0.,1/mut]);m1=(1+g)*m0
    S1=(1-g*g)*unit/3+g*g*zz
    X0=2*zz/mut**2;X1=2*((1+g)*zz+S1)/mut**2
    m=(mean-w0*m0-w1*m1)/w2
    second=(X-w0*X0-w1*X1)/w2
    return m,second,w2/mua


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count',type=int,default=1000000)
    p.add_argument('--mua',type=float,default=.0545);p.add_argument('--mus',type=float,default=.0151)
    p.add_argument('--g',type=float,default=.9);p.add_argument('--seed',type=int,default=90261009)
    p.add_argument('--output',type=Path,default=paths.work('green-map-20261009/stationary_mc'))
    args=p.parse_args()
    if args.mua<=0 or args.mus<=0 or abs(args.g)>=1 or args.count<2:p.error('positive rates/count, |g|<1 required')
    args.output.mkdir(parents=True,exist_ok=True)
    start=perf_counter();sample_endpoints(4,args.mua,args.mus,args.g,args.seed);warm=perf_counter()-start
    start,cpu=perf_counter(),process_time()
    xyz,n=sample_endpoints(args.count,args.mua,args.mus,args.g,args.seed)
    elapsed,cpu=perf_counter()-start,process_time()-cpu
    mean,second,mass=exact_stationary_moments(args.mua,args.mus,args.g)
    obs=np.column_stack((xyz[:,2],xyz[:,2]**2,xyz[:,0]**2+xyz[:,1]**2,n))
    target=np.array([mean[2],second[2,2],second[0,0]+second[1,1],2+args.mus/args.mua])
    stderr=obs.std(0,ddof=1)/np.sqrt(args.count)
    radius=np.linalg.norm(xyz,axis=1);mu=xyz[:,2]/radius
    redges=np.geomspace(3,300,33);muedges=np.linspace(-1,1,65)
    counts=np.histogram2d(radius,mu,bins=(redges,muedges))[0]
    volume=2*np.pi/3*np.diff(redges**3)[:,None]*np.diff(muedges)[None,:]
    F=mass*counts/(args.count*volume)
    error=mass*np.sqrt(counts)/(args.count*volume)
    np.savez_compressed(args.output/'histogram.npz',r_edges_m=redges,mu_edges=muedges,
        counts=counts,F_ge2=F,F_standard_error=error,volume_m3=volume)
    report=dict(count=args.count,mua=args.mua,mus=args.mus,g=args.g,seed=args.seed,
        fluence_mass_m=mass,expected_absorption_probability_ge2=(args.mus/(args.mua+args.mus))**2,
        wall_seconds=elapsed,cpu_seconds=cpu,jit_seconds=warm,
        observables=['z','z2','x2+y2','collision_count'],mean_observed=obs.mean(0).tolist(),
        mean_exact=target.tolist(),standard_error=stderr.tolist(),
        zscore=((obs.mean(0)-target)/stderr).tolist(),
        fraction_in_map=float(counts.sum()/args.count),
        method='absorption endpoints conditioned on at least two collisions; independent of RTE cache',
        F_units='m^-2 per emitted photon',normalization_scope='full-space integral of F_ge2 is fluence_mass_m')
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
