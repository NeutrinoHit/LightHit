#!/usr/bin/env python3
"""Small event-independent beta/clock sensitivity experiment for track q1.

No frozen cache is changed. The exact 82-node QE-weighted emitted-yield ratio is
separated from cone/clock/transport effects. Scalar yield rescaling is NOT exact
spectral propagation: the latter remains in the direct reference.
"""
import paths  # inputs and outputs: scripts/shower_basis/paths.py
import argparse,json
from copy import copy
from pathlib import Path
from time import perf_counter,process_time
import numpy as np

C=.299792458
BETAS=np.array([1.,.99999,.9999,.999,.99,.95,.85,.76])
LOOKS=np.array([[-.68,np.sqrt(1-.68**2),0.],[.23,.61,np.sqrt(1-.23**2-.61**2)],[0.,0.,1.]])


def metrics(q,b,qr,br):
    return dict(charge_relative_error=float(abs(q-qr)/max(qr,1e-300)),
        bins_l1_relative=float(abs(b-br).sum()/max(br.sum(),1e-300)),
        signed_charge_change=float((q-qr)/max(qr,1e-300)))


def direct_response(kernel,rho,looks,beta,speed,lo,hi,edges):
    """Raw accepted segment quadrature; retain tiny signed quadrature residuals."""
    from dataclasses import replace
    from lighthit.sources import SpectralLightElements
    from lighthit.experimental.muon_integral import order1_integrals
    from lighthit.experimental.primary_polyline import primary_spectral_quadrature
    local=copy(kernel);d=kernel.detector;n=len(looks)
    local.detector=replace(d,positions_m=np.tile([rho,0.,0.],(n,1)),orientations=looks,
        effective_area_m2=np.repeat(d.effective_area_m2[:1],n),
        identifiers={k:np.repeat(v[:1],n) for k,v in d.identifiers.items()})
    local.config=replace(kernel.config,relative_time_edges_ns=edges)
    local.wavelength=primary_spectral_quadrature(kernel.medium,kernel.detector,order=2,
        wavelength_range_nm=kernel.config.wavelength_range_nm)
    length=hi-lo
    source=SpectralLightElements(np.array([[0.,0.,-hi]]),np.array([[0.,0.,1.]]),np.array([length]),
        np.array([length]),np.array([-length/beta**2]),np.array([beta]),np.array([-hi/speed]),
        np.array([-lo/speed]),np.array([0]),np.array([0]),cone_model='spectral')
    charge,bins,meta=order1_integrals(local,source,time_origin_ns=np.zeros(n))
    return charge,bins,dict(negative_bin_mass=float(-np.minimum(bins,0).sum()),
        negative_relative_to_charge=float(-np.minimum(bins,0).sum()/max(charge.sum(),1e-300)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rho',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--atlas',type=Path,default=paths.Q1_ATLAS.path())
    args=p.parse_args()
    import numba
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "tables"))
    from build_track_first_atlas import optical_kernel
    from lighthit.experimental.track_first_primitive import build_track_first_rows,TrackFirstAtlasStore
    from lighthit.experimental.primary_polyline import primary_spectral_quadrature
    from lighthit.prompt import _spectral_tables
    if numba.get_num_threads()!=1:p.error('NUMBA_NUM_THREADS=1 required')
    if (args.output/'report.json').exists():p.error('choose unused output')
    kernel=optical_kernel();local=copy(kernel)
    local.wavelength=primary_spectral_quadrature(kernel.medium,kernel.detector,order=2,
        wavelength_range_nm=kernel.config.wavelength_range_nm)
    tables=_spectral_tables(local);phase=tables['phase_index'];ng=tables['group_index']
    weighted_yield=np.array([np.sum(tables['s0w']*np.maximum(1-1/(beta*phase)**2,0.)) for beta in BETAS])
    ratio=weighted_yield/weighted_yield[0]
    effective_phase=float(np.sum(phase*tables['s0w']*(1-1/phase**2))/weighted_yield[0])
    effective_group=float(np.sum(ng*tables['s0w']*(1-1/phase**2))/weighted_yield[0])
    root1=1/np.sqrt(effective_phase**2-1)
    cuts=np.array([[-1.,4.],[root1-.4,root1+.5],[root1-1.,root1+.03]])*args.rho
    edges=np.arange(-5.,806.,5.)+.37
    q=np.zeros((len(BETAS),2,len(cuts),len(LOOKS)))
    bins=np.zeros(q.shape+(len(edges)-1,))
    timings=[];diagnostics=[];completed=np.zeros(len(BETAS),bool)
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=args.output/'checkpoint.npz'
    if checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as saved:
            q[:]=saved['q'];bins[:]=saved['bins'];completed[:]=saved['completed']
            timings=json.loads(str(saved['timings']));diagnostics=json.loads(str(saved['diagnostics']))
    # A tiny independent warmup distinguishes compilation/import overhead.
    began=perf_counter()
    direct_response(kernel,args.rho,LOOKS,1.,C,root1*args.rho-.01,root1*args.rho+.01,edges)
    warmup=perf_counter()-began;began,cpu=perf_counter(),process_time()
    for i,beta in enumerate(BETAS):
        if completed[i]:continue
        start=process_time()
        for j,speed in enumerate([C,beta*C]):
            if i==0 and j==1:
                q[i,j]=q[i,0];bins[i,j]=bins[i,0];continue
            for k,(lo,hi) in enumerate(cuts):
                qi,bi,info=direct_response(kernel,args.rho,LOOKS,beta,speed,lo,hi,edges)
                q[i,j,k]=qi;bins[i,j,k]=bi
                diagnostics.append(dict(beta=float(beta),clock=j,cut=k,**info))
        item=dict(beta=float(beta),cpu_s=process_time()-start)
        timings.append(item);completed[i]=True
        np.savez_compressed(checkpoint,q=q,bins=bins,completed=completed,timings=json.dumps(timings),diagnostics=json.dumps(diagnostics))
        print(json.dumps(item),flush=True)
    direct_cpu=sum(item['cpu_s'] for item in timings);direct_wall=perf_counter()-began
    store=TrackFirstAtlasStore(args.atlas);store.validate_kernel(kernel)
    comparisons=[]
    for i,beta in enumerate(BETAS):
        for j,clock in enumerate(['c','beta_c']):
            frontshift=args.rho/C*(np.sqrt(effective_group**2-1)-np.sqrt(effective_group**2-1/(beta if j else 1.)**2))
            shift_u=args.rho*(1/np.sqrt((beta*effective_phase)**2-1)-root1)
            for k,(lo,hi) in enumerate(cuts):
                for ell,look in enumerate(LOOKS):
                    qr,br=q[i,j,k,ell],bins[i,j,k,ell]
                    row=dict(beta=float(beta),clock=clock,cut=k,look=ell,reference_charge=float(qr),
                        yield_ratio=float(ratio[i]),source_interval_m=[float(lo),float(hi)])
                    row['direct_beta1_yield']=metrics(q[0,0,k,ell]*ratio[i],bins[0,0,k,ell]*ratio[i],qr,br)
                    # Taylor basis in beta^-2-1, normalized by exact emitted yield.
                    for anchor in [3,4]:
                        param=1/beta**2-1;pa=1/BETAS[anchor]**2-1
                        tq=(q[0,0,k,ell]+param/pa*(q[anchor,j,k,ell]/ratio[anchor]-q[0,0,k,ell]))*ratio[i]
                        tb=(bins[0,0,k,ell]+param/pa*(bins[anchor,j,k,ell]/ratio[anchor]-bins[0,0,k,ell]))*ratio[i]
                        row[f'direct_taylor_anchor_{BETAS[anchor]}']=metrics(tq,tb,qr,br)
                    for name,du,dt in [('atlas_yield',0.,0.),('atlas_front',0.,frontshift),('atlas_front_cone',shift_u,frontshift)]:
                        try:
                            aq,ab=store.query(args.rho,look,lo-du,hi-du,edges+dt)
                            row[name]=metrics(aq*ratio[i],ab*ratio[i],qr,br)
                        except ValueError as error:row[name]=dict(unsupported=str(error))
                    comparisons.append(row)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/'responses.npz',beta=BETAS,rho_m=args.rho,looks=LOOKS,cuts=cuts,
        edges_ns=edges,charge=q,bins=bins,yield_ratio=ratio)
    report=dict(event_inputs=False,rho_m=args.rho,spectral_nodes=len(phase),betas=BETAS.tolist(),
        clock_modes=['c','beta_c'],looks=LOOKS.tolist(),cuts_m=cuts.tolist(),
        direct_cpu_s=direct_cpu,direct_wall_s=direct_wall,warmup_wall_s=warmup,timings=timings,
        quadrature_diagnostics=diagnostics,max_negative_bin_mass_fraction=max(v['negative_relative_to_charge'] for v in diagnostics),
        yield_ratio_definition='exact positive 82-node QE-weighted emitted yield; not full transport factorization',
        effective_phase_index=effective_phase,effective_group_index=effective_group,
        comparisons=comparisons)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    summary=[]
    for beta in BETAS:
        entries=[r for r in comparisons if r['beta']==beta and r['clock']=='beta_c']
        out=dict(beta=float(beta))
        for method in ['direct_beta1_yield','atlas_yield','atlas_front','atlas_front_cone','direct_taylor_anchor_0.999']:
            valid=[r[method] for r in entries if 'unsupported' not in r[method]]
            out[method]=dict(max_q=max(v['charge_relative_error'] for v in valid),
                max_bins=max(v['bins_l1_relative'] for v in valid)) if valid else None
        summary.append(out)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(dict(rho_m=args.rho,direct_cpu_s=direct_cpu,summary=summary),indent=2))

if __name__=='__main__':main()
