#!/usr/bin/env python3
"""Addressed cone-and-clock interpolation checks between physical beta rows."""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model','first-entry-tables','base','anchor','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--radii',nargs='+',type=float,default=[30.])
    p.add_argument('--betas',nargs='+',type=float,default=[.9995,.9999])
    p.add_argument('--align-glass-front',action='store_true')
    args=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/tables'))
    from build_track_first_atlas import optical_kernel,row_query
    from lighthit.muon.track_first_primitive import TrackFirstAtlasStore,build_track_first_rows
    kernel=optical_kernel(args.model,args.first_entry_tables);base=TrackFirstAtlasStore(args.base)
    manifest=json.loads((args.anchor/'manifest.json').read_text());anchor_beta=manifest['physics']['beta']
    complete_anchor=TrackFirstAtlasStore(args.anchor) if manifest['status']=='complete' else None
    looks=np.array([[.23,.61,np.sqrt(1-.23**2-.61**2)],[.8,0.,.6],[0.,0.,-1.]])
    records=[];scale=2*np.pi/137.035999084*1e9
    origin=(kernel._receiver_contract()['radius_m']/max(base.metadata['spectral_tables']['speed'])
        if args.align_glass_front else 0.)
    for radius in args.radii:
        row=None
        if complete_anchor is None:
            entry=next(e for e in manifest['rows'] if e['rho_m']==radius)
            with np.load(args.anchor/entry['file'],allow_pickle=False) as d:row={k:d[k].copy() for k in d.files}
        x=np.asarray(manifest['x_edges']);t=np.asarray(manifest['time_over_rho_edges']);powers=np.asarray(manifest['powers'])
        for beta in args.betas:
            root=float(np.mean(1/np.sqrt((beta*kernel.medium.phase_index)**2-1)))
            weight=(1/beta**2-1)/(1/anchor_beta**2-1)
            if not 0<=weight<=1:raise ValueError('beta outside anchors')
            for cut,(a,b) in enumerate([(-1.,4.),(root-.25,root+.2),(root+.6,root+3.)]):
                lo,hi=a*radius,b*radius;edges=np.arange(-5.,806.,5.)+.37*cut
                refs=build_track_first_rows(kernel,rho_m=radius,orientations=looks,beta=beta,
                    speed_m_per_ns=beta*.299792458,u_edges_m=[lo,hi],tau_edges_ns=edges)
                for look,ref in zip(looks,refs):
                    q0,b0=base.query(radius,look,lo,hi,edges,radial_time_origin_ns=origin)
                    q1,b1=(complete_anchor.query(radius,look,lo,hi,edges,radial_time_origin_ns=origin) if complete_anchor is not None
                        else row_query(row,radius,x,t,powers,look,lo,hi,edges,radial_time_origin_ns=origin))
                    q=(1-weight)*q0+weight*q1;bins=(1-weight)*b0+weight*b1
                    qr=ref.charge_cells[0];br=ref.bins_cells[0]
                    record=dict(radius_m=radius,beta=beta,cut=cut,look=look.tolist(),
                        charge_relative=abs(q-qr)/qr,bins_l1_relative=abs(bins-br).sum()/br.sum(),
                        charge_scaled=abs(q-qr)*scale/max(abs(qr)*scale,1e-5),
                        bins_l1_scaled=abs(bins-br).sum()*scale/max(abs(br).sum()*scale,1e-5))
                    records.append(record);print(json.dumps(record),flush=True)
    maximum=float(max(max(r['charge_scaled'],r['bins_l1_scaled']) for r in records))
    report=dict(criterion=.01,passed=bool(maximum<=.01),max_scaled_error=maximum,checks=records,
        radial_time_origin_ns=origin,model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
        base_manifest_sha256=hashlib.sha256((args.base/'manifest.json').read_bytes()).hexdigest(),
        anchor_manifest_sha256=hashlib.sha256((args.anchor/'manifest.json').read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
