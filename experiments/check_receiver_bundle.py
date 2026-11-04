#!/usr/bin/env python3
"""Reproduce saved charged events against a delivered optical cache.

No Geant4, photon generation or network connection is needed. Fixtures keep
numerical round-trip tolerances separate from the original photon-MC criteria.
Timing includes all orders and fresh source compilation after one warm-up.
"""
import os
for _name in ('NUMBA_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):os.environ.setdefault(_name,'1')
from pathlib import Path
import argparse,hashlib,json,time,resource
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def fixture(root,name,sha):
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('fixture escapes bundle')
    if digest(path)!=sha:raise ValueError('fixture checksum differs: '+name)
    return path


def compare(actual,expected,*,rtol,atol):
    charge=np.asarray(actual.charge_components_pe);reference=expected['charge_components_pe']
    bins=np.asarray(actual.components_pe);ref_bins=expected['components_pe']
    if charge.shape!=reference.shape or bins.shape!=ref_bins.shape:raise ValueError('response dimensions differ')
    diff=np.abs(charge-reference)
    bin_l1=np.abs(bins-ref_bins).sum(axis=1)
    passed=(np.all(diff<=atol+rtol*np.abs(reference))
        and np.all(bin_l1<=atol+rtol*np.abs(reference))
        and np.array_equal(actual.active,expected['active'])
        and np.allclose(actual.time_origin_ns,expected['time_origin_ns'],rtol=0,atol=1e-8))
    return dict(passed=bool(passed),max_charge_absolute_pe=float(diff.max()),
        max_bin_l1_pe=float(bin_l1.max()),charge_by_order=charge.sum(axis=0).tolist(),
        reference_by_order=reference.sum(axis=0).tolist(),rtol=rtol,atol_pe=atol)


def mc_groups(source,model,charge,record):
    from scipy.spatial import cKDTree
    # Same start/midpoint/end distance definition as the stored MC registry;
    # keep this verifier runnable from a delivered bundle without a checkout.
    start=np.asarray(source.start_m);end=start+source.length_m[:,None]*source.direction
    distance=cKDTree(np.r_[start,end,.5*(start+end)]).query(model.detector.positions_m)[0];rows=[]
    for old in record['metrics']['by_distance']:
        lo,hi=old['distance_m'];mask=(distance>=lo)&(distance<(np.inf if hi is None else hi))
        if int(mask.sum())!=old['oms']:raise ValueError('MC distance group changed')
        values={}
        for j,key in enumerate(('0','1','>=2','total')):
            reference=old[key];mc=reference['mc_pe'];sigma=reference['mc_sigma_pe']
            q=float(charge[mask,j].sum() if j<3 else charge[mask].sum())
            ratio=q/mc if mc else None;se=ratio*sigma/mc if mc else None
            values[key]=dict(ratio=ratio,ratio_sigma=se,significant=reference['mc_share']>.001,
                passed=None if ratio is None else abs(ratio-1)<=max(.03,2*se))
        rows.append(dict(distance_m=old['distance_m'],oms=int(mask.sum()),orders=values))
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=2,help='first call warms the route; default second call is the complete timed event')
    p.add_argument('--case',action='append',dest='selected',help='check only these named fixtures; repeat to select several')
    args=p.parse_args(argv)
    if args.repeats<1:p.error('positive repeat count required')
    import lighthit as lh,numba
    from lighthit.shower import ShowerTransport
    from lighthit.muon import MuonTransport
    numba.set_num_threads(1)
    root=args.bundle.resolve();cases=json.loads((root/'validation/cases.json').read_text());model=lh.load_model(root/'model/bgvd-model.npz')
    if not cases.get('cases'):raise ValueError('no validation fixtures in bundle')
    if args.selected:
        if not set(args.selected)<=set(c['name'] for c in cases['cases']):raise ValueError('unknown validation fixture')
        cases['cases']=[c for c in cases['cases'] if c['name'] in args.selected]
    rows=[];result=dict(code_version=lh.__version__,cases=rows,complete=False,passed=False,threads=1)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    try:
        for case in cases['cases']:
            path=fixture(root,case['steps'],case['steps_sha256'])
            with np.load(path,allow_pickle=False) as d:source=lh.SpectralLightElements(**{k:d[k] for k in d.files if k!='metadata'},cone_model='spectral')
            with np.load(fixture(root,case['reference'],case['reference_sha256']),allow_pickle=False) as d:expected={k:d[k] for k in d.files}
            ancestry=None
            if case['kind']=='muon':
                with np.load(fixture(root,case['parts'],case['parts_sha256']),allow_pickle=False) as d:
                    if not np.array_equal(d['mask_0'],np.arange(len(source))):raise ValueError('ancestry coverage differs')
                    ancestry={k[len('ancestry_0_'):]:d[k] for k in d.files if k.startswith('ancestry_0_')}
            engine=(MuonTransport(model.kernel(),root/'tables/muon-data.json',threads=1) if ancestry is not None
                else ShowerTransport(model.kernel(),root/'tables/shower-data.json',threads=1))
            engine.tables();timings=[]
            for repeat in range(args.repeats):
                if hasattr(engine,'_compiled'):engine._compiled.clear()
                wall=time.perf_counter();cpu=time.process_time()
                response=engine(source) if ancestry is None else engine(source,ancestry,primary_uid=case['primary_uid'])
                timing=dict(repeat=repeat,warm=repeat>0,wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu)
                timings.append(timing);print(json.dumps(dict(case=case['name'],**timing)),flush=True)
            numerical=compare(response,expected,rtol=case['numerical_rtol'],atol=case['numerical_atol_pe'])
            mc=json.loads(fixture(root,case['mc_record'],case['mc_record_sha256']).read_text())
            groups=mc_groups(source,model,response.charge_components_pe,mc)
            rows.append(dict(name=case['name'],steps_sha256=case['steps_sha256'],numerical=numerical,timings=timings,
                mc_by_distance=groups,peak_rss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            args.report.write_text(json.dumps(result,indent=2)+'\n')
        result.update(complete=True,passed=all(r['numerical']['passed'] for r in rows))
        args.report.write_text(json.dumps(result,indent=2)+'\n')
        return 0 if result['passed'] else 2
    except Exception as error:
        result['error']=str(error);args.report.write_text(json.dumps(result,indent=2)+'\n');raise


if __name__=='__main__':raise SystemExit(main())
