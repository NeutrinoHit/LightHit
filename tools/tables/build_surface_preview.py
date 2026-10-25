#!/usr/bin/env python3
"""First-entry arrival moments on the existing coarse muon preview grid.

The old MC preview is retained separately as an OM-selection proposal. This
table contains deterministic finite-kernel moments for output charges, with
the same source groups and interpolation layout. No event enters preparation.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,hashlib,json
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','pack','cache','first-entry-tables','template','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--phi-panel',type=float,default=.5)
    p.add_argument('--spatial-order',type=int,default=4)
    p.add_argument('--positive-reference-samples',type=int,default=32768,
                   help='independent positive Neumann samples for nonphysical spectral cells')
    p.add_argument('--reference-seed',type=int,default=41)
    p.add_argument('--refine-from',type=Path,help='reuse completed moments and refine their all-orientation certificate')
    p.add_argument('--rows',type=int,nargs='+',help='prepare this subset as resumable checkpoints')
    args=p.parse_args()
    if args.workers<1 or args.phi_panel<=0 or args.spatial_order<1 or args.positive_reference_samples<2:p.error('positive workers and quadrature settings required')
    from build_track_ge2_atlas import load_optics
    from lighthit.muon.positive_fluence import PositiveFluenceAtlas
    from lighthit.muon.track_ge2_primitive import cone_point_evaluator,orientation_basis_looks
    from lighthit.muon.angular_preview import cubic_response
    from lighthit.muon._surface_preview import cone_first_moments
    from lighthit.muon._surface_preview_mc import positive_multiple_cell,first_tail
    from lighthit.muon._preview_positivity import observer_bernstein_map,certified_moments
    from lighthit.receiver_first_spectral import SpectralFirstEntry
    from lighthit.prompt import _spectral_tables
    from lighthit.band_fold import zero_frequency
    kernel,bundle,metadata=load_optics(args.pack,args.cache,args.model,frequency_nodes=1);bundle=zero_frequency(bundle)
    template=PositiveFluenceAtlas.load(args.template);spectral=SpectralFirstEntry.load(args.first_entry_tables/'q1-bands')
    R=kernel._receiver_contract()['radius_m'];t=_spectral_tables(kernel)
    if R!=spectral.radius_m or kernel.medium.g!=spectral.g:raise ValueError('Q1 geometry differs from model')
    for key in ('mut','mus','speed','s0w','s2w'):
        if not np.allclose(t[key],spectral.tables[key],rtol=1e-12,atol=0):raise ValueError('Q1 optics differ: '+key)
    alpha=np.asarray(metadata['alpha']);poly=np.polynomial.legendre.leg2poly((2*np.arange(len(alpha))+1)*alpha/(4*np.pi))
    poly=np.pad(poly,(0,4-len(poly)));looks=orientation_basis_looks()
    design=np.array([[cubic_response(np.eye(10)[j],look[0],look[2],look[1]**2,poly) for j in range(10)] for look in looks])
    if np.linalg.matrix_rank(design)!=10:raise ValueError('raw moments require a nondegenerate cubic acceptance')
    inverse=np.linalg.inv(design);gx,gw=np.polynomial.legendre.leggauss(4);sx,sw=np.polynomial.legendre.leggauss(args.spatial_order)
    positivity,_=observer_bernstein_map(poly)
    zmid=(template.z_edges_m[:-1]+template.z_edges_m[1:])/2;zh=np.diff(template.z_edges_m)/2
    z=(zmid[:,None]+zh[:,None]*sx).ravel()
    common={k:metadata[k] for k in ('alpha','wavelength_nm','wavelength_weight_nm','phase_index','efficiency')}
    common['effective_area_m2']=1.
    q1args=(R,spectral.g,np.log(spectral.radii_m),spectral.hit_charge,spectral.miss_charge,
            t['mut'],t['mus'],t['speed'],t['s0w'],t['s2w'],t['phase_index'],gx,gw,args.phi_panel)
    albedo=np.divide(t['mus'],t['mut'],out=np.zeros_like(t['mus']),where=t['mut']>0)
    if np.any(albedo>=1):raise ValueError('the positive Neumann reference requires absorbing bands')
    reference_proposal=max(.5,float(albedo.max()))
    def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    from lighthit import _receiver_first_groups,_receiver_first_lookup
    from lighthit.muon import _preview_positivity
    identity=dict(model=digest(args.model),template=digest(args.template),
        q1_manifest=digest(args.first_entry_tables/'q1-bands/manifest.json'),
        q1_charge_files={name:digest(args.first_entry_tables/'q1-bands'/(name+'.npy')) for name in ('hit_charge','miss_charge')},
        ge2_optics=metadata['optics_fingerprint'],phi_panel=args.phi_panel,spatial_gauss_order=args.spatial_order,
        implementation=digest(Path(__file__)),cone_implementation=digest(Path(__import__('lighthit.muon._surface_preview',fromlist=['x']).__file__)),
        moment_implementation=digest(_receiver_first_groups.__file__),lookup_implementation=digest(_receiver_first_lookup.__file__),
        positive_reference_samples=args.positive_reference_samples,
        positive_reference_seed=args.reference_seed,positive_reference_proposal=reference_proposal,
        positive_reference_implementation=digest(Path(__import__('lighthit.muon._surface_preview_mc',fromlist=['x']).__file__)))
    identity['observer_certificate_implementation']=digest(_preview_positivity.__file__)
    parent=None
    if args.refine_from is not None:
        parent=PositiveFluenceAtlas.load(args.refine_from)
        for key in ('model','template','q1_manifest','q1_charge_files','ge2_optics','phi_panel','spatial_gauss_order'):
            if parent.metadata['preparation'][key]!=identity[key]:raise ValueError('parent preview physics differs: '+key)
        identity['parent_preview_sha256']=digest(args.refine_from)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True);rows_dir=args.output/'rows';rows_dir.mkdir(exist_ok=True)
    final_path=args.output/'surface-preview.npz'
    if final_path.exists():
        with np.load(final_path,allow_pickle=False) as d:old=json.loads(str(d['metadata']))
        if old.get('preparation_fingerprint')!=fingerprint:raise FileExistsError('completed preview has different inputs')
        print(json.dumps(dict(complete=True,reused=True,fingerprint=fingerprint)),flush=True)
        return
    nr=len(template.rho_nodes)
    tail_arguments=(R,spectral.g,np.log(spectral.radii_m),spectral.hit_charge,spectral.miss_charge,t['mut'],t['mus'],t['speed'])
    def row(ir):
        path=rows_dir/f'{ir:03d}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as d:
                if str(d['fingerprint'])!=fingerprint:raise ValueError('preview resume inputs differ')
                return ir,d['moments'].copy(),d['sigma'].copy(),d['reference_cells'].copy()
        if parent is None:
            result=np.zeros((len(template.beta_nodes),2,len(zmid),10))
            low,high=template.rho_edges_m[ir:ir+2]**2
            rho=np.sqrt((low+high)/2+(high-low)/2*sx)
            for ib,beta in enumerate(template.beta_nodes):
                for radial,wr in zip(rho,sw/2):
                    first=cone_first_moments(float(radial),z,float(beta),*q1args)
                    result[ib,0]+=wr*np.einsum('zsm,s->zm',first.reshape(len(zmid),args.spatial_order,10),sw/2)
                    if radial<metadata['radius_window_m'][1]:
                        values=cone_point_evaluator(bundle,impact_m=float(radial),look=looks,beta=float(beta),
                            backend='fused-looks',near_field=True,**common)(z)[...,0].real
                        raw=(inverse@values).T
                        result[ib,1]+=wr*np.einsum('zsm,s->zm',raw.reshape(len(zmid),args.spatial_order,10),sw/2)
            sigma=np.zeros(result.shape[:-1]);references=[]
        else:
            result=parent.angular_moments[:,:,ir].copy();sigma=parent.sigma[:,:,ir].copy()
            with np.load(args.refine_from.parent/'rows'/f'{ir:03d}.npz',allow_pickle=False) as d:references=d['reference_cells'].tolist()
        if not np.isfinite(result).all():raise ValueError('nonfinite preview kernel')
        certified=certified_moments(np.ascontiguousarray(result[:,1]).reshape(-1,10),positivity).reshape(len(template.beta_nodes),len(zmid))
        bad=(result[:,1,:,0]<0)|~certified
        if np.any(bad):
            np.savez_compressed(rows_dir/f'invalid-{ir:03d}.npz',moments=result,fingerprint=fingerprint)
            for ib,iz in np.argwhere(bad):
                beta=float(template.beta_nodes[ib]);active=beta*t['phase_index']>1
                weights=np.where(active,np.maximum(0.,t['s0w']-t['s2w']/beta**2),0.)
                cones=np.where(active,1/(beta*t['phase_index']),1.)
                seed=int(hashlib.sha256(f'{args.reference_seed}/{ir}/{ib}/{iz}'.encode()).hexdigest()[:8],16)
                mean,se=positive_multiple_cell(args.positive_reference_samples,seed,template.rho_edges_m[ir:ir+2],
                    template.z_edges_m[iz:iz+2],R,spectral.g,t['mut'],t['mus'],weights,cones,first_tail,tail_arguments,
                    proposal=reference_proposal,radial_high=metadata['radius_window_m'][1])
                result[ib,1,iz]=mean;sigma[ib,1,iz]=se[0]
                references.append((int(ib),int(iz),args.positive_reference_samples))
        if np.any(result[...,0]<0):raise ValueError('positive reference failed')
        if not certified_moments(np.ascontiguousarray(result[:,1]).reshape(-1,10),positivity).all():
            raise ValueError('positive reference failed the all-orientation certificate')
        references=np.asarray(references,dtype=np.int64).reshape(-1,3)
        temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as stream:np.savez_compressed(stream,moments=result,sigma=sigma,reference_cells=references,fingerprint=fingerprint)
        temporary.replace(path);return ir,result,sigma,references
    # Compile shared read-only kernels before worker threads enter them.
    cone_first_moments(3.2,np.array([0.]),1.,*q1args)
    cone_point_evaluator(bundle,impact_m=10.,look=looks,beta=1.,backend='fused-looks',near_field=True,**common)(np.array([0.]))
    selected=list(range(nr)) if args.rows is None else sorted(set(args.rows))
    if any(ir<0 or ir>=nr for ir in selected):p.error('row index outside template')
    moments=np.empty_like(template.angular_moments);uncertainty=np.zeros_like(template.sigma);reference_count=0
    with ThreadPoolExecutor(max_workers=min(args.workers,len(selected))) as pool:
        futures=[pool.submit(row,ir) for ir in selected]
        for future in as_completed(futures):
            ir,values,sigma,references=future.result();moments[:,:,ir]=values;uncertainty[:,:,ir]=sigma
            reference_count+=len(references)
            print(json.dumps(dict(row=ir,rows=nr,positive_reference_cells=len(references))),flush=True)
    if len(set(selected))!=nr:return
    meta=dict({k:template.metadata[k] for k in ('g','tables','units','coefficient_contract')},
        kind='first-entry angular preview with independent positive local references',angular_moments=True,
        angular_receiver_contract=kernel._receiver_contract(),angular_kernel_keys=bundle['kernel_keys'],
        preparation=identity,preparation_fingerprint=fingerprint,source_input='synthetic stationary cones; no event',
        error_scope='quadrature and coarse source/interpolation errors require controls; sigma covers positive-reference sampling only',
        count_per_order=0,inactivity_certified=False,proposal_atlas_sha256=identity['template'],
        angular_design_condition=float(np.linalg.cond(design)),radial_window_m=metadata['radius_window_m'])
    meta.update(angular_source_degree=int(kernel.config.source_degree),angular_scattering_degree=int(kernel.config.scattering_degree))
    meta.update(positive_reference_cells=reference_count,positive_reference_method='unbounded collision-prefix Neumann series with analytic first-entry Q1 tail')
    meta['observer_positivity']='Bernstein subdivision certificate over the complete propagation-look unit disk'
    PositiveFluenceAtlas(template.rho_edges_m,template.z_edges_m,template.beta_nodes,
        moments[...,0].copy(),uncertainty,meta,moments).save(args.output/'surface-preview.npz')
    print(json.dumps(dict(complete=True,fingerprint=fingerprint)),flush=True)


if __name__=='__main__':main()
