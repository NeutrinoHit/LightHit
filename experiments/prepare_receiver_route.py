#!/usr/bin/env python3
"""Make fast route packs from a completed first-entry directional bank.

The detector model supplies glass, optics and response. Existing band nodes
are retained; only kernels change. Input banks and the current data bundle
are preserved. The new shower manifest is explicit until event validation.
"""
from pathlib import Path
from dataclasses import asdict
import argparse,hashlib,json,os
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--first-entry-tables',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fold-template',type=Path,required=True)
    p.add_argument('--stationary-only',action='store_true',help='prepare a zero-frequency reference pack without a shower manifest')
    p.add_argument('--manifest-name',default='shower-data.json',help='candidate manifest name for an outer bundle builder')
    args=p.parse_args(argv)
    if Path(args.manifest_name).name!=args.manifest_name or args.manifest_name in ('','.','..'):
        p.error('--manifest-name must be a filename in the output directory')
    from lighthit.bgvd import load_model
    from lighthit.directional import DirectionalCache
    from lighthit.transport import KernelConfig
    from lighthit.receiver_vertex import ReceiverVertexConfig
    from lighthit.receiver_data import encode_settings
    from lighthit._portable import same_values
    from lighthit.band_fold import BandFold,fold_bundle,save_folded
    from lighthit.shower import ShowerConfig
    from lighthit.shower.api import FREQUENCIES_PER_NS,READOUT_PER_NS
    bank=json.loads((args.bank/'manifest.json').read_text())
    first=DirectionalCache.load(args.bank/bank['bands'][0]['file']);s=first.settings
    wanted=np.array([0.]) if args.stationary_only else FREQUENCIES_PER_NS
    if not np.array_equal(first.grid.omega_per_ns,wanted):raise ValueError('bank frequency grid differs from the requested pack')
    settings=dict(scattered_receiver='first-entry',scattering_degree=s.scattering_degree,source_degree=s.spatial_degree,
        k_max_per_m=s.k_max_per_m,k_panel_per_m=s.k_panel_per_m,k_order=s.k_order,
        radial_range_m=first.radius_range_m,radial_nodes=len(first.grid.radii_m),acceptance_degree=first.acceptance_degree,
        wavelength_nodes=len(bank['bands']),receiver_vertex=ReceiverVertexConfig(**first.receiver_contract['numerics']),
        first_entry_tables=args.first_entry_tables.resolve())
    config=KernelConfig(omega_per_ns=first.grid.omega_per_ns,readout_omega_per_ns=None if args.stationary_only else READOUT_PER_NS,
        cache_directory=args.output/'rte-cache',cache_policy='require',**settings)
    kernel=load_model(args.model).kernel(config)
    if not kernel._receiver_matches(first.receiver_contract):raise ValueError('bank glass differs from model')
    if not np.allclose(kernel.wavelength.wavelength_nm,[b['wavelength_nm'] for b in bank['bands']],rtol=1e-12,atol=0):
        raise ValueError('bank wavelengths differ from model quadrature')
    (args.output/'rte-cache').mkdir(parents=True,exist_ok=True);first=None
    for wave,entry in zip(kernel.wavelength.wavelength_nm,bank['bands']):
        source=args.bank/entry['file']
        if digest(source)!=entry['sha256']:raise ValueError('bank checksum mismatch: '+entry['file'])
        with np.load(source,allow_pickle=False) as data:m=json.loads(str(data['metadata']))
        band=dict(m['medium']);band.pop('provenance',None)
        if (not kernel._receiver_matches(m.get('receiver_contract') or {})
                or not same_values(band,kernel._band_key(float(wave))) or m['settings']!=asdict(kernel._solver_settings())):
            raise ValueError('bank optical or numerical contract differs')
        target=kernel._table_path('directional',kernel._directional_cache_key(float(wave)))
        if target.exists():
            if digest(target)!=entry['sha256']:raise FileExistsError('different cache at '+str(target))
        else:os.link(source,target)
    if args.stationary_only:
        from lighthit.shower_basis import Ge2KernelStore
        store=Ge2KernelStore(dtype=np.complex128)
    else:store=kernel.ge2_kernel_store()
    indices=np.arange(len(kernel.config.omega_per_ns))
    bundle=store.bundle(kernel,indices)
    if not args.stationary_only:
        # Preserve the fast route's compact stationary companion. A view
        # into the 65-frequency tensor has the same coefficients but makes
        # charge-only packet screening read strided cache lines.
        stationary=store.bundle(kernel,indices[:1])
        stationary['coefficients']=np.ascontiguousarray(stationary['coefficients'])
        stationary['coefficients'].setflags(write=False)
    store.save(args.output/'ge2-pack')
    if args.stationary_only:
        manifest=dict(rte_cache='rte-cache',ge2_pack='ge2-pack',kernel_settings=encode_settings(settings,args.output))
        (args.output/'kernel-data.json').write_text(json.dumps(manifest,indent=2)+'\n')
        report=dict(model_sha256=digest(args.model),bank_manifest_sha256=digest(args.bank/'manifest.json'),
            receiver_contract=kernel._receiver_contract(),ge2_pack_bytes=store.nbytes,
            source_degree=s.spatial_degree,frequency_nodes=1,shower_route=False)
        (args.output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        return 0
    template=args.fold_template/'folded.json' if args.fold_template.is_dir() else args.fold_template
    recorded=json.loads(template.read_text())['band_fold'];nodes=tuple(recorded['bands'])
    fold=BandFold.of(kernel,rank=len(nodes),bands=nodes)
    if not np.isclose(fold.metadata()['worst_residual_design'],recorded['worst_residual_design'],rtol=1e-9,atol=0):
        raise ValueError('source band design differs from template')
    folded=fold_bundle(bundle,fold,subdivide=ShowerConfig().fold_subdivide)
    save_folded(folded,args.output/'ge2-band-fold')
    manifest=dict(rte_cache='rte-cache',ge2_pack='ge2-pack',ge2_band_fold='ge2-band-fold',
                  kernel_settings=encode_settings(settings,args.output))
    (args.output/args.manifest_name).write_text(json.dumps(manifest,indent=2)+'\n')
    report=dict(model_sha256=digest(args.model),bank_manifest_sha256=digest(args.bank/'manifest.json'),
        receiver_contract=kernel._receiver_contract(),ge2_pack_bytes=store.nbytes,folded_bytes=folded['coefficients'].nbytes,
        band_nodes=list(nodes),event_validation_complete=False)
    (args.output/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
