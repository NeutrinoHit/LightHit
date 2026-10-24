"""Addressed >=2 controls on the original electron-100GeV MC event."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json

import numpy as np

from .cases import case


def _input(ctx):
    from . import inputs as I
    from .cases_mcmatrix import _nearest
    steps_path=I.inputs_dir()/'mc/electron-100GeV-run0-steps.npz'
    reference_path=Path(__file__).resolve().parents[1]/'results/photon-mc-matrix/electron-100GeV.json'
    recorded=json.loads(reference_path.read_text())
    if hashlib.sha256(steps_path.read_bytes()).hexdigest()!=recorded['inputs'][steps_path.name]:
        raise ValueError('this control requires the original MC run-0 steps, matching the recorded SHA256')
    ctx._note(steps_path,reference_path)
    source,_=I.load_steps(steps_path)
    kernel=ctx.kernel()
    distance=_nearest(source,kernel.detector.positions_m)
    ids=np.flatnonzero((distance>=20)&(distance<50))
    row=next(r for r in recorded['metrics']['by_distance'] if r['distance_m']==[20.,50.])
    if len(ids)!=row['oms']:
        raise ValueError('distance group differs from the recorded MC geometry')
    return kernel,source,ids,row


def _zero_sources(kernel,source,basis,fold):
    from lighthit.shower.ge2 import node_sources
    from lighthit.axial_source import AxisFrame
    phases=kernel.medium.sample(kernel.wavelength.wavelength_nm)['phase_index']
    valid=source.beta*phases.max()>1
    source=source if valid.all() else source.subset(valid)
    frame=AxisFrame.of(source.field(0,phase_index=float(phases.max())))
    return node_sources(kernel,source,basis,fold,np.array([0.]),frame)


def _apply(kernel,bundle,sources,ids):
    from lighthit.shower.ge2 import apply_folded
    d=kernel.detector
    return apply_folded(bundle,sources,d.positions_m[ids],-d.orientations[ids],
                        kernel.acceptance()['alpha'],d.effective_area_m2[ids],
                        kernel.config.radial_range_m)[0].real


@case('ge2-near-electron', 'charge-controls', kind='metrics',
      notes='Fifteen original-MC receivers, zero frequency only; source-grid and all-band controls')
def charge_controls(ctx):
    from types import SimpleNamespace
    from lighthit.shower import ShowerTransport
    from lighthit.band_fold import zero_frequency
    kernel,source,ids,mc=_input(ctx)
    engine=ShowerTransport(kernel);tables=engine.tables();fold=tables['fold']
    zero=zero_frequency(tables['folded'])
    baseline=_apply(kernel,zero,_zero_sources(kernel,source,engine.config.basis,fold),ids)
    finer=_apply(kernel,zero,_zero_sources(kernel,source,
        replace(engine.config.basis,cell_m=.125,stencil_m=.1,time_step_ns=.01),fold),ids)
    phases=kernel.medium.sample(kernel.wavelength.wavelength_nm)['phase_index']
    identity=SimpleNamespace(bands=tuple(range(len(phases))),phase_index=phases,scale=fold.scale)
    all_bands=_apply(kernel,tables['store'].bundle(kernel,np.array([0],int)),
                     _zero_sources(kernel,source,engine.config.basis,identity),ids)
    return dict(headline=dict(receivers=len(ids),charge_pe=float(baseline.sum()),
        finer_over_default=float(finer.sum()/baseline.sum()),
        all_bands_over_default=float(all_bands.sum()/baseline.sum())),
        receivers=ids.tolist(),charge_pe=baseline.tolist(),finer_pe=finer.tolist(),
        all_bands_pe=all_bands.tolist(),mc_pe=mc['>=2']['mc_pe'],
        mc_sigma_pe=mc['>=2']['mc_sigma_pe'],mc_runs=16,
        production_route_changed=False)


@case('ge2-near-electron', 'continuous-next-event', kind='metrics',
      notes='Independent continuous-spectrum collision estimator; no Geant4, RTE tables or finite collision cutoff in the reference')
def continuous_next_event(ctx):
    from lighthit.shower import ShowerTransport
    from lighthit.band_fold import zero_frequency
    from .ge2_reference import estimate
    kernel,source,ids,mc=_input(ctx)
    engine=ShowerTransport(kernel);tables=engine.tables()
    exact=_apply(kernel,zero_frequency(tables['folded']),
                  _zero_sources(kernel,source,engine.config.basis,tables['fold']),ids)
    # The RTE is used only to allocate samples between receivers. The
    # reference divides each score by that probability, so its mean is
    # independent of the allocation and of the RTE response values.
    reference=estimate(kernel,source,ids,samples=5_000_000,seed=552131,
                       batch_size=1_000_000,allocation=exact)
    return dict(headline=dict(reference_pe=reference['charge_pe'][1],
        reference_sigma_pe=reference['mc_sigma_pe'][1],
        route_pe=float(exact.sum()),saved_mc_pe=mc['>=2']['mc_pe']),
        reference=reference,saved_mc_sigma_pe=mc['>=2']['mc_sigma_pe'],
        saved_mc_runs=16,production_route_changed=False)
