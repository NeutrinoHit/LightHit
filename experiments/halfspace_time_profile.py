#!/usr/bin/env python
"""Compute and reuse a laterally integrated half-space time spectrum.

The normal pencil enters at t=0. The receiver collects the chosen exterior
angular cone over the whole exit plane. Optical inputs are parameters;
the defaults are an illustrative milk model, not a calibrated sample.
All output is per one exterior incident photon. Plotting needs matplotlib.

compute: solve the RTE once per frequency, save spectrum and a time profile.
plot: reuse a saved spectrum with new time bins or Gaussian timing width.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.special import ndtr, erfc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from lighthit.medium import Medium
from lighthit.readout import inverse_bins
from lighthit.experimental.halfspace import HalfSpaceSolver, HalfSpaceSettings, SurfacePort
from lighthit.experimental.halfspace_boundary import HalfSpaceBoundary


def time_arguments(parser):
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--time-start-ns", type=float, default=-.5)
    parser.add_argument("--time-stop-ns", type=float, default=10.)
    parser.add_argument("--bin-width-ns", type=float, default=.05)
    parser.add_argument("--timing-sigma-ns", type=float, default=.1,
                        help="Gaussian standard deviation of the combined pulse/readout response")
    parser.add_argument("--no-plot", action="store_true")


def temporal_profile(omega, spectrum, specular, args):
    values = np.array([args.time_start_ns, args.time_stop_ns, args.bin_width_ns, args.timing_sigma_ns])
    if (not np.isfinite(values).all() or args.time_stop_ns <= args.time_start_ns
            or args.bin_width_ns <= 0 or args.timing_sigma_ns < 0):
        raise ValueError("finite ordered time bounds, positive bin width and nonnegative timing sigma are required")
    count = int(math.ceil((args.time_stop_ns-args.time_start_ns)/args.bin_width_ns))
    edges = np.linspace(args.time_start_ns, args.time_stop_ns, count+1)
    period = 2*np.pi/np.max(np.diff(omega))
    if edges[-1]-edges[0] >= period:
        raise ValueError(f"time window must be shorter than the frequency-grid period {period:g} ns; reduce the frequency step")
    started = perf_counter()
    components = inverse_bins(omega, spectrum, edges, args.timing_sigma_ns).T
    prompt = np.zeros(len(edges)-1)
    if args.timing_sigma_ns > 0:
        prompt = specular*np.diff(ndtr(edges/args.timing_sigma_ns))
    else:
        index = np.searchsorted(edges, 0., side="right")-1
        if 0 <= index < len(prompt):
            prompt[index] = specular
    elapsed = perf_counter()-started
    total = components.sum(axis=1)+prompt
    charge = float(spectrum[0].sum().real+specular)
    band_per_ns = None
    if args.timing_sigma_ns > 0:
        # For a nonnegative impulse, |H(omega)| <= H(0). This bounds only
        # the omitted Gaussian-filtered band, not angular or grid errors.
        band_per_ns = (max(0.,float(spectrum[0].sum().real))
                       /(args.timing_sigma_ns*np.sqrt(2*np.pi))
                       *erfc(omega[-1]*args.timing_sigma_ns/np.sqrt(2)))
    diagnostics = {
        "inversion_seconds": elapsed, "bins": len(total),
        "actual_bin_width_ns": float(edges[1]-edges[0]),
        "fourier_period_ns": float(period),
        "omega_max_times_sigma": float(omega[-1]*args.timing_sigma_ns),
        "gaussian_omitted_band_l1_bound": None if band_per_ns is None else float(band_per_ns*(edges[-1]-edges[0])),
        "gaussian_omitted_band_max_bin_bound": None if band_per_ns is None else float(band_per_ns*np.max(np.diff(edges))),
        "all_time_reflected_fraction": charge,
        "window_reflected_fraction": float(total.sum()),
        "window_fraction_of_all_time": float(total.sum()/charge) if charge else 0.,
        "negative_bin_mass": float(np.maximum(-total,0).sum()),
        "zero_time": "normal pencil reaches exterior interface; prompt specular reflection is separate",
        "profile_scope": "all exit positions; specified exterior angular cone",
        "clipped_or_renormalized": False,
    }
    return edges, components, prompt, total, diagnostics


def draw(edges, components, prompt, total, metadata, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centers = (edges[1:]+edges[:-1])/2
    width = np.diff(edges)
    series = [(prompt/width,"Зеркальное",'#777777'),
              (components[:,1]/width,"1 рассеяние",'#56B4E9'),
              (components[:,2]/width,"≥2 рассеяний",'#D55E00'),
              (total/width,"Сумма",'#0072B2')]
    fig, axes = plt.subplots(1,2,figsize=(10.5,4.3),layout="constrained")
    for value,label,color in series:
        axes[0].plot(centers,value,label=label,color=color,lw=1.6)
        # A logarithmic display masks nonpositive samples; stored bins remain raw.
        axes[1].plot(centers,np.ma.masked_less_equal(value,0),label=label,color=color,lw=1.6)
    axes[0].set(xlim=(edges[0],min(edges[-1],1.5)),xlabel="Время t, нс",
                ylabel="Доля падающих фотонов / нс",title="Начало импульса")
    axes[1].set(yscale="log",xlim=(max(0.,edges[0]),edges[-1]),xlabel="Время t, нс",
                ylabel="Доля падающих фотонов / нс",title="Хвост отражённого сигнала")
    peak = max(float(np.max(total/width)),1e-20)
    axes[1].set_ylim(peak*1e-7,peak*2)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=8.5)
    p=metadata['parameters']
    title=(rf"Модель: $\mu_s={p['mu_s_per_cm']:g}$ см$^{{-1}}$, "
           rf"$\mu_a={p['mu_a_per_cm']:g}$ см$^{{-1}}$, $g={p['g']:g}$; "
           rf"$n={p['phase_index_ratio']:g}$, $n_g={p['group_index']:g}$; "
           rf"$\sigma_t={1000*args.timing_sigma_ns:g}$ пс")
    fig.suptitle(title,fontsize=11)
    fig.savefig(args.output_directory/"profile.png",dpi=190)
    fig.savefig(args.output_directory/"profile.svg")
    plt.close(fig)


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode",required=True)
    compute = sub.add_parser("compute",help="calculate the optical spectrum and a profile")
    time_arguments(compute)
    compute.add_argument("--mu-a-per-cm",type=float,default=.01)
    compute.add_argument("--mu-s-per-cm",type=float,default=250.)
    compute.add_argument("--g",type=float,default=.9)
    compute.add_argument("--phase-index-ratio",type=float,default=1.35)
    compute.add_argument("--group-index",type=float,default=1.35,
                         help="specified group index; default is an illustrative nondispersive assumption")
    compute.add_argument("--wavelength-nm",type=float,default=633.)
    compute.add_argument("--aperture-deg",type=float,default=90.)
    compute.add_argument("--polar-nodes",type=int,default=48)
    compute.add_argument("--phase-degree",type=int,default=80)
    compute.add_argument("--detector-polar-nodes",type=int,default=64)
    compute.add_argument("--omega-max-per-ns",type=float,default=60.)
    compute.add_argument("--omega-step-per-ns",type=float,default=.2)
    plot = sub.add_parser("plot",help="rebin a saved spectrum without solving transport")
    time_arguments(plot)
    plot.add_argument("--spectrum",type=Path,required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True,exist_ok=True)
    if args.mode == "compute":
        if (not np.isfinite([args.omega_max_per_ns,args.omega_step_per_ns]).all()
                or min(args.omega_max_per_ns,args.omega_step_per_ns)<=0):
            raise ValueError("frequency maximum and step must be finite and positive")
        count = int(math.ceil(args.omega_max_per_ns/args.omega_step_per_ns))
        omega = np.linspace(0.,args.omega_max_per_ns,count+1)
        medium = Medium(100*args.mu_a_per_cm,100*args.mu_s_per_cm,args.g,args.group_index,
                        args.wavelength_nm,"specified illustrative parameters; not an experimental calibration")
        boundary = HalfSpaceBoundary(args.phase_index_ratio)
        settings = HalfSpaceSettings(args.polar_nodes,1,args.phase_degree,plane_parallel=True)
        t=perf_counter();kernel=HalfSpaceSolver(medium,boundary,settings)
        construction=perf_counter()-t
        source=SurfacePort.pencil()
        detector=SurfacePort.aperture(np.deg2rad(args.aperture_deg),
                                      polar_nodes=args.detector_polar_nodes,azimuth_nodes=2)
        spectrum=np.zeros((len(omega),3),complex)
        durations=[]
        t=perf_counter()
        for index,frequency in enumerate(omega):
            one=perf_counter()
            spectrum[index]=kernel.response(source,detector,omega_per_ns=frequency)
            durations.append(perf_counter()-one)
            if index and index%100==0:
                print(f"frequencies {index+1}/{len(omega)}, elapsed {perf_counter()-t:.2f} s",flush=True)
        spectrum_seconds=perf_counter()-t
        specular=float(boundary.coefficients(1.,incidence="outside").reflection)
        parameters={k:getattr(args,k) for k in ['mu_a_per_cm','mu_s_per_cm','g','phase_index_ratio',
                    'group_index','wavelength_nm','aperture_deg','polar_nodes','phase_degree','detector_polar_nodes']}
        files=['src/lighthit/experimental/halfspace.py','src/lighthit/experimental/halfspace_boundary.py',
               'src/lighthit/readout.py','experiments/halfspace_time_profile.py']
        metadata={"schema":1,"created_utc":datetime.now(timezone.utc).isoformat(),"parameters":parameters,
                  "optical_parameter_status":"specified model; mu_a and group index require experimental calibration",
                  "scope":"laterally integrated reflection; normal pencil; no finite receiver offset",
                  "source_sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
        timing={"construction_seconds":construction,"spectrum_seconds":spectrum_seconds,
                "frequencies":len(omega),"angular_directions":len(kernel.mu),
                "mean_seconds_per_frequency":float(np.mean(durations)),
                "first_frequency_seconds":durations[0]}
        np.savez(args.output_directory/"spectrum.npz",omega_per_ns=omega,spectrum=spectrum,
                 specular_fraction=specular,metadata_json=json.dumps(metadata))
    else:
        with np.load(args.spectrum,allow_pickle=False) as data:
            omega=data['omega_per_ns'].copy();spectrum=data['spectrum'].copy()
            specular=float(data['specular_fraction']);metadata=json.loads(str(data['metadata_json']))
        timing={"transport_recomputed":False,"frequencies":len(omega)}
    edges,components,prompt,total,diagnostics=temporal_profile(omega,spectrum,specular,args)
    np.savez(args.output_directory/"profile.npz",edges_ns=edges,components_fraction=components,
             specular_bins_fraction=prompt,total_bins_fraction=total,timing_sigma_ns=args.timing_sigma_ns)
    if not args.no_plot:
        draw(edges,components,prompt,total,metadata,args)
    timing.update(diagnostics)
    timing['wall_seconds']=perf_counter()-started
    (args.output_directory/"run.json").write_text(json.dumps({"metadata":metadata,"timing":timing},
                                                            ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(timing,ensure_ascii=False,indent=2),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
