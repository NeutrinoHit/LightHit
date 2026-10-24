"""Student experiment: one directed photon, exact low orders, RTE and MC.

    python -m lighthit.examples.pencil_validation --profile quick --output results/pencil
    python -m lighthit.examples.pencil_validation --profile full --output results/pencil-full

No Geant4, detector bundle or prepared event tables are required.
"""
from pathlib import Path
import argparse
import hashlib
import json

from ..validation.pencil_experiment import run_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', choices=('quick', 'full'), default='quick')
    parser.add_argument('--output', type=Path, default=Path('results/pencil-validation'))
    parser.add_argument('--photons-per-batch', type=int)
    parser.add_argument('--batches', type=int)
    parser.add_argument('--seed', type=int, default=78123)
    parser.add_argument('--target-relative-error', type=float, default=.03)
    parser.add_argument('--resume', action='store_true', help='reuse only data with matching inputs and code hashes')
    parser.add_argument('--time', action=argparse.BooleanOptionalAction, default=None,
                        help='one-cell time/Fourier control; enabled by the full profile')
    parser.add_argument('--model', type=Path, help='optional BGVD model: use only its water at --wavelength')
    parser.add_argument('--wavelength', type=float, default=450.)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--strict', action='store_true', help='nonzero exit on discrepancy or unresolved precision')
    args = parser.parse_args(argv)
    medium = None
    if args.model is not None:
        from ..bgvd import load_model
        medium = load_model(args.model).medium.band(args.wavelength)
    report = run_experiment(args.output, profile=args.profile, medium=medium,
        photons_per_batch=args.photons_per_batch, batches=args.batches, seed=args.seed,
        target_relative_error=args.target_relative_error, resume=args.resume, time_check=args.time,
        progress=lambda value: print(value, flush=True))
    if args.model is not None:
        report['input_model'] = dict(filename=args.model.name,
            sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(), wavelength_nm=args.wavelength)
        (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if not args.no_plots:
        try:
            from ..validation.pencil_plots import plot_report
            plot_report(report, args.output)
        except ImportError as error:
            raise ImportError("plots require pip install 'lighthit[validation]'; numerical files were saved") from error
    print(f"Report: {(args.output/'REPORT.md').resolve()}")
    if args.strict:
        if (report['no_scattering_maximum_absolute_error'] > 1e-10
                or report['global_check']['maximum_pull'] > 6
                or report['global_check']['rte_zero_k_relative_error'] > 1e-10
                or max(abs(x['relative_difference']) for x in report['first_order_independent_quadrature']) > 1e-7
                or report['status_counts']['discrepancy']):
            return 1
        if any(report['status_counts'][key] for key in
               ('insufficient_histories', 'statistics_limited', 'resolution_limited')):
            return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
