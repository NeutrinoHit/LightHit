"""Student pencil-source laboratory, requiring no detector/event data bundle."""
from pathlib import Path
from .cases import case


def _run(ctx, profile):
    from lighthit.validation import beam_mc, beam_rte, cells, pencil_experiment
    from .record import RESULTS
    for module in (beam_mc, beam_rte, cells, pencil_experiment):
        ctx._note(Path(module.__file__))
    output = RESULTS/'pencil-green-validation'/f'{profile}-artifacts'
    report = pencil_experiment.run_experiment(output, profile=profile, resume=True, progress=None)
    return dict(headline=dict(
        photons=report['mc']['photons'], cells=len(report['cells']),
        consistent=report['status_counts']['consistent'],
        structural_zeros=report['status_counts']['structural_zero'],
        discrepancies=report['status_counts']['discrepancy'],
        global_maximum_pull=report['global_check']['maximum_pull'],
        numerical_work_seconds=report['numerical_work_seconds'],
        invocation_seconds=report['seconds'], cached_mc=report['mc']['cache_reused']),
        status_counts=report['status_counts'], medium=report['medium'],
        code_sha256=report['code_sha256'],
        artifacts=str(output.relative_to(RESULTS)),
        time_controls={k:report['time'][k] for k in ('frequency_grid_l1_change',
            'half_cutoff_l1_change','histogram_fourier_l1_change')} if 'time' in report else None,
        production_route_changed=False,
        timing_scope='validation invocation; matching stored arrays may be reused; numerical_work_seconds sums their generation timings')


@case('pencil-green-validation','quick',kind='metrics',warmup=False,
      notes='Standalone synthetic directed photon; may reuse parameter/code-hashed arrays; not an event speed benchmark')
def quick(ctx):
    return _run(ctx,'quick')


@case('pencil-green-validation','full',kind='metrics',warmup=False,heavy=True,
      notes='Predeclared MC/RTE refinements, limits and time controls; may reuse matching arrays; no event tables')
def full(ctx):
    return _run(ctx,'full')
