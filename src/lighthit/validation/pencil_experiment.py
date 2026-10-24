"""Reproducible pencil-source experiment and its numerical controls.

All parameter profiles are fixed before the run. Refinements are reported,
not selected to improve agreement with a particular Monte Carlo realization.
"""
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import csv
import hashlib
import json
import platform
import tempfile
from zipfile import BadZipFile
import numpy as np

from .. import __version__
from ..green import SolverSettings, PointGreenSolver
from ..medium import synthetic_medium
from ..readout import inverse_bins
from .cells import AxisymmetricCell
from .beam_mc import estimate_cells, BeamMCResult
from .beam_rte import cell_spectra, BeamRTEResult, global_spectra, sphere_ballistic_probability

ORDERS = ("0", "1", ">=2")


def default_cells(profile="quick"):
    radii = (30., 125.) if profile == "quick" else (10., 30., 125.)
    angles = [("cap", .995, 1.), ("forward", .8, .95),
              ("side", .1, .4), ("backward", -.8, -.4)]
    return tuple(AxisymmetricCell(r-1., r+1., lo, hi, f"{r:g}m-{label}")
                 for r in radii for label, lo, hi in angles)


def _code_hash():
    from . import beam_mc, beam_rte, cells
    from .. import angular, green, single, medium, _angular_numba, single_fast
    h = hashlib.sha256()
    for module in (beam_mc, beam_rte, cells, angular, green, single, medium,
                   _angular_numba, single_fast):
        h.update(Path(module.__file__).read_bytes())
    import scipy, numba
    h.update(f"numpy={np.__version__};scipy={scipy.__version__};numba={numba.__version__}".encode())
    return h.hexdigest()


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _complex(value):
    a = np.asarray(value)
    return dict(real=a.real.tolist(), imag=a.imag.tolist())


def _load_arrays(path, key):
    """Reject interrupted, mismatched or nonfinite cached calculations."""
    try:
        with np.load(path, allow_pickle=False) as d:
            metadata = json.loads(str(d['metadata']))
            if metadata['key'] != key:
                return None
            arrays = {name:d[name] for name in d.files if name != 'metadata'}
            if not all(np.isfinite(a).all() for a in arrays.values()):
                return None
        return arrays, metadata
    except (OSError, EOFError, ValueError, KeyError, TypeError, BadZipFile):
        return None


def _save_arrays(path, *, metadata, **arrays):
    """An interrupted write must not replace a complete numerical artifact."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.stem+'-',
                                     suffix='.npz', delete=False) as file:
        temporary = Path(file.name)
    try:
        np.savez_compressed(temporary, **arrays, metadata=json.dumps(metadata))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _cached_rte(output, label, medium, cells, omega, settings, radial, polar,
                resume, source="beam", progress=None):
    inputs = dict(medium=asdict(medium), cells=[asdict(c) for c in cells],
                  omega=list(omega), settings=asdict(settings), radial=radial,
                  polar=polar, source=source, code_sha256=_code_hash())
    path = output/f"rte-{label}.npz"
    if resume and path.exists():
        cached = _load_arrays(path, _key(inputs))
        if cached is not None:
            arrays, meta = cached
            if arrays.get('components', np.empty(0)).shape == (len(omega), len(cells), 3):
                return BeamRTEResult(tuple(cells), np.asarray(omega), arrays['components'],
                                      settings, dict(meta['diagnostics'], cache_reused=True))
    result = cell_spectra(medium, cells, omega, settings=settings,
                         radial_order=radial, polar_order=polar, source=source,
                         angular_backend="numba", single_backend="numba", progress=progress)
    result.diagnostics['cache_reused'] = False
    _save_arrays(path, components=result.components,
        metadata=dict(key=_key(inputs), inputs=inputs, diagnostics=result.diagnostics))
    return result


def _cached_mc(output, label, medium, cells, omega, photons, batches, seed,
               resume, *, edges=(), fraction=0., mode="implicit", roulette=1e-5,
               source="beam", progress=None):
    inputs = dict(medium=asdict(medium), cells=[asdict(c) for c in cells],
                  omega=list(omega), photons=photons, batches=batches, seed=seed,
                  edges=list(edges), fraction=fraction, mode=mode, roulette=roulette,
                  source=source, code_sha256=_code_hash())
    path = output/f"mc-{label}.npz"
    if resume and path.exists():
        cached = _load_arrays(path, _key(inputs))
        if cached is not None:
            a, meta = cached
            shapes = dict(components=(batches,len(omega),len(cells),3),
                          global_components=(batches,len(omega),3),
                          second=(batches,len(omega),len(cells)+1,4,2),
                          bins=(batches,len(cells),max(0,len(edges)-1),3))
            if all(a.get(k, np.empty(0)).shape == shape for k,shape in shapes.items()):
                return BeamMCResult(tuple(cells), np.asarray(omega), a['components'],
                    a['global_components'], a['second'], np.asarray(edges), a['bins'],
                    photons, seed, dict(meta['diagnostics'], cache_reused=True))
    result = estimate_cells(medium, cells, omega_per_ns=omega,
        photons_per_batch=photons, batches=batches, seed=seed,
        time_edges_ns=edges, isotropic_fraction=fraction, absorption_mode=mode,
        roulette_weight=roulette, source=source, progress=progress)
    result.diagnostics['cache_reused'] = False
    _save_arrays(path, components=result.batch_components,
        global_components=result.batch_global, second=result.second_moments,
        bins=result.batch_bins, metadata=dict(key=_key(inputs), inputs=inputs,
                                             diagnostics=result.diagnostics))
    return result


def _sphere_controls(medium, photons, seed):
    """Physical Q0 hit counter, independent of the volume-density experiment."""
    rng = np.random.default_rng(seed)
    radius = .216
    first_scatter = rng.exponential(1/medium.scattering_per_m, photons) if medium.scattering_per_m else np.full(photons, np.inf)
    rows = []
    for z in (30., 125., -30.):
        for rho in (0., radius*.99, radius*1.01):
            expected = sphere_ballistic_probability(medium, [rho, 0., z], radius)
            # Independent elementary ray geometry, rather than calling the
            # sphere intersection implementation under test a second time.
            entry = z-np.sqrt(radius*radius-rho*rho) if rho < radius and z > 0 else np.inf
            hits = first_scatter > entry
            p = hits.mean()
            absorption = np.exp(-medium.absorption_per_m*entry)
            value = p*absorption
            sigma = absorption*np.sqrt(p*(1-p)/(photons-1))
            rows.append(dict(rho_m=rho, z_m=z, radius_m=radius, expected=expected,
                             mc=float(value), sigma=float(sigma), hit_histories=int(hits.sum()),
                             analytic_clock_ns=(float(entry/medium.speed_m_per_ns)
                                                if np.isfinite(entry) else None)))
    return rows


def _comparisons(cells, reference, mc, refinements, target, coarse_refinements=()):
    zero = int(np.flatnonzero(mc.omega_per_ns == 0.)[0])
    batch_se = mc.standard_error()[zero, ..., 0]
    history_se = mc.history_standard_error()[zero, ..., :3, 0]
    # Predeclared conservative diagnostic, not a posterior change of criterion.
    error = np.maximum(batch_se, history_se)
    mean = mc.components[zero].real
    effective = mc.effective_histories()
    sensitivity = np.zeros_like(mean)
    for result in refinements:
        sensitivity = np.maximum(sensitivity, abs(result.components[zero]-reference.components[zero]))
    coarse_sensitivity = np.zeros_like(mean)
    for result in coarse_refinements:
        coarse_sensitivity = np.maximum(coarse_sensitivity,
            abs(result.components[zero]-reference.components[zero]))
    rows = []
    for ic, cell in enumerate(cells):
        for order in range(3):
            observed, expected, sigma = float(mean[ic, order]), float(reference.components[zero, ic, order].real), float(error[ic, order])
            scale = max(abs(observed), abs(expected), 1e-300)
            pull = (expected-observed)/sigma if sigma else None
            if observed == 0. and expected == 0.:
                status = "structural_zero"
            elif observed == 0. or effective[ic, order] < 50:
                status = "insufficient_histories"
            elif sigma/scale > target:
                status = "statistics_limited"
            elif sensitivity[ic, order]/scale > target:
                status = "resolution_limited"
            elif pull is not None and abs(pull) > 3:
                status = "discrepancy"
            else:
                status = "consistent"
            rows.append(dict(cell=cell.label, order=ORDERS[order], rte=expected, mc=observed,
                             se_batch=float(batch_se[ic, order]), se_history=float(history_se[ic, order]),
                             se_used=sigma, effective_histories=float(effective[ic, order]),
                             observed_rte_spread=float(sensitivity[ic, order]), pull=pull, status=status))
            rows[-1]['coarse_rte_spread'] = float(coarse_sensitivity[ic, order])
    return rows


def _markdown(report):
    lines = ["# Направленный фотон: RTE, аналитика и независимый МК", "",
             "Источник — один фотон в начале координат, направление +z. "
             "В таблице измеряется объёмно усреднённая скалярная флюенсность [м⁻²]. "
             "Все повторные проходы через ячейку учитываются. Это не вероятность первого входа в OM.", "",
             f"Профиль: `{report['profile']}`. Фотонов: {report['mc']['photons']}. "
             f"Среда: μa={report['medium']['absorption_per_m']:g} м⁻¹, "
             f"μs={report['medium']['scattering_per_m']:g} м⁻¹, g={report['medium']['g']:g}.", "",
             "SE — максимум заранее выбранных оценок по независимым блокам и полным "
             "историям. Статусы различают согласие, недостаток историй/статистики, "
             "чувствительность к разрешению и расхождение. 3 SE — диагностический "
             "критерий одной строки, не одновременная гарантия всех строк. "
             "Наблюдаемый разброс RTE при уточнении не является строгой границей ошибки. "
             f"Сравнение использует заранее заданный профиль `{report['comparison_reference']}`. "
             "Чувствительность грубых профилей хранится отдельно.", "",
             "| Ячейка | Порядок | RTE | МК ± SE | N_eff | Pull | Статус |",
             "|---|---|---:|---:|---:|---:|---|"]
    for row in report['comparisons']:
        pull = f"{row['pull']:.2f}" if row['pull'] is not None else "—"
        lines.append(f"| {row['cell']} | {row['order']} | {row['rte']:.5g} | "
                     f"{row['mc']:.5g} ± {row['se_used']:.2g} | {row['effective_histories']:.0f} | {pull} | {row['status']} |")
    lines += ["", "## Нормировка по всему пространству", "",
              "Аналитические спектры: Q0=1/d, Q1=μs/d², Q≥2=μs²/(a d²), "
              "где a=μa−iω/v, d=μa+μs−iω/v. Сумма=1/a. "
              "Нормировка проверяет также частотную фазу и разделение порядков.", "",
              f"Максимальный нормированный остаток МК: {report['global_check']['maximum_pull']:.3g} SE. "
              f"Ошибка RTE при k=0: {report['global_check']['rte_zero_k_relative_error']:.3g} относительно. "
              "Независимая χ-квадратура Q1 в регулярных точках отличается максимум на "
              f"{max(abs(x['relative_difference']) for x in report['first_order_independent_quadrature']):.3g} относительно.", "",
              "## Разрешение", "",
              "| Расчёт | L рассеяния | J пространства | k_max [м⁻¹] | Δk панели | Время [с] |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, row in report['rte_runs'].items():
        s = row['settings']
        lines.append(f"| {label} | {s['scattering_degree']} | {s['spatial_degree']} | "
                     f"{s['k_max_per_m']:g} | {s['k_panel_per_m']:g} | {row['seconds']:.2f} |")
    lines += ["", "## Как читать результат", "",
              "- Полярная шапка включает баллистическую ось: Q0 конечен после усреднения. "
              "На математической оси точечные ответы являются распределениями/сингулярностями.",
              "- Вне оси Q0 строго ноль. Уменьшение ячейки проверяет переход к регулярной точке; "
              "шапка, сжимающаяся к оси, регулярного точечного предела не имеет.",
              "- `rte-cell-refinement.npz` и `point-centres.npz` разделяют усреднение по объёму "
              "и ошибку решения при фиксированном разрешении.",
              "- `mc-physical-hg.npz`, `mc-analog.npz` и `mc-roulette.npz` проверяют "
              "альтернативные несмещённые оценки; их статистика может быть ограничена.",
              "- Никакие заряды и временные бины не обрезаются и не перенормируются.",
              "- Для Q0 конечной сферы отдельно проверяется вероятность первого входа "
              "(без чувствительности, эффективность=1), единицы — безразмерные."]
    lines += ["", "## Дополнительные оценки МК", "",
              "Это проверки альтернативных оценок и предельных случаев. Их меньшая "
              "статистика не обязана обеспечивать основной допуск 3%.", "",
              "| Контроль | Максимальный модуль pull | Максимальная относительная SE |",
              "|---|---:|---:|"]
    for name, check in report['control_checks'].items():
        lines.append(f"| {name} | {check['maximum_pull']:.3g} | "
                     f"{100*check['maximum_relative_standard_error']:.2f}% |")
    if 'time' in report:
        t = report['time']
        lines += ["", "## Время", "",
                  f"Ячейка `{t['cell']}`, период частотной сетки {t['period_ns']:.1f} нс. "
                  "Сравниваются исходная гистограмма МК, её восстановление на той же "
                  "конечной частотной сетке и RTE. Отличие первых двух измеряет "
                  "эффект преобразования Фурье, не транспортной модели.",
                  f"Гистограмма покрывает {100*t['window_fraction']:.4f}% измеренного "
                  "МК-заряда. Контроль дополнительного укрупнения гистограммы: "
                  f"{t['histogram_fourier_l1_change']:.3g} по нормированной L1. "
                  f"Уточнение шага частоты: {t['frequency_grid_l1_change']:.3g}; "
                  f"изменение верхнего среза вдвое: {t['half_cutoff_l1_change']:.3g}. "
                  f"L1 RTE против МК на одинаковой частотной сетке: {t['rte_filtered_mc_l1']:.3g}; "
                  f"МК до/после частотного ограничения: {t['raw_filtered_mc_l1']:.3g}. "
                  "Это самостоятельные диагностики Фурье; статус таблицы зарядов не означает "
                  "доказанной точности неограниченного временного спектра."]
    lines += ["", "Полные настройки, семена, численные данные и ошибки: `report.json`, "
              "`comparison.csv` и сохранённые массивы NPZ. Графики: `charge.png`, "
              "`geometry.png`, `convergence.png` и, если выбран временной контроль, `time.png`.", ""]
    return "\n".join(lines)


def run_experiment(output, *, profile="quick", medium=None, photons_per_batch=None,
                   batches=None, seed=78123, target_relative_error=.03, resume=False,
                   time_check=None, progress=print):
    """Run predeclared controls and save a self-contained numerical experiment."""
    if profile not in ("quick", "full"):
        raise ValueError("profile must be quick or full")
    if not np.isfinite(target_relative_error) or target_relative_error <= 0:
        raise ValueError("target_relative_error must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    medium = medium or synthetic_medium()
    cells = default_cells(profile)
    photons = photons_per_batch if photons_per_batch is not None else (5000 if profile == "quick" else 50000)
    batches = batches if batches is not None else (12 if profile == "quick" else 32)
    omega = np.array([0., .03, .1])
    base = SolverSettings(24, 128, 4., .05, 8)
    configurations = dict(baseline=base, scattering=replace(base, scattering_degree=48),
                          spatial=replace(base, spatial_degree=256))
    if profile == "full":
        configurations.update(cutoff=replace(base, k_max_per_m=6.),
                              panels=replace(base, k_panel_per_m=.025))
    reference_settings = (SolverSettings(48, 256, 6., .025, 8) if profile == "quick" else
                          SolverSettings(64, 384, 8., .025, 10))
    configurations['reference'] = reference_settings
    if profile == 'full':
        configurations.update(
            event_bandwidth=replace(base, spatial_degree=32),
            verify_scattering=replace(reference_settings, scattering_degree=96),
            verify_spatial=replace(reference_settings, spatial_degree=512),
            verification=SolverSettings(96, 512, 10., .025, 10))
    began = perf_counter()
    def note(value):
        if progress is not None: progress(value)
    fraction = 0. if profile == "quick" else .25
    note("Monte Carlo: directed source, independent batches")
    mc = _cached_mc(output, 'main', medium, cells, omega, photons, batches, seed,
                    resume, fraction=fraction,
                    progress=lambda i, n: note(f"MC batch {i}/{n}") if i == n or i % 4 == 0 else None)
    results = {}
    for label, settings in configurations.items():
        note(f"RTE: {label}")
        results[label] = _cached_rte(output, label, medium, cells, omega, settings,
                                    12 if label == 'verification' else 8,
                                    32 if label == 'verification' else 24, resume)
    if profile == "full":
        note("RTE: independent volume-quadrature refinement")
        results['averaging'] = _cached_rte(output, 'averaging', medium, cells, omega,
                                            base, 16, 48, resume)
    note("Finite-cell versus regular point limit")
    refined_cells = tuple(c.refined(.5) for c in cells)
    refined = _cached_rte(output, 'cell-refinement', medium, refined_cells, omega,
                          base, 8, 24, resume)
    # The point output is diagnostic: it must not replace the cell average
    # when comparing with volume-track-length Monte Carlo.
    point = PointGreenSolver(medium, base, angular_backend='numba').solve(
        omega, np.array([c.centre_m for c in cells]))
    point.save(output/'point-centres.npz')
    zero_medium = replace(medium, scattering_per_m=0.)
    no_scatter = _cached_mc(output, 'no-scattering', zero_medium, cells, omega,
                             4, 2, seed+300, resume)
    expected_zero = cell_spectra(zero_medium, cells, omega).components
    ballistic_error = float(np.max(abs(no_scatter.components-expected_zero)))
    note("Controls: analog absorption, HG proposal and roulette")
    near = tuple(c for c in cells if c.r_max_m <= 32.)
    controls = {}
    control_rte = {}
    for label, mode, proposal, roulette in [('analog', 'analog', 0., 1e-5),
            ('physical-hg', 'implicit', 0., 1e-5), ('roulette', 'implicit', fraction, .001)]:
        controls[label] = _cached_mc(output, label, medium, near, [0.],
            max(1000, photons//4), max(8, batches//2), seed+10+len(controls),
            resume, fraction=proposal, mode=mode, roulette=roulette)
    if profile == "full":
        note("Limits: isotropic source and isotropic scattering")
        controls['isotropic-source'] = _cached_mc(output, 'isotropic-source', medium,
            near, [0.], photons//4, max(8, batches//2), seed+20, resume, source='isotropic')
        control_rte['isotropic-source'] = _cached_rte(output, 'isotropic-source', medium,
            near, [0.], base, 8, 24, resume, source='isotropic')
        isotropic_medium = replace(medium, g=0.)
        controls['isotropic-scattering'] = _cached_mc(output, 'isotropic-scattering',
            isotropic_medium, near, [0.], photons//4, max(8, batches//2), seed+21, resume)
        control_rte['isotropic-scattering'] = _cached_rte(output, 'isotropic-scattering',
            isotropic_medium, near, [0.], replace(base, scattering_degree=0), 8, 24, resume)
    comparison_reference = 'verification' if profile == 'full' else 'reference'
    reference = results[comparison_reference]
    fine_names = ('reference', 'verify_scattering', 'verify_spatial')
    refinements = ([results[k] for k in fine_names] if profile == 'full' else list(results.values()))
    rows = _comparisons(cells, reference, mc, refinements, target_relative_error,
                        coarse_refinements=list(results.values()))
    control_checks = {}
    for name, sample in controls.items():
        expected = (control_rte[name].components[0].real if name in control_rte
                    else reference.components[0, :len(near)].real)
        mean = sample.components[0].real
        error = np.maximum(sample.standard_error()[0, ..., 0],
                           sample.history_standard_error()[0, ..., :3, 0])
        residual = np.divide(abs(mean-expected), error,
                             out=np.zeros_like(mean), where=error > 0)
        relative_error = np.divide(error, abs(mean), out=np.zeros_like(mean), where=mean != 0)
        control_checks[name] = dict(maximum_pull=float(residual.max()),
            maximum_relative_standard_error=float(relative_error.max()),
            precision_scope='supplementary control, not the main 3% precision requirement')
    global_mean = mc.batch_global.mean(0)
    global_error = np.maximum(mc.standard_error(global_space=True),
                              mc.history_standard_error(global_space=True)[..., :3, :])
    global_exact = global_spectra(medium, omega)
    residual = np.stack(((global_mean-global_exact).real, (global_mean-global_exact).imag), axis=-1)
    pull = np.divide(abs(residual), global_error, out=np.zeros_like(residual), where=global_error > 0)
    from ..angular import angular_components
    zero_k = np.array([[v[0, 0]/np.sqrt(2) for v in angular_components(
        np.array([0.]), w, medium, base.scattering_degree, 0, backend='numba')] for w in omega])
    from ..prompt_reference import pencil_order1_bins
    first_controls = []
    for ic, c in enumerate(cells):
        p = c.centre_m;radius = float(np.linalg.norm(p));theta = float(np.arccos(p[2]/radius))
        q, _ = pencil_order1_bins(radius, theta, 1., 0., medium, np.array([4*np.pi]), np.array([0., 1.]))
        first_controls.append(dict(cell=c.label, chi_quadrature=float(q),
            point_solver=float(point.components[0, ic, 1].real),
            relative_difference=float(point.components[0, ic, 1].real/q-1)))
    report = dict(profile=profile, medium=asdict(medium), cells=[asdict(c) for c in cells],
        omega_per_ns=omega.tolist(), package_version=__version__, code_sha256=_code_hash(),
        machine=dict(platform=platform.platform(), numpy=np.__version__),
        mc=mc.diagnostics, seed=seed, batches=batches, photons_per_batch=photons,
        target_relative_error=target_relative_error, comparisons=rows,
        comparison_reference=comparison_reference,
        global_check=dict(expected=_complex(global_exact), measured=_complex(global_mean),
                          standard_error=global_error.tolist(), maximum_pull=float(pull.max()),
                          rte_zero_k=_complex(zero_k),
                          rte_zero_k_relative_error=float(np.max(abs(zero_k-global_exact)/np.maximum(abs(global_exact),1e-300)))),
        first_order_independent_quadrature=first_controls,
        no_scattering_maximum_absolute_error=ballistic_error,
        sphere_controls=_sphere_controls(medium, max(10000, photons*batches), seed+100),
        rte_runs={name:dict(settings=asdict(x.settings), seconds=x.diagnostics['seconds'],
                           components=_complex(x.components)) for name, x in results.items()},
        cell_refinement=_complex(refined.components), point_centres=_complex(point.components),
        control_checks=control_checks,
        control_mc={name:dict(diagnostics=x.diagnostics,
                              global_measured=_complex(x.batch_global.mean(0)),
                              cells=[asdict(c) for c in x.cells],
                              components=_complex(x.components),
                              standard_error=x.standard_error().tolist(),
                              history_standard_error=x.history_standard_error().tolist(),
                              rte=_complex(control_rte[name].components) if name in control_rte else None)
                    for name, x in controls.items()})
    if time_check is None: time_check = profile == "full"
    if time_check:
        note("Time: one regular off-axis cell")
        cell = next(c for c in cells if c.label == '30m-forward')
        grid = np.linspace(0., .6, 97) if profile == 'quick' else np.linspace(0., 1.2, 385)
        time_settings = reference_settings if profile == 'quick' else SolverSettings(64, 384, 12., .025, 10)
        tr = _cached_rte(output, 'time', medium, [cell], grid,
                        time_settings, 12 if profile == 'full' else 8, 24, resume,
                        progress=lambda i, n: note(f"Time RTE frequency {i}/{n}") if i % 16 == 0 else None)
        edges = np.arange(0., 1000.5, .5)
        tm = _cached_mc(output, 'time', medium, [cell], omega,
            max(2000, photons//2), max(8, batches//2), seed+200, resume,
            fraction=.25, edges=edges)
        mean_bins = tm.batch_bins.mean(0)[0]
        centers = (edges[:-1]+edges[1:])/2
        # A fine raw histogram supplies an independent Fourier approximation;
        # coarsening it quantifies this additional numerical approximation.
        fourier = np.einsum('to,wt->wo', mean_bins, np.exp(1j*grid[:, None]*centers))
        coarse = mean_bins.reshape(-1, 2, 3).sum(1)
        coarse_time = centers.reshape(-1, 2).mean(1)
        coarse_fourier = np.einsum('to,wt->wo', coarse, np.exp(1j*grid[:, None]*coarse_time))
        view_edges = np.arange(90., 500.1, 5.)
        locations = np.searchsorted(edges, view_edges)
        raw_view = np.stack([tm.batch_bins[:, 0, a:b].sum(1)
                             for a, b in zip(locations[:-1], locations[1:])], axis=1)
        raw_view_mean = raw_view.mean(0)
        raw_view_error = raw_view.std(0, ddof=1)/np.sqrt(len(raw_view))
        values = np.stack([inverse_bins(grid, tr.components[:, :, o], view_edges)[0] for o in range(3)], axis=-1)
        filtered = np.stack([inverse_bins(grid, fourier[:, o:o+1], view_edges)[0] for o in range(3)], axis=-1)
        coarser = np.stack([inverse_bins(grid, coarse_fourier[:, o:o+1], view_edges)[0] for o in range(3)], axis=-1)
        frequency_coarse = np.stack([inverse_bins(grid[::2], tr.components[::2, :, o], view_edges)[0]
                                      for o in range(3)], axis=-1)
        half = (len(grid)+1)//2
        cutoff_coarse = np.stack([inverse_bins(grid[:half], tr.components[:half, :, o], view_edges)[0]
                                  for o in range(3)], axis=-1)
        norm = max(abs(values).sum(),1e-300)
        report['time'] = dict(cell=cell.label, period_ns=float(2*np.pi/(grid[1]-grid[0])),
            omega_max_per_ns=float(grid[-1]), frequencies=len(grid),
            settings=asdict(time_settings),
            window_fraction=float(mean_bins.sum()/tm.components[0, 0].real.sum()),
            rte_filtered_mc_l1=float(abs(values-filtered).sum()/norm),
            raw_filtered_mc_l1=float(abs(raw_view_mean-filtered).sum()/max(abs(raw_view_mean).sum(), 1e-300)),
            histogram_fourier_l1_change=float(abs(filtered-coarser).sum()/max(abs(filtered).sum(), 1e-300)),
            frequency_grid_l1_change=float(abs(frequency_coarse-values).sum()/norm),
            half_cutoff_l1_change=float(abs(cutoff_coarse-values).sum()/norm),
            causal_front_ns=float(cell.r_min_m/medium.speed_m_per_ns),
            histogram_edges_ns=edges.tolist(), raw_mc_bins=mean_bins.tolist(),
            view_edges_ns=view_edges.tolist(), raw_mc_view=raw_view_mean.tolist(),
            raw_mc_view_se=raw_view_error.tolist(),
            rte_bins=values.tolist(), filtered_mc_bins=filtered.tolist())
    report['seconds'] = perf_counter()-began
    report['numerical_work_seconds'] = sum(x.diagnostics['seconds'] for x in results.values())
    report['numerical_work_seconds'] += sum(x.diagnostics['seconds'] for x in controls.values())
    report['numerical_work_seconds'] += mc.diagnostics['seconds']+refined.diagnostics['seconds']+no_scatter.diagnostics['seconds']+point.timings_s['total']
    if time_check:report['numerical_work_seconds'] += tr.diagnostics['seconds']+tm.diagnostics['seconds']
    report['status_counts'] = {label:sum(r['status'] == label for r in rows) for label in
        ('consistent', 'structural_zero', 'insufficient_histories', 'statistics_limited',
         'resolution_limited', 'discrepancy')}
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (output/'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/'REPORT.md').write_text(_markdown(report))
    note(f"Finished: {report['status_counts']}")
    return report
