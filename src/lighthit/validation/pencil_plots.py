"""Optional standalone scientific figures for the pencil-source experiment."""
from pathlib import Path
import numpy as np


def plot_report(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge
    output = Path(output)
    labels = [c['label'] for c in report['cells']]
    indices = np.arange(len(labels))
    rows = report['comparisons']
    with plt.rc_context({'font.size': 10, 'axes.grid': True, 'grid.alpha': .25}):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        for order, ax in enumerate(axes):
            selected = [row for row in rows if row['order'] == ('0', '1', '>=2')[order]]
            mc = np.array([x['mc'] for x in selected])
            se = np.array([x['se_used'] for x in selected])
            rte = np.array([x['rte'] for x in selected])
            nonzero = np.r_[abs(mc[mc != 0]), abs(rte[rte != 0])]
            ax.errorbar(indices-.08, mc, yerr=se, fmt='o', ms=4, capsize=2, label='MC ± SE')
            ax.plot(indices+.08, rte, 'x', ms=6, label='RTE cell average')
            if len(nonzero):ax.set_yscale('symlog', linthresh=float(nonzero.min())*.1)
            ax.set_xticks(indices, labels, rotation=65, ha='right')
            ax.set_title(f"Collision order {('0', '1', '>=2')[order]}")
            ax.set_ylabel('Fluence per emitted photon [m⁻²]')
            ax.legend()
        fig.suptitle('Identical finite-volume observables; no charge normalization')
        fig.savefig(output/'charge.png', dpi=170, metadata={'Creator': 'LightHit'})
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
        colours = dict(cap='#ce6a2a', forward='#2471a3', side='#239b56', backward='#7d3c98')
        seen = set()
        for c in report['cells']:
            category = c['label'].split('-')[-1]
            theta_low, theta_high = np.degrees(np.arccos([c['mu_max'], c['mu_min']]))
            label = category if category not in seen else None
            seen.add(category)
            ax.add_patch(Wedge((0., 0.), c['r_max_m'], 90-theta_high, 90-theta_low,
                              width=c['r_max_m']-c['r_min_m'],
                              facecolor=colours[category], edgecolor=colours[category],
                              alpha=.75, label=label))
        high = max(c['r_max_m'] for c in report['cells'])*1.1
        ax.annotate('', (0., high), (0., 0.), arrowprops=dict(arrowstyle='->', color='black'))
        ax.plot(0., 0., 'ko', label='source')
        ax.set(xlim=(-high*.05, high), ylim=(-high, high), xlabel='Transverse radius ρ [m]',
               ylabel='z [m]', title='Meridional sections of 3D azimuth-averaged probes', aspect='equal')
        ax.legend(loc='lower right')
        fig.savefig(output/'geometry.png', dpi=170, metadata={'Creator': 'LightHit'})
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        reference_name = report.get('comparison_reference', 'reference')
        reference = np.array(report['rte_runs'][reference_name]['components']['real'])[0, :, 2]
        for name, row in report['rte_runs'].items():
            if name == reference_name:continue
            value = np.array(row['components']['real'])[0, :, 2]
            difference = 100*(value-reference)/np.maximum(abs(reference), 1e-300)
            ax.plot(indices, difference, 'o-', ms=3, label=name)
        ax.axhspan(-3., 3., alpha=.08, color='green', label='±3% diagnostic target')
        ax.set_yscale('symlog', linthresh=1.)
        ax.set_xticks(indices, labels, rotation=55, ha='right')
        ax.set(ylabel='Change from fixed reference [%]', title='Order ≥2: independent numerical refinements')
        ax.legend(ncol=3)
        fig.savefig(output/'convergence.png', dpi=170, metadata={'Creator': 'LightHit'})
        plt.close(fig)

        if 'time' in report:
            t = report['time'];edges = np.array(t['view_edges_ns']);centers = (edges[:-1]+edges[1:])/2
            width = np.diff(edges)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
            for order, ax in zip((1, 2), axes):
                raw = np.array(t['raw_mc_view'])[:, order]/width
                se = np.array(t['raw_mc_view_se'])[:, order]/width
                filtered = np.array(t['filtered_mc_bins'])[:, order]/width
                rte = np.array(t['rte_bins'])[:, order]/width
                ax.errorbar(centers, raw, yerr=se, fmt='.', ms=3, alpha=.7, label='MC raw time bins')
                ax.plot(centers, filtered, label='MC, same finite Fourier grid')
                ax.plot(centers, rte, '--', label='RTE, finite Fourier grid')
                ax.axvline(t['causal_front_ns'], color='gray', lw=1, label='earliest cell front')
                ax.set(xlabel='Time since emission [ns]', ylabel='Fluence rate [m⁻² ns⁻¹]',
                       title=f"{t['cell']}, order {('0','1','>=2')[order]}")
                ax.legend(fontsize=8)
            fig.savefig(output/'time.png', dpi=170, metadata={'Creator': 'LightHit'})
            plt.close(fig)
