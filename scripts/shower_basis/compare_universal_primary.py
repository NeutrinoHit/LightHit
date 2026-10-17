#!/usr/bin/env python3
"""Compare a new primary run with a saved run and its independent controls."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(reference, candidate):
    old=json.loads((reference/'report.json').read_text())
    new=json.loads((candidate/'report.json').read_text())
    for key in ('source_sha256','cache_manifest_sha256'):
        if old[key]!=new[key]:raise ValueError('different '+key)
    with np.load(reference/'response.npz') as a,np.load(candidate/'response.npz') as b:
        for key in ('positions_m','receiver_indices','relative_time_edges_ns','time_origin_ns'):
            np.testing.assert_array_equal(a[key],b[key],err_msg='different '+key)
        q=b['charge_orders_pe'].copy();bins=b['bins_orders_pe'].copy()
        oldq=a['charge_orders_pe'];oldb=a['bins_orders_pe']
        active=b['active'].copy();ids=b['receiver_indices'].copy()
        computed=np.asarray(new['metadata']['time_computed_mask'])
        common=computed & np.asarray(old['metadata']['time_computed_mask'])
        total=oldq.sum(1);bright=(a['active']|active)
        summary=dict(
            reference_seconds=old['metadata']['elapsed_seconds'],candidate_seconds=new['metadata']['elapsed_seconds'],
            reference_cpu_seconds=old['metadata']['cpu_seconds'],candidate_cpu_seconds=new['metadata']['cpu_seconds'],
            wall_speedup=old['metadata']['elapsed_seconds']/new['metadata']['elapsed_seconds'],
            cpu_speedup=old['metadata']['cpu_seconds']/new['metadata']['cpu_seconds'],
            active=int(active.sum()),activity_changes=ids[a['active']!=active].tolist(),
            max_charge_absolute_change_pe=float(abs(q-oldq).max()),
            max_active_total_charge_relative_change=float(np.max(abs(q.sum(1)[bright]-total[bright])/np.maximum(abs(total[bright]),1e-30),initial=0)),
            max_common_total_bins_relative_l1_change=float(np.max(abs(bins.sum(2)[common]-oldb.sum(2)[common]).sum(1)/np.maximum(abs(oldb.sum(2)[common]).sum(1),1e-30),initial=0)),
            process_peak_mib=new['process_peak_mib'],timings=new['metadata']['timings'],controls=[])
    for control in old['controls']:
        om=control['om'];index=int(np.flatnonzero(ids==om)[0])
        with np.load(reference/f'control-{om}.npz') as z:
            refq=z['reference_charge'];refb=z['reference_bins']
        summary['controls'].append(dict(om=om,
            reference_total_charge_pe=float(refq.sum()),
            total_charge_absolute_error_pe=float(abs(q[index].sum()-refq.sum())),
            charge_relative=(abs(q[index]-refq)/np.maximum(abs(refq),1e-30)).tolist(),
            total_charge_relative=float(abs(q[index].sum()-refq.sum())/max(abs(refq.sum()),1e-30)),
            bins_relative_l1=(abs(bins[index]-refb).sum(0)/np.maximum(abs(refb).sum(0),1e-30)).tolist() if computed[index] else None,
            total_bins_relative_l1=float(abs(bins[index].sum(1)-refb.sum(1)).sum()/max(abs(refb.sum(1)).sum(),1e-30)) if computed[index] else None,
            candidate_active=bool(active[index]),direct_active=control['direct_active']))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.reference,args.candidate)
    (args.candidate/'comparison.json').write_text(json.dumps(result,indent=2))
    controls=result['controls']
    display={k:v for k,v in result.items() if k!='controls'}
    display.update(control_count=len(controls),
        max_control_total_charge_relative=max((r['total_charge_relative'] for r in controls),default=None),
        max_active_control_total_charge_relative=max((r['total_charge_relative'] for r in controls if r['direct_active']),default=None),
        max_inactive_control_charge_absolute_error_pe=max((r['total_charge_absolute_error_pe'] for r in controls if not r['direct_active']),default=None),
        max_control_total_bins_relative_l1=max((r['total_bins_relative_l1'] for r in controls if r['total_bins_relative_l1'] is not None),default=None),
        control_activity_disagreements=[r['om'] for r in controls if r['candidate_active']!=r['direct_active']])
    print(json.dumps(display,indent=2))


if __name__=='__main__':main()
