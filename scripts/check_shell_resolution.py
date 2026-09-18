"""Refine shell width, flight quadrature and maximum path in the MC check.

Changes of maximum path alter random-number consumption between photons; that
comparison contains statistical variation as well as any cutoff bias.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from lighthit.experimental.shell_mc import shell_estimate, ratio_with_standard_error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='.build/absorption')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    runs={}
    for name,width,max_path,quadrature in [('base',.5,1000.,8),('width',.25,1000.,8),
                                            ('cutoff',.5,1500.,8),('quadrature',.5,1000.,16)]:
        start=perf_counter()
        batch=shell_estimate(absorptions=[.02,.07,.14],radii_m=[20.,80.,128.],
            widths_m=width,photons_per_batch=4000,batches=32,max_path_m=max_path,
            seed=20260917,quadrature_order=quadrature)
        mean,se=ratio_with_standard_error(batch)
        runs[name]={'seconds':perf_counter()-start,'width_m':width,'max_path_m':max_path,
                    'quadrature_order':quadrature,'mean_multiple':mean[:,:,2].tolist(),
                    'se_multiple':se[:,:,2].tolist()}
    reference=np.array(runs['base']['mean_multiple']);se=np.array(runs['base']['se_multiple'])
    for name in ['width','cutoff','quadrature']:
        difference=np.array(runs[name]['mean_multiple'])-reference
        runs[name]['max_abs_difference']=float(np.max(abs(difference)))
        runs[name]['max_difference_over_reference_SE']=float(np.max(abs(difference)/se))
    (out/'mc-resolution.json').write_text(json.dumps(runs,indent=2))
    print(json.dumps(runs,indent=2))


if __name__=='__main__':main()
