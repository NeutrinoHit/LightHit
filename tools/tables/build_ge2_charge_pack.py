#!/usr/bin/env python3
"""Exact compact omega-zero companion to an immutable full RTE pack.

The full tensor is referenced by relative path; only the stationary slice is
written. All source/medium keys and numerical coefficients remain unchanged.
"""
import argparse,copy,hashlib,json,os
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('fresh output required')
    source=args.source.resolve();out=args.output.resolve();header=json.loads((source/'pack.json').read_text())
    out.mkdir(parents=True);result=copy.deepcopy(header);result['bundles']=[];audit=[]
    for index,item in enumerate(header['bundles']):
        if len(item['meta']['omega'])!=65:continue
        original=source/item['file'];full=np.load(original,mmap_mode='r')
        alias=copy.deepcopy(item);alias['file']=os.path.relpath(original,out);result['bundles'].append(alias)
        compact=copy.deepcopy(item);compact['file']=f'charge-{index}.npy'
        compact['key'][2]=[0];compact['meta']['omega']=[0.]
        values=full[...,:1].copy()
        if not np.isfinite(values).all():raise ValueError('nonfinite stationary coefficients')
        np.save(out/compact['file'],values);result['bundles'].append(compact)
        loaded=np.load(out/compact['file'],mmap_mode='r')
        np.testing.assert_array_equal(loaded,full[...,:1])
        audit.append(dict(original_file=str(original),full_bytes=int(full.nbytes),stationary_bytes=int(values.nbytes),
            stationary_sha256=hashlib.sha256((out/compact['file']).read_bytes()).hexdigest(),
            original_stat=dict(size=original.stat().st_size,mtime_ns=original.stat().st_mtime_ns)))
    if not audit:raise ValueError('no65-frequency bundle found')
    (out/'pack.json').write_text(json.dumps(result,indent=2))
    report=dict(source_pack_sha256=hashlib.sha256((source/'pack.json').read_bytes()).hexdigest(),
        source=str(source),event_input=False,exact_slice=True,audit=audit)
    (out/'charge-pack-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
