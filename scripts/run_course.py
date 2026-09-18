#!/usr/bin/env python3
"""Execute the course in separate kernels, using this script's Python.

Source notebooks remain untouched. Outputs and machine-readable evidence go to
.build/course by default. No network or private provider is needed. Notebook 7
uses Numba when installed, otherwise reports its optional Monte Carlo omission.
"""
from __future__ import annotations
import argparse
import html
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='.build/course')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Per-cell timeout in seconds')
    parser.add_argument('--select', nargs='*',
                        help='Notebook stems or filenames; default is the full series')
    args=parser.parse_args()
    try:
        import nbformat
        from nbclient import NotebookClient
        from nbconvert import HTMLExporter
        from jupyter_client import KernelManager
        from jupyter_client.kernelspec import KernelSpecManager
    except ImportError as exc:
        raise SystemExit("Install the course extra: python -m pip install -e '.[notebook,accelerate]'\n"+str(exc))
    root=Path(__file__).resolve().parents[1]
    out=Path(args.output).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    files=sorted((root/'notebooks/course').glob('[0-9][0-9]_*.ipynb'))
    if args.select:
        wanted={Path(x).stem for x in args.select}
        known={p.stem for p in files}
        if wanted-known:raise SystemExit(f'Unknown notebooks: {sorted(wanted-known)}')
        files=[p for p in files if p.stem in wanted]
    if not files:raise SystemExit('No notebooks found')
    report={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'base_commit':'3ba25b711408f65ed2511f4abe5fbb0ee0d4457c',
            'notebooks':[], 'source_notebooks_modified':False}
    # Private temporary kernelspec avoids registering or changing a user's kernel.
    with tempfile.TemporaryDirectory(prefix='lighthit-kernel-') as td:
        specdir=Path(td)/'lighthit-course';specdir.mkdir()
        (specdir/'kernel.json').write_text(json.dumps({
            'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'LightHit course temporary kernel','language':'python'}))
        specmanager=KernelSpecManager(kernel_dirs=[td])
        for path in files:
            print(f'Execute {path.name}',flush=True)
            nb=nbformat.read(path,as_version=4)
            km=KernelManager(kernel_name='lighthit-course',kernel_spec_manager=specmanager)
            start=perf_counter(); status='ok';error=None
            try:
                client=NotebookClient(nb,km=km,timeout=args.timeout,
                                      resources={'metadata':{'path':str(root)}},
                                      allow_errors=False)
                client.execute(cwd=str(root))
            except Exception as exc:
                status='failed';error=f'{type(exc).__name__}: {exc}'
            finally:
                if km.has_kernel:km.shutdown_kernel(now=True)
            elapsed=perf_counter()-start
            dst=out/path.name;nbformat.write(nb,dst)
            exporter=HTMLExporter()
            body,_=exporter.from_notebook_node(nb)
            (out/(path.stem+'.html')).write_text(body,encoding='utf-8')
            cells=sum(c.cell_type=='code' for c in nb.cells)
            report['notebooks'].append({'name':path.name,'status':status,'seconds':elapsed,
                                         'code_cells':cells,'error':error})
            (out/'execution.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print(f'  {status}: {cells} cells, {elapsed:.2f} s',flush=True)
            if error:
                print(error,file=sys.stderr);return 1
    report['total_seconds']=sum(r['seconds'] for r in report['notebooks'])
    (out/'execution.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    links='\n'.join(f'<li><a href="{html.escape(Path(r["name"]).stem)}.html">'
                    f'{html.escape(r["name"])}</a> — {r["seconds"]:.2f} с</li>'
                    for r in report['notebooks'])
    (out/'index.html').write_text('''<!doctype html><html lang="ru"><meta charset="utf-8">
<title>LightHit: учебный курс</title><style>body{max-width:1000px;margin:3rem auto;
font:18px/1.6 system-ui;padding:0 1rem}h1{line-height:1.2}li{margin:.7rem 0}
code{font-size:.9em}</style><h1>LightHit: от баланса фотонов к отклику детектора</h1>
<p>Восемь исполняемых ноутбуков. Все примеры используют открытые искусственные
параметры; приватная оптика и реальные события не загружаются.</p>
<p>Выводы и рисунки ниже получены исполнением исходных ячеек. Интерактивное
изменение параметров требует запуска .ipynb в локальном Jupyter.</p><ol>'''+links+
'</ol><p><a href="execution.json">Протокол исполнения</a>. '
'Для исходников и команд см. notebooks/course/README.md в патче.</p></html>',encoding='utf-8')
    print(f'HTML index: {out / "index.html"}')
    return 0


if __name__=='__main__':raise SystemExit(main())
