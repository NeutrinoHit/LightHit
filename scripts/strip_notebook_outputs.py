"""Remove notebook outputs and execution counts before a public Git commit."""
import argparse
from pathlib import Path
import nbformat
p=argparse.ArgumentParser()
p.add_argument('notebooks',nargs='+',type=Path)
a=p.parse_args()
for path in a.notebooks:
    nb=nbformat.read(path,as_version=4)
    for cell in nb.cells:
        if cell.cell_type=='code':
            cell.outputs=[]
            cell.execution_count=None
    nbformat.write(nb,path)
    print(path)
