"""Publish complete construction checkpoints without exposing partial files."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import numpy as np


def savez(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='table-',dir=path.parent) as directory:
        temporary=Path(directory)/'checkpoint.npz'
        np.savez_compressed(temporary,**arrays)
        temporary.replace(path)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='table-',dir=path.parent) as directory:
        temporary=Path(directory)/'checkpoint.json'
        temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        temporary.replace(path)
