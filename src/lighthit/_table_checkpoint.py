"""Bounded radial construction jobs with durable joined-array checkpoints.

Only table preparation uses this module. Joined numerical values retain their
original order; progress covers exact slices of mmap arrays, not approximations.
"""
from pathlib import Path
import hashlib,json,shutil,io
import numpy as np


def radial_blocks(count):
    """Two or three radii per job, independent of available worker count."""
    return np.array_split(np.arange(count),max(1,count//2))


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


class JoinedArrays:
    def __init__(self,output,shapes,contract,*,axis=0):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.path=self.output/'build-progress.json';self.axis=axis;self.arrays={}
        key=signature(dict(contract=contract,shapes=shapes,axis=axis,dtype='float64'))
        if self.path.exists():
            self.state=json.loads(self.path.read_text())
            if self.state['signature']!=key:raise ValueError('radial checkpoint inputs changed; choose a new directory')
            for name,shape in shapes.items():
                a=np.load(self.output/(name+'.npy'),mmap_mode='r+',allow_pickle=False)
                if a.shape!=tuple(shape) or a.dtype!=np.dtype(float):raise ValueError('radial checkpoint array differs')
                self.arrays[name]=a
        else:
            required=0
            for shape in shapes.values():
                header=io.BytesIO()
                np.lib.format.write_array_header_1_0(header,dict(descr=np.dtype(float).str,fortran_order=False,shape=tuple(shape)))
                required+=int(np.prod(shape,dtype=object))*np.dtype(float).itemsize+header.tell()
            free=shutil.disk_usage(self.output).free
            if required>free:
                raise OSError(f'joined arrays need {required} bytes; only {free} bytes available')
            self.state=dict(signature=key,completed={})
            for name,shape in shapes.items():
                self.arrays[name]=np.lib.format.open_memmap(self.output/(name+'.npy'),mode='w+',dtype=float,shape=shape)
            atomic_json(self.path,self.state)
        for item in self.state['completed'].values():
            if self._digest(item['indices'])!=item['sha256']:raise ValueError('joined radial checkpoint checksum differs')

    def _digest(self,indices):
        h=hashlib.sha256()
        for name,array in sorted(self.arrays.items()):
            h.update(name.encode());h.update(np.ascontiguousarray(np.take(array,indices,axis=self.axis)).tobytes())
        return h.hexdigest()

    def contains(self,number,indices):
        old=self.state['completed'].get(str(number))
        if old is not None and old['indices']!=np.asarray(indices).tolist():raise ValueError('radial checkpoint partition changed')
        return old is not None

    def merge(self,number,indices,values,*,metadata=None):
        indices=np.asarray(indices)
        for name,array in self.arrays.items():
            selection=[slice(None)]*array.ndim;selection[self.axis]=indices
            array[tuple(selection)]=values[name];array.flush()
        self.state['completed'][str(number)]=dict(indices=indices.tolist(),sha256=self._digest(indices),metadata=metadata)
        atomic_json(self.path,self.state)

    def metadata(self,number):return self.state['completed'][str(number)]['metadata']

    def discard(self,path):
        """Remove only a merged job's derived checkpoint inside this output."""
        path=Path(path).resolve();parent=(self.output/'blocks').resolve()
        if not path.is_relative_to(parent):raise ValueError('checkpoint cleanup escapes block directory')
        shutil.rmtree(path)
