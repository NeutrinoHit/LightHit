"""Charge-only OM batches on the same finite Q1 groups and interpolation.

Python validates optical/source inputs once per spectral unit. The compiled
loop retains receiver-dependent parent selection and the full angular rule;
it removes repeated Python entry and histogram-error work from screening.
"""
import numpy as np
from numba import njit
from ._receiver_first_groups import prepare


@njit(cache=True,nogil=True)
def screen_batch(positions,looks,origins,targets,source_arguments,evaluate,lookup_arguments):
    charge=np.empty(len(targets));counts=np.empty(len(targets),np.int64);samples=np.empty(len(targets),np.int64)
    for j in range(len(targets)):
        om=targets[j]
        groups=prepare(positions[om],looks[om],origins[om],*source_arguments)
        charge[j]=evaluate(groups[0],groups[1],groups[4],*lookup_arguments).sum()
        counts[j]=len(groups[0]);samples[j]=groups[5]
    return charge,counts,samples
