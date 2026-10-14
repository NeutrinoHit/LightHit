"""Bounded local angular/time source compression; not a transport backend.

Rank is local to a spatial point and learned from its own source coefficients.
The RTE cache is unchanged. Reconstructed arrays only validate accuracy; they
do not accelerate application. A compressed contraction is a separate task.
"""
from time import process_time
import numpy as np


def charge_preserving_ranks(channels,ranks):
    """Yield rank, reconstructed coefficients, and source-error diagnostics.

    C[:,w]=C[:,0] phi(w)+R[:,w], with phi(0)=1 and R[:,0]=0.
    Truncate the SVD of R, then restore column zero exactly. Thus each angular
    stationary coefficient survives; conservation holds after any linear RTE.
    No bound on waveform/receiver-relative error follows from source SVD error.
    """
    original=np.asarray(channels)
    if original.ndim!=3 or not np.isfinite(original).all():
        raise ValueError('finite point/channel/frequency array required')
    choices=list(ranks)
    if any(isinstance(r,bool) or not isinstance(r,(int,np.integer)) or r<1 for r in choices):
        raise ValueError('positive integer ranks required')
    start=process_time();c=original.astype(np.complex128)
    stationary=c[:,:,0]
    norm=np.sum(abs(stationary)**2,axis=1)
    phase=np.einsum('pc,pcw->pw',stationary.conj(),c)/np.where(norm>0,norm,1.)[:,None]
    phase[:,0]=1.
    base=stationary[:,:,None]*phase[:,None,:]
    residual=c-base;residual[:,:,0]=0.
    u,s,v=np.linalg.svd(residual,full_matrices=False)
    preparation_cpu=process_time()-start
    for rank in choices:
        start=process_time();keep=min(rank-1,len(s[0]))
        approx=base+(u[:,:,:keep]*s[:,None,:keep])@v[:,:keep,:]
        approx[:,:,0]=stationary
        approx=approx.astype(original.dtype)
        difference=abs(approx-original)
        yield int(rank),approx,dict(preparation_cpu_s=preparation_cpu,
            reconstruction_cpu_s=process_time()-start,
            source_relative_frobenius=float(np.linalg.norm(difference)/max(np.linalg.norm(original),1e-30)),
            stationary_bitwise_equal=bool(np.array_equal(approx[:,:,0],original[:,:,0])),
            compressed_application_measured=False)
