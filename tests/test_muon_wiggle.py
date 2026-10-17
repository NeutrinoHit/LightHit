"""Physics-probe guards: conserved source and spectral crossing ownership."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from lighthit import SpectralLightElements
from lighthit.model import SpectralMedium,DetectorArray

import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/shower_basis'))   # the probe imports its paths helper
spec=importlib.util.spec_from_file_location('muon_wiggle_probe',Path(__file__).parents[1]/'scripts/shower_basis/probe_muon_wiggle.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def straight():
    count=10;start=np.column_stack((np.zeros(count),np.zeros(count),np.arange(count)*10.))
    return SpectralLightElements(start,np.tile([0.,0.,1.],(count,1)),np.full(count,10.),np.full(count,10.),np.full(count,-10.),np.ones(count),np.arange(count)*10/.299792458,np.arange(1,count+1)*10/.299792458,np.arange(count),np.zeros(count),cone_model='spectral')


def test_continuous_polyline_preserves_spectral_fields_and_clock():
    s=straight();fit,_=probe.fit_pieces(s,3)
    np.testing.assert_allclose(fit.coefficient0.sum(),s.coefficient0.sum())
    np.testing.assert_allclose(fit.coefficient2.sum(),s.coefficient2.sum())
    np.testing.assert_allclose(fit.start_m[1:],fit.start_m[:-1]+fit.length_m[:-1,None]*fit.direction[:-1])
    np.testing.assert_allclose(fit.start_ns[1:],fit.end_ns[:-1])
    assert fit.start_ns[0]==s.start_ns[0] and fit.end_ns[-1]==s.end_ns[-1]


def test_spectral_interval_integral_invariant_under_straight_step_partition():
    s=straight();fit,_=probe.fit_pieces(s,1)
    med=SpectralMedium(np.array([350.,610.]),np.array([.01,.012]),np.array([.02,.018]),np.array([1.4,1.3]),np.array([1.42,1.32]))
    det=DetectorArray(np.array([[5.,0.,55.],[20.,0.,75.]]),np.array([[0.,0.,1.],[0.,0.,1.]]),1.,lambda x:np.ones_like(x),lambda x:np.ones_like(x))
    k=SimpleNamespace(medium=med,detector=det,acceptance=lambda: {'alpha':np.array([4*np.pi])})
    a=probe.interval_atoms(k,s,32);b=probe.interval_atoms(k,fit,32)
    qa=np.bincount(a[0],weights=a[2],minlength=2);qb=np.bincount(b[0],weights=b[2],minlength=2)
    np.testing.assert_allclose(qa,qb,rtol=1e-12)
    for j in range(2):
        np.testing.assert_allclose(np.average(a[1][a[0]==j],weights=a[2][a[0]==j]),np.average(b[1][b[0]==j],weights=b[2][b[0]==j]),rtol=1e-12)
