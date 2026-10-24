"""The experimental arc CDF preserves its density and bin-L1 certificate."""
from pathlib import Path
import sys
import numpy as np
import pytest

pytest.importorskip('numba')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from harness.entry_cdf import make_basis
from harness.cases_entry_cdf import validate

POLY=np.array([.3082,.54192,.19831,-.04912])


def test_cdf_density_and_disjoint_bins_against_independent_arc_integrals():
    result=validate(POLY)
    h=result['headline']
    assert h['cdf_checks']>=150 and h['fallback_geometries']>0
    assert h['maximum_cdf_relative_error']<2e-6/64
    assert h['maximum_density_relative_error']<2e-6/64
    assert h['maximum_bin_l1_relative_error']<2e-6/64
    assert h['moment_bytes']==64


def test_a_different_acceptance_cannot_reuse_the_density_certificate():
    prepare,_,_=make_basis(POLY,1e-8)
    different=POLY.copy();different[0]*=.1
    _,scale,bound=prepare(.7,np.sqrt(.51),1.,0.,0.,.03,100.,.001,1e-5,different)
    assert scale<0 and np.isinf(bound)


def test_nonpositive_angular_polynomial_selects_exact_fallback():
    poly=np.array([.1,1.,0.,0.]);prepare,_,_=make_basis(poly,1e-8)
    _,scale,_=prepare(.7,np.sqrt(.51),1.,0.,0.,.03,100.,.001,1e-5,poly)
    assert scale<0


@pytest.mark.parametrize('tolerance',[0.,-1.,np.inf,np.nan])
def test_invalid_density_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError,match='positive density tolerance'):
        make_basis(POLY,tolerance)
