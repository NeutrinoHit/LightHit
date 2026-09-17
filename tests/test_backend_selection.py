"""Default compatibility and clear failures without the optional dependency."""
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from lighthit import PointGreenSolver, synthetic_medium
from lighthit.angular import _free_moments_and_ratios


def test_backend_name_is_validated():
    with pytest.raises(ValueError, match="angular_backend"):
        PointGreenSolver(synthetic_medium(), angular_backend="typo")
    with pytest.raises(ValueError, match="angular_backend"):
        _free_moments_and_ratios([1.], 0.1, 2, backend="typo")


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("k,d0,N", [([], 1, 2), ([-1], 1, 2), ([1], 0, 2),
    ([1], np.nan, 2), ([1], 1, -1), ([1], 1, True)])
def test_invalid_input_precedes_backend_import(k, d0, N, backend):
    with pytest.raises(ValueError):
        _free_moments_and_ratios(k, d0, N, backend=backend)


def test_default_import_and_numpy_work_without_numba():
    code = '''
import sys
sys.modules['numba'] = None
from lighthit.angular import _free_moments_and_ratios
b, r = _free_moments_and_ratios([0., 0.1], 0.2, 4)
assert 'lighthit._angular_numba' not in sys.modules
try:
    _free_moments_and_ratios([0., 0.1], 0.2, 4, backend='numba')
except ImportError as exc:
    assert 'accelerate' in str(exc)
else:
    raise AssertionError('missing Numba must be reported')
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
