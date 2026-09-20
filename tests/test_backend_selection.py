"""Default compatibility and clear failures without the optional dependency."""
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from lighthit import KernelConfig, PointGreenSolver, synthetic_medium
from lighthit.angular import _free_moments_and_ratios


def test_backend_name_is_validated():
    with pytest.raises(ValueError, match="angular_backend"):
        PointGreenSolver(synthetic_medium(), angular_backend="typo")
    with pytest.raises(ValueError, match="angular_backend"):
        _free_moments_and_ratios([1.], 0.1, 2, backend="typo")
    with pytest.raises(ValueError, match="angular_backend"):
        KernelConfig(angular_backend="typo")


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


def test_public_build_auto_falls_back_without_numba():
    """The base ``pip install lighthit`` quick start must not require Numba."""
    code = '''
import sys
sys.modules["numba"] = None
import numpy as np
import lighthit as lh

medium = lh.SpectralMedium(
    [400., 500.], [.03, .04], [.02, .02], [1.34, 1.33], [1.38, 1.37], g=.9)
detector = lh.DetectorArray(
    [[10., 0., 0.]], [[-1., 0., 0.]], .05,
    lambda x: np.ones_like(np.asarray(x, float)),
    lambda w: np.full_like(np.asarray(w, float), .2))
source = lh.IsotropicFlash.monochromatic([0., 0., 0.], 1e5, 450.)
config = lh.KernelConfig(
    omega_per_ns=np.array([0.]), relative_time_edges_ns=np.array([0., 10.]),
    wavelength_nodes=2, scattering_degree=2, source_degree=2,
    k_max_per_m=.4, k_panel_per_m=.2, k_order=4,
    radial_range_m=(5., 15.), radial_nodes=4)
kernel = lh.build(medium, detector, source, config=config)
assert kernel.config.angular_backend == "auto"
assert kernel.resolved_angular_backend() == "numpy"
response = kernel.transport(source)
assert response.charge_pe[0] > 0
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run([sys.executable, "-c", code], env=env, check=True,
                   capture_output=True, text=True)


def test_single_auto_falls_back_without_numba():
    code = '''
import sys
sys.modules["numba"] = None
from lighthit import synthetic_medium
from lighthit.single import single_quadrature_backend, single_spectrum
assert single_quadrature_backend("auto") == "numpy"
value, error = single_spectrum([0.0], 20.0, None, synthetic_medium())
assert value[0].real > 0 and error >= 0
try:
    single_quadrature_backend("numba")
except ImportError as exc:
    assert "accelerate" in str(exc)
else:
    raise AssertionError("explicit missing Numba must be reported")
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run([sys.executable, "-c", code], env=env, check=True,
                   capture_output=True, text=True)
