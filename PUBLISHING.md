# PyPI release checklist

LightHit is packaged as `0.2.0a6`, under BSD-3-Clause
(`LICENSE`, `Copyright (c) 2026, Dmitry Naumov`), declared in
`pyproject.toml` as `license = {text = "BSD-3-Clause"}` with the matching OSI
classifier. Raising the build requirement to `setuptools>=77` would allow the
PEP 639 SPDX form and `license-files`; that is a follow-up, not a blocker.

## Preflight

```bash
LIGHTHIT_G4_FILE=g4_data/sim_e_100GeV_10.h5 python -m pytest -q
quarto render docs
python -m build --sdist --wheel
python -m twine check dist/lighthit-0.2.0a6*
```

Both forms of the build command must work --- with isolation, where the
backend is installed fresh, and `--no-isolation`, where it is the one in the
current environment --- and neither may need `PYTHONPATH`. The distribution
filter lives in `packaging_filter.py` beside `setup.py`, because it decides
what the package contains and must not be inside it; under PEP 517
`pyproject_hooks` runs the backend from a helper script of its own, so
`sys.path[0]` is that helper's directory and a bare import of the filter fails
exactly where it matters. `setup.py` puts its own directory on `sys.path`
first, and `tests/test_packaging.py` reproduces the hook-style invocation with
`PYTHONPATH` removed so the import cannot be rescued by the environment.

`python -m build` does **not** start from a clean tree, and `build_py` copies
without ever removing. A checkout whose `build/lib` predates the distribution
filter therefore used to hand every experimental module to the wheel while the
sdist stayed correct. `packaging_filter.ProductionBuildPy` now prunes
`build_lib` of anything the current build would not produce, so `rm -rf build`
is no longer load-bearing; `tests/test_packaging.py` plants a forbidden module
in a build tree and asserts that it is gone, and runs
`python -m build --no-isolation --sdist --wheel` on a copy of the project with
a deliberately dirty `build/lib`, inspecting both archives (that last test
skips where the `build` frontend is not installed). Keep
`packaging_filter.REQUIRED_EXPERIMENTAL` and the `MANIFEST.in` allowlist equal
--- a test checks that too.

Inspect both archives before upload. They must not contain `g4_data`, the
private `bgvd_model`, caches, results, docs, notebooks, slides, scripts or
tests, and the `lighthit/experimental` directory must hold exactly the seven
modules of the allowlist:

```bash
python -m zipfile -l dist/lighthit-0.2.0a6-*.whl | grep experimental/
tar tzf dist/lighthit-0.2.0a6.tar.gz | grep experimental/
```

The sdist intentionally includes `LICENSE`, `setup.py`, `packaging_filter.py`
and only three `examples/production_*.py` files in addition to build metadata
and package sources.

## TestPyPI, then PyPI

```bash
python -m twine upload --repository testpypi dist/lighthit-0.2.0a6*
# Test in a clean environment using the exact uploaded version.
python -m twine upload dist/lighthit-0.2.0a6*
```

Credentials/tokens are supplied by the publisher at upload time and must never
be stored in this repository. The separately installed private `bgvd_model` is
a runtime input, not a LightHit dependency or distribution asset.
